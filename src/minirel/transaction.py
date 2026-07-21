"""
minirel.transaction
=====================

Snapshot-isolation MVCC on top of the (xmin, xmax) tuple headers defined
in types.py, plus the WAL-replay logic that decides which logged
operations get re-applied during crash recovery.

Visibility rule (this is the real Postgres snapshot-isolation rule, not
a simplified stand-in): a row version with header (xmin, xmax) is
visible to a transaction with snapshot S if and only if::

    xmin was created by S's own transaction, OR xmin committed strictly
    before S started (i.e. xmin < S.xid) AND xmin wasn't one of the
    transactions still in-flight when S started (xmin not in
    S.active_xids)

    ...AND the same test, applied to xmax, is *false* (i.e. the version
    hasn't been deleted by a transaction visible to S) -- unless xmax is
    the "not deleted" sentinel.

Transactions whose outcome was ABORT are handled by keeping a permanent
`aborted_xids` set and treating any xmin/xmax stamped by one as if it
never happened, at every snapshot, forever -- which also means minirel
never physically reclaims an aborted transaction's rows (no undo
logging, no vacuum). That's consistent with the rest of the storage
layer's "no space reclamation for heap pages either" simplification,
and is called out again in the README.

Concurrent writers additionally get a first-committer-wins conflict
check: stamping xmax on a row that another still-live (non-aborted)
transaction has already stamped raises WriteConflictError instead of
silently clobbering it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .types import INFINITY_TXN
from .wal import LogRecord, RecordType, WriteAheadLog


class WriteConflictError(Exception):
    """Raised on a first-committer-wins conflict: two transactions tried to
    modify the same row version concurrently.
    """


class TxnState(str, Enum):
    ACTIVE = "active"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class Snapshot:
    xid: int
    active_xids: frozenset[int]

    def committed_before(self, other_xid: int) -> bool:
        return other_xid < self.xid and other_xid not in self.active_xids


@dataclass(slots=True)
class Transaction:
    txn_id: int
    snapshot: Snapshot
    manager: TransactionManager = field(repr=False)
    state: TxnState = TxnState.ACTIVE

    def commit(self) -> None:
        self.manager.commit(self)

    def abort(self) -> None:
        self.manager.abort(self)

    def log_operation(self, kind: str, **fields) -> None:
        self.manager.wal.log_operation(self.txn_id, kind, **fields)


class TransactionManager:
    """Single-threaded (no internal locking -- see README) transaction
    coordinator: hands out monotonically increasing transaction ids,
    tracks who's currently active for snapshot construction, and decides
    row visibility.
    """

    def __init__(
        self,
        wal: WriteAheadLog,
        next_txn_id: int = 1,
        initial_aborted_xids: set[int] | None = None,
    ):
        self.wal = wal
        self._next_txn_id = next_txn_id
        self._active_xids: set[int] = set()
        self._aborted_xids: set[int] = set(initial_aborted_xids or ())

    @property
    def has_active_transactions(self) -> bool:
        return bool(self._active_xids)

    def begin(self) -> Transaction:
        txn_id = self._next_txn_id
        self._next_txn_id += 1
        snapshot = Snapshot(xid=txn_id, active_xids=frozenset(self._active_xids))
        self._active_xids.add(txn_id)
        self.wal.log_begin(txn_id)
        return Transaction(txn_id=txn_id, snapshot=snapshot, manager=self)

    def commit(self, txn: Transaction) -> None:
        if txn.state != TxnState.ACTIVE:
            raise ValueError(f"cannot commit a transaction in state {txn.state}")
        self.wal.log_commit(txn.txn_id)
        self._active_xids.discard(txn.txn_id)
        txn.state = TxnState.COMMITTED

    def abort(self, txn: Transaction) -> None:
        if txn.state != TxnState.ACTIVE:
            raise ValueError(f"cannot abort a transaction in state {txn.state}")
        self.wal.log_abort(txn.txn_id)
        self._active_xids.discard(txn.txn_id)
        self._aborted_xids.add(txn.txn_id)
        txn.state = TxnState.ABORTED

    def is_visible(self, xmin: int, xmax: int, snapshot: Snapshot) -> bool:
        if xmin in self._aborted_xids:
            return False
        xmin_visible = xmin == snapshot.xid or snapshot.committed_before(xmin)
        if not xmin_visible:
            return False
        if xmax == INFINITY_TXN or xmax in self._aborted_xids:
            return True
        xmax_visible = xmax == snapshot.xid or snapshot.committed_before(xmax)
        return not xmax_visible

    def check_write_conflict(self, txn: Transaction, current_xmax: int) -> None:
        """Call before stamping xmax on a row this transaction wants to
        update/delete. Raises if some *other* non-aborted transaction has
        already claimed it (first-committer-wins).
        """
        if current_xmax == INFINITY_TXN:
            return
        if current_xmax == txn.txn_id:
            return
        if current_xmax in self._aborted_xids:
            return
        raise WriteConflictError(
            f"txn {txn.txn_id} cannot modify a row already modified by txn {current_xmax}"
        )


def committed_operations_since_checkpoint(wal: WriteAheadLog) -> list[LogRecord]:
    """The records recovery needs to replay: every OP record belonging to a
    transaction whose COMMIT record also appears (since the last
    checkpoint), in original log order. Operations from transactions that
    never committed (crashed mid-transaction, or explicitly aborted) are
    dropped -- that's what gives recovery its atomicity guarantee.
    """
    records = wal.read_since_last_checkpoint()
    committed_txns = {r.txn_id for r in records if r.type == RecordType.COMMIT}
    return [r for r in records if r.type == RecordType.OP and r.txn_id in committed_txns]


def uncommitted_txn_ids_since_checkpoint(wal: WriteAheadLog) -> set[int]:
    """Transactions that began (since the last checkpoint) but have no
    COMMIT record -- either they were explicitly aborted, or the process
    died mid-transaction. Either way, any data they wrote must be treated
    as permanently invisible on restart.

    This matters because the buffer pool is allowed to flush a dirty page
    to disk for *any* reason (LRU eviction under memory pressure), not
    just at commit -- minirel's WAL doesn't implement full ARIES-style
    no-steal page pinning. So a page containing an uncommitted insert can
    legitimately already be sitting in the data file when the process
    restarts. Seeding a fresh TransactionManager's aborted-set with these
    ids (see Database._recover) is what makes such a row correctly
    invisible instead of silently "becoming committed" by virtue of its
    txn id merely being less than every future transaction's.

    This is sound only because Database.checkpoint() refuses to run while
    any transaction is active (see its docstring) -- so every txn id that
    could show up here began *after* the checkpoint, and the checkpoint
    itself is the reason earlier history doesn't need this treatment.
    """
    records = wal.read_since_last_checkpoint()
    began = {r.txn_id for r in records if r.type == RecordType.BEGIN}
    committed = {r.txn_id for r in records if r.type == RecordType.COMMIT}
    return began - committed


def next_txn_id_after_recovery(wal: WriteAheadLog) -> int:
    """So a freshly restarted TransactionManager doesn't reuse a txn id that
    already appears in the log (which would be visible to old snapshots by
    coincidence of the visibility rule's `<` comparison).
    """
    max_seen = 0
    for record in wal.read_all():
        max_seen = max(max_seen, record.txn_id)
    return max_seen + 1
