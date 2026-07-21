"""
minirel.wal
============

An append-only write-ahead log providing crash durability: once a
transaction's COMMIT record has been written and fsynced, its effects
are guaranteed to be recoverable even if the process dies before the
affected data pages ever reach disk.

Scope, stated plainly (this is the honest-caveat companion to
transaction.py's MVCC docstring): this is *logical* redo logging, not
physical/ARIES-style logging. Each mutating operation is logged as a
high-level, replayable action ("insert this row into this table",
"mark this RID's xmax", ...) rather than as a raw before/after page
image. Recovery works by re-executing, in order, the operations of every
transaction that reached COMMIT since the last CHECKPOINT -- it does not
do fine-grained per-page LSN comparison against the data file the way a
production WAL does. That's a deliberate scope cut: it keeps the log
format and recovery algorithm easy to read end-to-end in one sitting
while still exercising the real idea (durability + atomicity from a
sequential log, verified by an actual kill-the-process-mid-transaction
test in test_wal_recovery.py), which is the point of building this by
hand instead of using a library.

On-disk record format, one after another with no separators (each
record is self-framing via its length prefix)::

    [type: u8][txn_id: u32][payload_len: u32][payload: JSON bytes]
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from enum import IntEnum

_HEADER = struct.Struct("<BII")  # type, txn_id, payload_len


class RecordType(IntEnum):
    BEGIN = 1
    OP = 2
    COMMIT = 3
    ABORT = 4
    CHECKPOINT = 5


@dataclass(frozen=True, slots=True)
class LogRecord:
    offset: int
    type: RecordType
    txn_id: int
    payload: dict


class WriteAheadLog:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            open(path, "wb").close()
        self._append_file = open(path, "ab")

    # -- writing -------------------------------------------------------------

    def _append(self, record_type: RecordType, txn_id: int, payload: dict) -> int:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8") if payload else b""
        offset = self._append_file.tell()
        self._append_file.write(_HEADER.pack(record_type, txn_id, len(body)))
        self._append_file.write(body)
        return offset

    def log_begin(self, txn_id: int) -> int:
        return self._append(RecordType.BEGIN, txn_id, {})

    def log_operation(self, txn_id: int, kind: str, **fields) -> int:
        payload = {"kind": kind, **fields}
        return self._append(RecordType.OP, txn_id, payload)

    def log_commit(self, txn_id: int) -> int:
        offset = self._append(RecordType.COMMIT, txn_id, {})
        self.flush()
        return offset

    def log_abort(self, txn_id: int) -> int:
        return self._append(RecordType.ABORT, txn_id, {})

    def log_checkpoint(self) -> int:
        offset = self._append(RecordType.CHECKPOINT, 0, {})
        self.flush()
        return offset

    def flush(self) -> None:
        """Force buffered writes to physical storage. Called after every
        COMMIT/CHECKPOINT -- those are exactly the durability points the log
        promises to honor, so they can't be left sitting in an OS buffer.
        """
        self._append_file.flush()
        os.fsync(self._append_file.fileno())

    def close(self) -> None:
        self._append_file.close()

    # -- reading / recovery ----------------------------------------------------

    def _iter_records(self, handle) -> list[LogRecord]:
        records = []
        while True:
            offset = handle.tell()
            header = handle.read(_HEADER.size)
            if len(header) < _HEADER.size:
                break  # clean EOF, or a torn write from a crash mid-append
            record_type, txn_id, payload_len = _HEADER.unpack(header)
            body = handle.read(payload_len)
            if len(body) < payload_len:
                break  # torn write: the last record never finished flushing
            payload = json.loads(body) if body else {}
            records.append(LogRecord(offset, RecordType(record_type), txn_id, payload))
        return records

    def read_all(self) -> list[LogRecord]:
        with open(self.path, "rb") as f:
            return self._iter_records(f)

    def read_since_last_checkpoint(self) -> list[LogRecord]:
        """Records to replay on recovery: everything after the most recent
        CHECKPOINT (or the whole log, if there isn't one). Records before a
        checkpoint don't need replaying because a checkpoint only gets
        written after the caller has flushed all dirty pages to disk --
        by definition, everything up to that point is already durable in
        the data file itself.
        """
        all_records = self.read_all()
        last_checkpoint = None
        for i, record in enumerate(all_records):
            if record.type == RecordType.CHECKPOINT:
                last_checkpoint = i
        if last_checkpoint is None:
            return all_records
        return all_records[last_checkpoint + 1 :]
