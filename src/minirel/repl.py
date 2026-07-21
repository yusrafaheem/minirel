"""
minirel.repl
=============

A small interactive SQL shell: `minirel path/to.db` (installed as the
`minirel` console script via pyproject.toml). Statements are
newline-buffered and executed on a trailing `;`, auto-committing each
one -- there's no `BEGIN`/`COMMIT` support at the REPL level (the
Database/Transaction API supports it programmatically; wiring it into
the shell's grammar was left out to keep this file focused on being a
demo/inspection tool rather than a full client).
"""

from __future__ import annotations

import sys

from .database import Database, ExecuteResult

_DOT_COMMANDS = {".exit", ".quit", ".tables", ".help"}


def _format_rows(result: ExecuteResult) -> str:
    if not result.rows:
        return "(0 rows)"
    columns = list(result.rows[0].keys())
    col_widths = [
        max(len(col), max((len(str(row.get(col, ""))) for row in result.rows), default=0))
        for col in columns
    ]
    lines = []
    header = " | ".join(col.ljust(w) for col, w in zip(columns, col_widths))
    lines.append(header)
    lines.append("-+-".join("-" * w for w in col_widths))
    for row in result.rows:
        cells = (str(row.get(col, "")).ljust(w) for col, w in zip(columns, col_widths))
        lines.append(" | ".join(cells))
    lines.append(f"({len(result.rows)} row{'s' if len(result.rows) != 1 else ''})")
    return "\n".join(lines)


def run_statement(db: Database, sql: str) -> str:
    result = db.execute(sql)
    if result.rows is not None:
        return _format_rows(result)
    return result.message


def repl(db: Database, input_stream=None, output_stream=None) -> None:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    buffer = ""

    output_stream.write("minirel -- type SQL statements ending in ';', or .exit to quit\n")
    while True:
        prompt = "minirel> " if not buffer else "     ...> "
        output_stream.write(prompt)
        output_stream.flush()
        line = input_stream.readline()
        if line == "":  # EOF
            break
        stripped = line.strip()

        if not buffer and stripped in (".exit", ".quit"):
            break
        if not buffer and stripped == ".tables":
            output_stream.write("\n".join(sorted(db.catalog.tables)) + "\n")
            continue
        if not buffer and stripped == ".help":
            output_stream.write("Enter SQL ending in ';'. Dot-commands: .tables, .exit\n")
            continue

        buffer += line
        if ";" in line:
            statement, _, rest = buffer.partition(";")
            buffer = rest.strip()
            statement = statement.strip()
            if not statement:
                continue
            try:
                output_stream.write(run_statement(db, statement) + "\n")
            except Exception as exc:  # noqa: BLE001 - a REPL should report, not crash, on bad input
                output_stream.write(f"error: {exc}\n")

    output_stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: minirel <path-to-database-file>", file=sys.stderr)
        return 2
    db = Database(argv[0])
    try:
        repl(db)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
