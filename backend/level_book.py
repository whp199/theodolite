"""The level book (`backend/level_book.py`): the ledger of the weight-edit kernel, one line a call, each line chained
to the one before.

What it is. A file that is only ever added to. Every call of `levelling.read` writes one line of JSON into it, whole:
the plan as it was run, the edit, the items with their text and ids, every row's figures, and how the reading ended.
A reading that was refused or came out void is entered like any other, so what was tried cannot later be told apart
from what was kept by leaving lines out of the middle. A `note` is a line that is not a reading (the card check
enters its map of row counts as one). Each line carries the sha256 of the line before it, so a line changed, dropped,
moved or put in by hand before the last one shows when the chain is walked (`verify`). The chain does not fix where
the book ENDS: see "What the chain does not show".

Where. `$THEODOLITE_LEVEL_BOOK`, else `${XDG_DATA_HOME:-~/.local/share}/theodolite/level_book.jsonl`: outside every
checkout, beside the saved edits. A book this module makes is made with mode 0o600; one that already exists keeps
the mode it has, and its folder the mode the process gives a new folder. It keeps the items' text and every per-item
row, which is right for the owner's own session and is to be looked at again before anything but that session is
given a way to read it. The checks point the variable into a temporary folder, and so does a stand-in run unless the
variable is already set (it then writes the book the variable names, and its first printed line says so); the card
check keeps a book of its own beside its JSON.

A line.

    {"v": 1, "seq": <0, 1, 2, ...>, "at": "<UTC to the second>", "kind": "paired" | "written" | "note",
     "prev": <the "hash" of the line before; 64 zeros for line 0>, "body": {...}, "hash": "<64 hex>"}

`hash` is the sha256 of the canonical JSON of the entry without its own `hash` (`canonical`: keys sorted, no spaces,
ASCII, no NaN), and the line written is the canonical JSON of the whole entry and a newline. Two worked values: the
entry {"v": 1, "seq": 0, "at": "2026-11-14T22:00:00Z", "kind": "note", "prev": 64 zeros, "body": {"says": "first
line"}} hashes to 81318d96758c7187ba3ed1ad33e101f36e8d983aaae04190ebb680ae6d07317a, and the next, {"v": 1, "seq": 1,
"at": "2026-11-14T22:00:01Z", "kind": "note", "prev": that hash, "body": {"x": 0.1, "n": None}}, to
75969610dd2dc42dd9d49f7ca703b72dbaf010f705085e4e158fe82e9a1ba5c1. Lines count from 0, as layers do.

How a line is written (`append`). The file is opened for appending and locked (`fcntl.flock`, exclusive), its last
line is read and must be a whole entry, the new line is written in one `os.write`, and the file is synced before the
lock is let go. Two processes that append at once therefore each land a line and the chain stays whole. Nothing here
rewrites, truncates or moves the file. A last line that is torn (no newline, not JSON, not an entry, a hash that is
not its own, or bytes that are not those its content is written as) is never written over: `append` and `head` raise
`BookBroken` and say which line, and a person looks at the file.

What it refuses. A kind that is not one of `KINDS`, and a body that is not a dict of plain JSON values (dict with
string keys, list, str, int, float, bool, None, by exact type: a numpy scalar, a tuple and a number that is not finite
are refused, and the words say where in the body it sits). Both are a `ValueError` and nothing is written.

What the chain does not show. Nothing is signed: whoever rewrites a line and every line after it has a file that
verifies. The chain is there for accidents and for a quiet change to a line, not against someone who sets out to
rewrite the record. Lines left off the END, or the last line rewritten with its own hash worked again, leave a book
that verifies, and `append` goes on from there: the head a run prints when it ends (how many lines, the last hash) is
the only record of where the end was, and it is kept nowhere by this module. Nor does the chain show that a line's
body is true, only that it is the body that was written.

What was run, and what was not. Nothing here was measured on the card, and it rests on no pilot. On the CPU, 14
November 2026 [ran], in temporary folders only (and, by a reviewer's probe the same day: a book cut at its end, or
with its last line rewritten and rehashed, verifies): the two worked hashes above, by `entry_hash` and through `append`
with the clock set; append, head and verify on books of up to 2,000 lines (4.5 MB: the head in under a millisecond,
the walk in about 30); a byte changed in a middle line named by its line (also one that leaves the JSON's meaning
alone: an exponent's e written E); a line dropped, doubled, swapped or put in by hand; seven kinds of torn last line,
each refused with the file left as it was; two writers at once, 200 lines each, all 400 landing in one whole chain.
Those two writers were two threads of one process, each append opening the file afresh: `flock` is on the open
file, so they contend as two processes do, but two processes themselves are scripts/check_levelling.py's to run.
Not tried: a book on a network file system (where `flock` may not lock), and a book of many thousands of long lines
(`head` reads the last line only, but `entries` and `verify` walk the file from its first line).

Provisional: the name. "Level book" is the owner's to choose; it sits in this file's name, in one environment
variable and in docstrings, and in nothing a page shows.

Standard library only. This module imports nothing of the kernel's other modules, and they may import it freely.
"""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

VERSION = 1
BOOK_ENV = "THEODOLITE_LEVEL_BOOK"
GENESIS = "0" * 64
KINDS = ("paired", "written", "note")

_KEYS = ("v", "seq", "at", "kind", "prev", "body", "hash")
_BLOCK = 1 << 16  # bytes read at a time when the last line's beginning is looked for from the file's end


class BookBroken(RuntimeError):
    """The book cannot be read on from here, or cannot take a line: its words say which line and what to look at."""


def book_path() -> Path:
    """Where the book lives, read from the environment at every call (so a check or a dry run can point it elsewhere
    after this module is imported)."""
    env = os.environ.get(BOOK_ENV)
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base).expanduser() / "theodolite" / "level_book.jsonl"


def canonical(obj: Any) -> bytes:
    """The one way a value is written before it is hashed or put in the book: keys sorted, no spaces, ASCII, and a
    ValueError for NaN or an infinity."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def entry_hash(entry: Dict[str, Any]) -> str:
    """The sha256 of the entry without its "hash" key (one given is ignored)."""
    return hashlib.sha256(canonical({k: v for k, v in entry.items() if k != "hash"})).hexdigest()


def head(path: Optional[Path] = None) -> Tuple[int, str]:
    """(entries so far, the last entry's hash); (0, GENESIS) when there is no file or it is empty. Only the last line
    is read, so "entries so far" is the last line's seq + 1 and the lines before it are `verify`'s to walk. Raises
    `BookBroken` when the last line is not a whole entry whose hash is its own."""
    p = _path(path)
    try:
        fd = os.open(p, os.O_RDONLY)
    except FileNotFoundError:
        return 0, GENESIS
    try:
        return _head(fd, _size_at_rest(fd), p)
    finally:
        os.close(fd)


def append(kind: str, body: Dict[str, Any], *, path: Optional[Path] = None) -> Dict[str, Any]:
    """Write one line and return the entry written, as it is in the book (a copy: changing the body afterwards does
    not change it). Raises `BookBroken` as `head` does, or when the file took only part of the line; a `ValueError`
    for a kind that is not one of `KINDS` or a body that is not a dict of plain JSON values. In every such case but
    the short write, nothing was written."""
    if kind not in KINDS:
        raise ValueError(f"A book entry is one of {', '.join(KINDS)}; {kind!r} is none of them. Nothing was written.")
    try:
        bad = _unclean(body, "body") if type(body) is dict else "the body is not a dict"
    except RecursionError:
        bad = "the body is nested too deep to walk, or holds itself"
    if bad:
        raise ValueError("A book entry's body is a dict that holds only dict, list, str, int, float, bool and None (no "
                         f"numpy scalar, no tuple, no number that is not finite): {bad}. Nothing was written.")
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            seq, prev = _head(fd, os.fstat(fd).st_size, p)
            entry: Dict[str, Any] = {"v": VERSION, "seq": seq, "at": _now(), "kind": kind, "prev": prev, "body": body}
            entry["hash"] = entry_hash(entry)
            line = canonical(entry) + b"\n"
            took = os.write(fd, line)
            if took != len(line):
                raise BookBroken(f"The book took {took} of {len(line)} bytes of line {seq}: its last line is torn. "
                                 f"Look at {p} before anything more is read.")
            os.fsync(fd)
            if seq == 0:
                _sync_folder(p.parent)  # a book's first line: the folder's own record of the new file, too
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return json.loads(line)


def entries(path: Optional[Path] = None, *, start: int = 0, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """The book's entries in order, from line `start` (counting from 0), at most `limit` of them. It reads what was
    in the file when it began and does not walk the chain (`verify` does): a line that is not an entry at all raises
    `BookBroken`, naming it."""
    p = _path(path)
    if limit is not None and limit <= 0:
        return
    given = 0
    for i, raw in _lines(p):
        if i < start:
            continue
        entry, why = _parse(raw)
        if entry is None:
            raise BookBroken(f"Line {i} of the book is not a whole entry: {why}. Look at {p} before anything more "
                             "is read.")
        yield entry
        given += 1
        if limit is not None and given >= limit:
            return


def verify(path: Optional[Path] = None) -> Dict[str, Any]:
    """Walk every line: it ends in a newline, parses, has exactly the seven keys, its `v` is this module's and its
    `kind` one of `KINDS`, its `seq` is its line's number, its `prev` is the hash of the line before (64 zeros for line
    0), its `hash` is its own, and its bytes are those its content is written as. Returns {"ok", "entries", "head",
    "first_bad", "why"}: with a whole book, how many lines and the last one's hash; with a bad one, the first line
    that fails (`first_bad`), why in plain words, and `entries` and `head` for the whole lines before it. It never
    raises for a bad book, nor for one that cannot be read (a folder where the book should be). It does not show a
    book cut at its end: see the module's "What the chain does not show"."""
    before, n = GENESIS, 0
    try:
        for i, raw in _lines(_path(path)):
            entry, why = _parse(raw)
            if entry is not None:
                if type(entry["seq"]) is not int or entry["seq"] != i:
                    why = f"its seq is {entry['seq']!r} and it is line {i}: lines count from 0 with none missing"
                elif entry["prev"] != before:
                    why = ("its prev is not the hash of the line before" if i
                           else "its prev is not 64 zeros, as line 0's is")
                else:
                    why = _not_own(entry, raw)
            if why is not None:
                return {"ok": False, "entries": n, "head": before, "first_bad": i, "why": why}
            before, n = entry["hash"], n + 1
    except OSError as exc:  # a folder where the book should be, a file that may not be read
        return {"ok": False, "entries": n, "head": before, "first_bad": n,
                "why": f"the book cannot be read from here ({exc.strerror or type(exc).__name__})"}
    return {"ok": True, "entries": n, "head": before, "first_bad": None, "why": None}


# ------------------------------------------------------------------------------------------------ private helpers

def _path(path: Optional[Path]) -> Path:
    return Path(path) if path is not None else book_path()


def _now() -> str:
    """UTC to the second, "2026-11-14T22:00:00Z"."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sync_folder(folder: Path) -> None:
    """Sync a folder, so a file just made in it outlasts a power cut. Where a file system will not sync a folder
    the line itself is still synced, and that is all that is lost."""
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _unclean(obj: Any, where: str) -> Optional[str]:
    """None when `obj` holds only plain JSON values (by exact type, so a numpy float64, a float's subclass, is not
    one), else where the first one that is not sits and what it is."""
    t = type(obj)
    if obj is None or t is bool or t is int or t is str:
        return None
    if t is float:
        return None if math.isfinite(obj) else f"{where} is {obj!r} (a number that is not finite is written as None)"
    if t is list:
        for i, x in enumerate(obj):
            bad = _unclean(x, f"{where}[{i}]")
            if bad:
                return bad
        return None
    if t is dict:
        for k, x in obj.items():
            if type(k) is not str:
                return f"{where} has a key that is not a string ({k!r})"
            bad = _unclean(x, f"{where}.{k}")
            if bad:
                return bad
        return None
    return f"{where} is a {t.__module__}.{t.__qualname__}"


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")  # json.loads would read NaN and Infinity; no line of the book holds one


def _parse(raw: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """One line as it is in the file, its newline included when it has one: (the entry, None), or (None, why it is
    not one). Its hash and its chain are not looked at here."""
    if not raw.endswith(b"\n"):
        return None, "it does not end in a newline"
    try:
        entry = json.loads(raw, parse_constant=_no_constant)
    except (ValueError, RecursionError):
        return None, "it is not JSON"
    if type(entry) is not dict or set(entry) != set(_KEYS):
        return None, f"it has not exactly the seven keys of an entry ({', '.join(_KEYS)})"
    if type(entry["v"]) is not int or entry["v"] != VERSION:
        return None, f"its v is {entry['v']!r}, and this book's lines are of v {VERSION}"
    if entry["kind"] not in KINDS:
        return None, f"its kind is {entry['kind']!r}, which is none of {', '.join(KINDS)}"
    return entry, None


def _not_own(entry: Dict[str, Any], raw: bytes) -> Optional[str]:
    """None when the entry's hash is its own and the line's bytes are those its content is written as, else which
    fails. The second catches a byte changed where the JSON's meaning is not (an exponent's e written E)."""
    if entry["hash"] != entry_hash(entry):
        return "its hash is not its own"
    if canonical(entry) + b"\n" != raw:
        return "its bytes are not those an entry of this content is written as"
    return None


def _size_at_rest(fd: int) -> int:
    """The file's size at a moment no line is being written: taken under a shared lock, let go at once. The file only
    grows, so every byte below that size stays as it is."""
    fcntl.flock(fd, fcntl.LOCK_SH)
    try:
        return os.fstat(fd).st_size
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _pread(fd: int, n: int, at: int) -> bytes:
    parts = []
    while n > 0:
        got = os.pread(fd, n, at)
        if not got:
            break
        parts.append(got)
        n, at = n - len(got), at + len(got)
    return b"".join(parts)


def _last_line(fd: int, size: int) -> Tuple[int, bytes]:
    """(where the last line begins, its bytes), of the first `size` bytes of the file (`size` above 0). The last
    byte is the line's own newline when the line is whole, so the newline before it is looked for from there back."""
    stop, start = size - 1, 0
    while stop > 0:
        lo = max(0, stop - _BLOCK)
        i = _pread(fd, stop - lo, lo).rfind(b"\n")
        if i >= 0:
            start = lo + i + 1
            break
        stop = lo
    return start, _pread(fd, size - start, start)


def _newlines(fd: int, upto: int) -> int:
    """How many lines end before byte `upto`: the number of the line that begins there. Read only to name a torn
    line."""
    n, at = 0, 0
    while at < upto:
        got = os.pread(fd, min(_BLOCK, upto - at), at)
        if not got:
            break
        n, at = n + got.count(b"\n"), at + len(got)
    return n


def _head(fd: int, size: int, p: Path) -> Tuple[int, str]:
    """`head` on an open file whose first `size` bytes are at rest."""
    if size == 0:
        return 0, GENESIS
    start, raw = _last_line(fd, size)
    entry, why = _parse(raw)
    if entry is not None:
        if type(entry["seq"]) is not int or entry["seq"] < 0:
            why = f"its seq is {entry['seq']!r}, which is no line's number"
        else:
            why = _not_own(entry, raw)
    if why is not None:
        raise BookBroken(f"The book's last line (line {_newlines(fd, start)}) is not a whole entry: {why}. Nothing "
                         f"was written. Look at {p} before anything more is read.")
    return entry["seq"] + 1, entry["hash"]


def _lines(p: Path) -> Iterator[Tuple[int, bytes]]:
    """(line number, the line's bytes with its newline) for what was in the file when the walk began; nothing when
    there is no file. A last line with no newline is given as it is."""
    try:
        f = open(p, "rb")
    except FileNotFoundError:
        return
    with f:
        left, i = _size_at_rest(f.fileno()), 0
        while left > 0:
            raw = f.readline(left)
            if not raw:
                break
            yield i, raw
            left, i = left - len(raw), i + 1
