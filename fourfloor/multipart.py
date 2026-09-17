"""A streaming ``multipart/form-data`` parser for the local web app.

``cgi.FieldStorage`` was the stdlib answer to this and is gone in 3.13, so the
app parses uploads itself. The rules it follows are the ones that matter when
the payload is a 200 MB song:

* nothing is held in memory -- file parts are streamed to a temporary file in
  chunks, and only small non-file fields are buffered;
* the total is capped, and the cap is enforced while reading, not after, so an
  oversized or lying ``Content-Length`` cannot fill the disk;
* part headers are handed to :mod:`email.parser`, which already knows how to
  unquote ``filename="my song (1).mp3"`` and RFC 2231 continuations;
* a filename is data, never a path. Callers get the raw string to display and
  are expected to choose the name on disk themselves.
"""

from __future__ import annotations

import email.parser
import email.policy
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

CHUNK = 256 * 1024
MAX_HEADER_BYTES = 16 * 1024
MAX_FIELD_BYTES = 64 * 1024


class MultipartError(ValueError):
    """The body was not a multipart payload this parser can read."""


class PayloadTooLarge(MultipartError):
    """The body was longer than the caller allows."""


@dataclass
class Part:
    """One part of the payload: either a field or an uploaded file."""

    name: str
    filename: str | None = None
    content_type: str = "application/octet-stream"
    value: bytes = b""          # set for non-file parts
    path: Path | None = None    # set for file parts
    size: int = 0

    @property
    def is_file(self) -> bool:
        return self.path is not None

    @property
    def text(self) -> str:
        return self.value.decode("utf8", "replace")


def boundary_of(content_type: str) -> bytes:
    """Pull the boundary out of a ``Content-Type`` header."""
    msg = email.parser.Parser(policy=email.policy.HTTP).parsestr(
        f"Content-Type: {content_type}\n\n")
    if (msg.get_content_type() or "").lower() != "multipart/form-data":
        raise MultipartError(
            f"expected multipart/form-data, got {content_type.split(';')[0].strip()!r}")
    boundary = msg.get_param("boundary")
    if not boundary:
        raise MultipartError("multipart/form-data without a boundary")
    if isinstance(boundary, tuple):                     # RFC 2231 form
        boundary = boundary[2]
    boundary = str(boundary)
    if not 1 <= len(boundary) <= 200:
        raise MultipartError("multipart boundary is not a usable length")
    return boundary.encode("ascii", "strict")


class _Reader:
    """A capped chunk reader over the request body."""

    def __init__(self, stream: BinaryIO, length: int | None, limit: int) -> None:
        self.stream = stream
        self.remaining = length
        self.limit = limit
        self.read_total = 0

    def chunk(self) -> bytes:
        want = CHUNK if self.remaining is None else min(CHUNK, self.remaining)
        if want <= 0:
            return b""
        data = self.stream.read(want)
        if not data:
            self.remaining = 0
            return b""
        if self.remaining is not None:
            self.remaining -= len(data)
        self.read_total += len(data)
        if self.read_total > self.limit:
            raise PayloadTooLarge(
                f"upload is larger than the {self.limit // (1024 * 1024)} MB limit")
        return data


def parse(stream: BinaryIO, content_type: str, content_length: int | None,
          dest: Path, limit: int) -> list[Part]:
    """Parse a body into parts, spooling file parts into ``dest``.

    The caller owns everything written into ``dest`` and is responsible for
    moving or deleting it.
    """
    boundary = boundary_of(content_type)
    if content_length is not None and content_length > limit:
        raise PayloadTooLarge(
            f"upload is larger than the {limit // (1024 * 1024)} MB limit")
    dest.mkdir(parents=True, exist_ok=True)
    reader = _Reader(stream, content_length, limit)
    parts: list[Part] = []
    try:
        for part in _parts(reader, boundary, dest):
            parts.append(part)
    except BaseException:
        for p in parts:
            if p.path is not None:
                p.path.unlink(missing_ok=True)
        raise
    return parts


def _parts(reader: _Reader, boundary: bytes, dest: Path) -> Iterator[Part]:
    # Prefixing a CRLF makes the opening boundary look exactly like every other
    # one, so there is a single delimiter pattern to search for.
    delim = b"\r\n--" + boundary
    buf = b"\r\n"
    eof = False

    def fill() -> bool:
        nonlocal buf, eof
        if eof:
            return False
        data = reader.chunk()
        if not data:
            eof = True
            return False
        buf += data
        return True

    # Skip the preamble up to the first boundary. Only the tail that could hold
    # a delimiter split across two reads is carried over -- keeping the whole
    # preamble would mean buffering the body, and dropping it wholesale would
    # throw away the boundary that arrived in the same chunk.
    while (idx := buf.find(delim)) < 0:
        keep = len(delim) + 4
        if len(buf) > keep:
            buf = buf[-keep:]
        if not fill():
            raise MultipartError("no multipart boundary in the body")
    buf = buf[idx + len(delim):]

    while True:
        while len(buf) < 2 and fill():
            pass
        if buf[:2] == b"--":
            return                                   # closing boundary
        nl = buf.find(b"\r\n")
        while nl < 0 and fill():
            nl = buf.find(b"\r\n")
        if nl < 0:
            return                                   # truncated, nothing usable left
        buf = buf[nl + 2:]

        # headers
        while (end := buf.find(b"\r\n\r\n")) < 0:
            if len(buf) > MAX_HEADER_BYTES:
                raise MultipartError("multipart part headers are too long")
            if not fill():
                raise MultipartError("multipart part ended inside its headers")
        head, buf = buf[:end], buf[end + 4:]
        part = _part_from_headers(head)

        sink: BinaryIO | None = None
        if part.filename is not None:
            fd, tmp = tempfile.mkstemp(prefix="upload-", suffix=".part", dir=dest)
            part.path = Path(tmp)
            sink = open(fd, "wb")

        body = bytearray()
        try:
            while True:
                idx = buf.find(delim)
                if idx >= 0:
                    _emit(part, sink, body, buf[:idx])
                    buf = buf[idx + len(delim):]
                    break
                keep = len(delim) + 3
                if len(buf) > keep:
                    _emit(part, sink, body, buf[:-keep])
                    buf = buf[-keep:]
                if not fill():
                    raise MultipartError(
                        f"multipart part {part.name!r} ended before its boundary")
        finally:
            if sink is not None:
                sink.close()
        if sink is None:
            part.value = bytes(body)
        yield part


def _emit(part: Part, sink: BinaryIO | None, body: bytearray, data: bytes) -> None:
    if not data:
        return
    part.size += len(data)
    if sink is not None:
        sink.write(data)
        return
    if part.size > MAX_FIELD_BYTES:
        raise PayloadTooLarge(f"form field {part.name!r} is too long")
    body.extend(data)


def _part_from_headers(head: bytes) -> Part:
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(head + b"\r\n\r\n")
    disposition = msg.get("content-disposition")
    if not disposition:
        raise MultipartError("multipart part without a Content-Disposition header")
    name = msg.get_param("name", header="content-disposition")
    if name is None:
        raise MultipartError("multipart part without a name")
    filename = msg.get_param("filename", header="content-disposition")
    if isinstance(name, tuple):
        name = name[2]
    if isinstance(filename, tuple):
        filename = filename[2]
    return Part(
        name=str(name),
        filename=None if filename is None else str(filename),
        content_type=(msg.get_content_type() or "application/octet-stream"),
    )


def cleanup(parts: list[Part]) -> None:
    """Delete every temporary file a parse produced."""
    for p in parts:
        if p.path is not None:
            p.path.unlink(missing_ok=True)


def move(part: Part, target: Path) -> Path:
    """Move a spooled file part to ``target``, which the caller named."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if part.path is None:
        raise MultipartError(f"{part.name!r} is not a file part")
    shutil.move(str(part.path), str(target))
    part.path = target
    return target
