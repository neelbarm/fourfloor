"""What a multipart upload cut off halfway leaves on disk: nothing."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from fourfloor import multipart
from tests.test_server import BOUNDARY, body_for


def test_an_upload_cut_off_mid_file_leaves_no_temporary(tmp_path: Path) -> None:
    """The part being streamed was never handed back, so nobody deleted it."""
    body = body_for([("file", "big.wav", b"x" * (3 * multipart.CHUNK))])
    cut = body[: len(body) // 2]                        # the browser gave up
    with pytest.raises(multipart.MultipartError):
        multipart.parse(io.BytesIO(cut), f"multipart/form-data; boundary={BOUNDARY}",
                        None, tmp_path, 8 << 20)
    assert not list(tmp_path.glob("*.part"))


def test_an_upload_over_the_limit_mid_file_leaves_no_temporary(tmp_path: Path) -> None:
    body = body_for([("note", None, b"hi"),
                     ("file", "big.wav", b"x" * (3 * multipart.CHUNK))])
    with pytest.raises(multipart.PayloadTooLarge):
        multipart.parse(io.BytesIO(body), f"multipart/form-data; boundary={BOUNDARY}",
                        None, tmp_path, 2 * multipart.CHUNK)
    assert not list(tmp_path.glob("*.part"))
