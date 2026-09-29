"""The page ``fourfloor serve`` hands the browser: it parses, and it is whole.

There is no JavaScript test runner in this project, so the app's behaviour is
checked in a real browser. These catch the two mistakes that break every
screen at once: a syntax error in ``app.js``, and ``app.js`` reaching for an
element that ``index.html`` does not have (``$('#toast')`` returning null).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "fourfloor" / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_app_js_parses():
    out = subprocess.run(["node", "--check", str(WEB / "app.js")],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr


def test_every_id_app_js_uses_is_on_the_page():
    js = (WEB / "app.js").read_text(encoding="utf8")
    html = (WEB / "index.html").read_text(encoding="utf8")
    used = set(re.findall(r"\$\('#([A-Za-z][\w-]*)", js))
    present = set(re.findall(r'\bid="([^"]+)"', html))
    present |= set(re.findall(r"\.id = '([^']+)'", js))     # built by the script
    missing = sorted(used - present)
    assert not missing, f"app.js looks up ids index.html does not have: {missing}"


def test_compare_hint_names_the_flip_key_the_page_uses():
    # Tab used to be the flip, which trapped keyboard focus on the compare
    # screen; the hint must not send anyone back to it
    html = (WEB / "index.html").read_text(encoding="utf8")
    hint = re.search(r'class="note ab-hint">(.*?)</p>', html, re.S).group(1)
    assert "<kbd>Tab</kbd>" not in hint
    assert "<kbd>F</kbd>" in hint
