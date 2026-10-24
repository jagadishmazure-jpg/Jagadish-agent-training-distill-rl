import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_brands  # noqa: E402


def test_brand_blocklist_is_not_empty():
    assert len(check_brands.TERM_HASHES) > 50 and len(check_brands.SPELLING_HASHES) > 50


def test_brand_check_flags_a_hashed_term():
    term = "Examplecorp Brand"
    hits = check_brands.find_hits(f"we sold to {term}", {check_brands.digest(term.lower())}, {check_brands.digest(term)})
    assert hits == [term]


def test_brand_check_passes_on_repo():
    r = subprocess.run([sys.executable, "scripts/check_brands.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_no_real_pii_in_committed_text():
    from distillrl.pii import scrub

    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    for f in files:
        if f.startswith(("tests/", "src/distillrl/data.py", "src/distillrl/pii.py")) or not f.endswith((".md", ".json", ".csv")):
            continue
        text = (ROOT / f).read_text()
        assert scrub(text) == text, f
