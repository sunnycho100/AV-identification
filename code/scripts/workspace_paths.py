"""Locations for current code and immutable manifests written before layout v2."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PAPER = PROJECT_ROOT / 'paper'
ARCHIVE = PROJECT_ROOT / 'archive'

def resolve_legacy(relative):
    """Resolve a project-relative path from a pre-reorganization manifest."""
    value = str(relative).replace('\\', '/')
    for old, new in [('paper/', PAPER), ('archives/', ARCHIVE/'source_packages'), ('revisions/', ARCHIVE/'revisions')]:
        if value.startswith(old): return new / value[len(old):]
    exact = {'README.txt': 'docs/history/TRAINING_DATA_HANDOFF_20261101.txt', 'README.md': 'docs/PROJECT_HISTORY.md',
             'leader_review_decisions.json': 'data/annotations/leader_review_decisions.json',
             'trajectory_extension_review.json': 'data/annotations/trajectory_extension_review.json'}
    return ROOT / exact.get(value, value)


def verify_recorded_input(relative, expected_sha256):
    """Check an old manifest against current inputs or an explicitly preserved source revision.

    A source change is accepted only when its current hash matches the authorized
    ledger AND the exact recorded source is still available in a backup. Data and
    model changes therefore cannot be silently excused by the directory move.
    """
    relative = str(relative).replace('\\', '/')

    def sha(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    current = resolve_legacy(relative)
    actual = sha(current)
    if actual == expected_sha256:
        return 'unchanged_input'
    ledger = json.loads((ROOT / 'catalogs/workspace_changes.json').read_text(encoding='utf-8'))
    row = next((r for r in ledger if r['path'].replace('\\', '/') == relative), None)
    if row and row.get('current_sha256') == actual:
        candidates = [resolve_legacy(row['backup']), ARCHIVE / 'reorganization_20261103/before' / relative]
        if any(p.is_file() and sha(p) == expected_sha256 for p in candidates):
            return 'recorded_source_preserved_current_revision_verified'
    raise AssertionError(f'Unrecorded input change: {relative}')
