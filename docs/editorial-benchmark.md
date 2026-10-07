# Editorial Benchmark (Task 07, rev2)

Local reproducible benchmark recording what clips were produced, how many
distinct useful moments humans accept, and their editorial/framing/text
defects. Human judgments stay separate from automated checks; unreviewed
stays unknown, invalid is an error.

## Ownership

- `src/clippyme/pipeline/editorial_benchmark.py` â€” implementation
- `tests/pipeline/test_editorial_benchmark.py` â€” 17 tests
- `docs/editorial-benchmark.md` â€” this file
- `tests/fixtures/editorial_benchmark/` â€” synthetic fixtures

## Input manifest

Versioned JSON (version 1). Sources require id, file, duration, genre,
and validated asset with sha256 fingerprint. Clips require clip_id,
source_id, start/end, file (rendered asset), and optional moment_id.
Fingerprints frozen; mismatch rejects stale assets. Asset paths use
real resolved containment (not string prefix); traversal rejected.

Six genres: interview/podcast, reaction/comedy, gaming/stream,
tutorial/screen, sport/action, performance/visual. All six reported
even when missing.

## Human reviews

JSON or CSV with opaque clip IDs. Strict boolean parsing: true/false
only; 'definitely' is an error, not unknown. Unknown is null/blank/
omitted. framing_errors is a list (CSV: comma-separated). Unknown
clip IDs rejected. Human opening/payoff/ending judgments reported
with known denominators, separate from acceptance.

## CLI and reports

Strict finite JSON. Per-run stats include commit, settings, source
fingerprints, candidate/reviewed/accepted, acceptance rate,
distinct moments (source-scoped (source_id, moment_id); None if
partial labels), label coverage, per-run source hours and
accepted-per-hour, review completeness, and judgment tallies.
Per-genre for all six. Paired sources show runs, fingerprint
compatibility, and missing runs. Media QA results (if --check-media)
kept separate from human ratings. Output collision rejected.

## Real footage

Not run. Synthetic fixtures only. Corpus planned, not assembled.
