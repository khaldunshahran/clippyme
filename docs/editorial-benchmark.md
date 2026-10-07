# Editorial Benchmark (Task 07, rev3)

Local reproducible benchmark recording what clips were produced, how many
distinct useful moments humans accept, and their editorial/framing/text
defects. Human judgments stay separate from automated checks; unreviewed
stays unknown, invalid is an error.

## Ownership

- `src/clippyme/pipeline/editorial_benchmark.py` â€” implementation
- `tests/pipeline/test_editorial_benchmark.py` â€” 11 tests
- `docs/editorial-benchmark.md` â€” this file
- `tests/fixtures/editorial_benchmark/` â€” synthetic fixtures + generator

## Input manifest

Versioned JSON (version 1). Sources require id, file, duration, genre,
synthetic (strict bool), and validated asset with sha256 fingerprint.
Clips require clip_id, source_id, start/end, file (rendered asset),
and optional moment_id. Fingerprints frozen; stale rejected.
Asset paths use real resolved containment; traversal rejected.

Runs have explicit `source_ids` (including zero-candidate sources)
and `source_fingerprints` (frozen at run time). Mismatch with current
source fingerprints rejects the manifest.

Six genres: interview/podcast, reaction/comedy, gaming/stream,
tutorial/screen, sport/action, performance/visual. All six reported.

## Human reviews

JSON or CSV. Strict booleans: JSON requires true/false (strings
rejected); CSV allows documented tokens true/false/1/0/yes/no.
Unknown is null/blank/omitted. `framing_errors` is list or null
(None = unknown, [] = known no defects). Unknown clip IDs rejected.
Hook/caption obstruction and framing reported with known denominators.

## CLI and reports

`--check-media` runs real `media_qa.inspect_clip` with
`expected_duration` and `expected_aspect`. Results in
`media_checks` separate from human ratings.

Per-run stats: commit, settings, source_ids, frozen source_fingerprints,
candidates, reviewed, accepted, acceptance rate, distinct moments
(source-scoped; None if partial labels), label coverage, source hours
(including zero-candidate), accepted/hour, review completeness,
judgments with known denominators, and clip fingerprints.

Per-genre for all six with per-run breakdown. Paired sources use
per-run frozen fingerprints (not tautological); missing runs listed.

Output collisions rejected: manifest, reviews, assets, and identical
destinations protected. Write failures return clean nonzero exit.

## Demo

`tests/fixtures/editorial_benchmark/generate_demo.py` creates
accurate-duration synthetic mp4s and frozen manifest. Explicitly
synthetic. Real footage not measured.
