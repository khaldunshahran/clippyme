# Editorial Benchmark (Task 07)

Local reproducible benchmark recording what clips were produced, how many
distinct useful moments humans accept, and their editorial/framing/text
defects. Human judgments stay separate from automated checks; unreviewed
stays unknown.

## Ownership

- `src/clippyme/pipeline/editorial_benchmark.py` — implementation
- `tests/pipeline/test_editorial_benchmark.py` — tests
- `docs/editorial-benchmark.md` — this file
- `tests/fixtures/editorial_benchmark/` — synthetic fixtures

Does not modify discovery, models, prompts, renderers, frontend, or
existing test manifests. Reuses `media_qa.inspect_clip` for objective
media checks without changing its policy.

## Input manifest

Versioned JSON. Example:

```json
{
  "version": 1,
  "asset_root": ".",
  "sources": [
    {
      "id": "src1",
      "file": "src1.mp4",
      "duration": 3600.0,
      "genre": "interview/podcast",
      "fingerprint": "sha256:...",
      "synthetic": false
    }
  ],
  "runs": [
    {
      "run_id": "baseline",
      "commit": "abc123",
      "settings": {"model": "v1"},
      "clips": [
        {
          "clip_id": "c1",
          "source_id": "src1",
          "start": 120.5,
          "end": 145.0,
          "moment_id": "m42"
        }
      ]
    }
  ]
}
```

- `asset_root` resolved relative to manifest; traversal rejected.
- `fingerprint` (sha256) optional; mismatch raises stale-asset error.
- `moment_id` optional manual label for distinct-moment counting.
- `genre` must be one of six: `interview/podcast`, `reaction/comedy`,
  `gaming/stream`, `tutorial/screen`, `sport/action`, `performance/visual`.
- Synthetic fixtures must set `"synthetic": true`.

## Human reviews

JSON or CSV. Opaque clip IDs. Fields:

- `opening_understandable`, `payoff_delivered`, `ending_natural` (bool)
- `framing_errors` (list, e.g. `["face_cropped"]`)
- `hook_obstruction`, `caption_obstruction` (bool)
- `accepted` (bool) — overall accept/reject
- `notes` (optional)

Unreviewed clips stay unknown; never inferred from AI scores. Do not
fabricate speaker labels, retention, cost, or ratings.

CSV template: `tests/fixtures/editorial_benchmark/reviews_template.csv`

## CLI

```bash
python -m clippyme.pipeline.editorial_benchmark \
  --manifest manifest.json \
  --reviews reviews.json \
  --json-out report.json \
  --text-out report.txt
```

Strict JSON output (no NaN/Infinity). Readable report includes:

- Candidate/reviewed/accepted counts and acceptance rate
- Distinct accepted moments (only where `moment_id` labels exist)
- Per-genre and aggregate stats
- Accepted per source hour (discloses incomplete review)
- Paired runs by source ID; missing pairs visible

AI quality estimates and media errors are reported separately from
human acceptance. Reports never overwrite clips or projects.

## Real-footage corpus

Approximately 30 sources planned (per research plan), not yet assembled.
When authorized media/reviews are absent, the report states:
"Real-footage benchmark has not run; synthetic fixtures only."
Do not pretend the corpus exists.

## Tests

`tests/pipeline/test_editorial_benchmark.py` covers:

- Missing/traversal/stale assets rejected
- Duplicate clip/review IDs rejected
- Unreviewed vs null vs zero distinctions
- Denominator correctness (accepted/reviewed, not accepted/candidate)
- Per-genre missing coverage
- Paired-source mismatch visible
- Manually labeled duplicate moments counted once
- Strict finite JSON
- CLI end-to-end; malformed input fails clearly
