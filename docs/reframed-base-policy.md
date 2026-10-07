# Reframed-Base Lineage Policy (Task 05)

Preserves automatic speaker/layout framing when exporting a project with
unchanged crop, timeline, and render geometry.

## Lineage

Newly emitted auto projects include optional `reframed_base`:
- `base_ref`: job-relative path to the original reframed clip (`clip_final`,
  before overlay composition). Resolves safely inside job_dir only.
- `identity`: file/size/mtime for staleness detection.
- `source`: file/width/height baseline.
- `segments`: ordered list of {id, start, end, crop{x,y,w,h}, transition_in}.
- `render`: width/height/fps baseline.

Survives model validation, JSON serialization, versioned save/load, undo.
Malformed references never read unrelated files.

## Eligibility

Semantic equivalence to the stored baseline:
- **Source**: file name, width, height must match.
- **Timeline**: ordered segments; each id/start/end/crop/transition must match.
- **Render**: width, height, fps must match.
- **Base file**: exists, size/mtime match (not missing/stale/replaced).

Metadata-only changes do NOT invalidate:
- version, origin, caption edits, overlays, grade, audio/music, segment labels.

Invalidating changes (use existing source renderer):
- Crop x/y/w/h, segment start/end, segment count/order, transition,
  source file/dimensions, render width/height/fps.

Origin=user or version>1 alone does NOT invalidate (caption edits set these).

## Render behavior

`render_project()` base selection:
1. `project.base_clip` set → legacy path (unchanged, byte-identical).
2. `project.reframed_base` present AND eligible → reuse validated base.
   Apply CURRENT project composition (captions, overlays, grade) and music.
   Do NOT use stale legacy toggles.
3. Otherwise → materialize timeline from source (existing path).

Missing/stale/invalid base on an otherwise eligible project: fail clearly
before publication. Never silently fall back to static center crop.
Never reuse a mismatched asset.

Old projects without lineage: existing routes unchanged.

## QA

Selected base must match project's output aspect/dimensions. Duration/FPS/
encoder tolerances documented; audio preserved; no byte-identity promise
across encoders.
