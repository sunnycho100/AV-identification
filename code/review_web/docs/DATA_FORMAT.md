# Project and review data contract

The file schemas are versioned independently of the app: `trajectory-review-project/v1` describes inputs; `trajectory-review/v1` describes human decisions. `schemas/` contains machine-readable structural definitions. The server also validates image bounds, source hashes, native frame times, interval overlap and timestamp continuity, which require the dataset itself.

## Project layout

```text
project/
  project.json
  clips/CASE_001.json
  frames/CASE_001/000.jpg ...
  videos/CASE_001.mp4
```

Use the demo's `project.json` as a complete example. Set `media_root` relative to the manifest. For a project discovered in `code/data/review_projects/<id>/`, `../../..` refers to the `code/` dataset root. Every resolved media path must stay inside the designated workspace. An external `--project` is self-contained inside its own directory unless it lies within `--workspace`.

Each case has a unique `clip`, `video` (relative path or null), `frames_template` (e.g. `frames/CASE_001/{frame:03d}.jpg`), and `frame_data` relative to the project directory. `source_video_sha256` identifies the original video; `frame_data_sha256` hashes the exact frame JSON bytes. The loader verifies the frame JSON hash; build the video hash when preparing the dataset. It does not rehash a whole video on every frame request.

Frame JSON:

```json
{
  "clip": "CASE_001", "width": 1920, "height": 1080,
  "frames": [
    {"frame": 0, "t_s": 0.0, "boxes": [
      {"id": "candidate-1", "bbox": [100, 200, 180, 260]}
    ]}
  ],
  "timestamp_gaps": []
}
```

Populate **all** available frames, sorted by native timestamp, with unique integer frame IDs. `t_s` is the native frame time, not `frame / nominal_fps`. Boxes are optional candidate proposals in native image pixels `[left, top, right, bottom]`; they are not accepted identities until selected. A timestamp gap annotation uses `{ "before": 179, "after": 180, "gap_s": 3.003 }`. The server checks actual adjacent timestamps rather than trusting the displayed gap list. JPG, PNG and the demo's SVG frames are supported by the browser.

## Review semantics

Each submission contains `schema`, `project_id`, and one or more `records` (at most one per clip). Each record preserves source hashes, native start/end frame and time, `anchors`, `trim`, `excluded_ranges`, `identity_continuous`, `decision`, `issues` and `note`.

| Field | Meaning |
|---|---|
| `anchors[].frame`, `bbox`, `source` | Human-confirmed ego location at a native frame; source is `manual` or `candidate`. Pixel coordinates remain in the original full-resolution image even when zoomed. |
| `trim.keep_start_frame`, `keep_end_frame` | Inclusive frames to keep in subsequent processing. No destructive video edit occurs in the reviewer. |
| `excluded_ranges[]` | Inclusive frame ranges and reasons (`image_freeze`, `occlusion`, etc.). These must not overlap a confirmed usable interval. |
| `ready_for_retracking` | Human confirms sufficient visible time and continuous ego identity; reconstruction and metric accuracy remain unverified. |
| `needs_followup` | A partial or uncertain annotation worth retaining. May include short intervals or timestamp gaps. |
| `unusable` | Human explicitly found no suitable interval. Never assigned automatically from a detector's short output. |

For `ready_for_retracking`, the interval must meet `required_duration_s` (currently 3), be inside the retained range, have at least two distinct anchors and continuous identity, exclude all rejected frames, and have no native adjacent timestamp gap greater than 0.2 s. It is **not** permissible to fill missing recorded seconds with interpolation. Substantial frame sampling intervals require a new dataset policy; do not merely relabel timestamps to pass validation.

The browser recommends start/middle/end anchors. If shortening an interval leaves anchors outside it, delete or move those anchors before saving; they are highlighted in red. Exclusions can be removed, but any previous trim boundary remains visible for explicit adjustment.

## Storage and API

Workspace projects save to `data/annotations/trajectory_reviews/<project_id>/review_<UTC>_<unique>.json`. Explicit external projects and the standalone demo use a `reviews/<project_id>/` subdirectory beside their manifest. Saves create files exclusively; previous revisions are not overwritten. Server metadata records save time and app version. Latest state is the newest record for each clip across all revisions.

| Endpoint | Result |
|---|---|
| `GET /api/projects` | Project cards and the legacy relation-review URL, if available |
| `GET /api/projects/<id>` | Case metadata and review progress |
| `GET /api/projects/<id>/clips/<clip>` | Native frame times and candidate boxes |
| `GET /api/projects/<id>/reviews` | Latest per-clip records plus revision list |
| `GET /api/projects/<id>/reviews/<revision>` | Read-only historical submission |
| `POST /api/projects/<id>/validate` | Validate an import without writing it |
| `POST /api/projects/<id>/reviews` | Validate and append a submission |
| `GET /media/<id>/<clip>/frame/<frame>` | Original image |
| `GET /media/<id>/<clip>/video` | Original video, with HTTP Range support |

Writes require a loopback same-origin browser request, including a forwarded local port. There is no multi-user authentication service. The intended deployment is a local research workspace accessed through SSH. Concurrent saves preserve both revisions; the newest save wins for the per-clip view. Use the revision list to inspect conflicting edits.

Downstream processing should load the latest submitted record, apply the retained/excluded frame ranges, use the human ego anchors as tracking constraints, then independently evaluate the reconstructed duration and errors. These human boxes are not metric three-dimensional ground truth. The legacy leader relation schema remains separate and unchanged.
