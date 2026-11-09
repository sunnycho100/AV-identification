# Changelog

## 3.0.1 ? 2026-11-03

- Show exact review and retained-frame bounds immediately when they conflict, including loaded drafts.
- Add an explicit interval-fit action; keep all ego anchors for inspection and flag anchors outside the new interval.
- Persist numeric edits while typing and read the displayed bounds again on submission.
- Distinguish invalid frame numbers from an interval outside the retained video.

## 3.0.0 — 2026-11-03

- Added a project home and a full-video ego-trajectory review workspace.
- Added a five-video Todd SE supplementary queue; no clip is preclassified as impossible from its old extraction duration.
- Added native-time interval selection, multiple ego anchors, manual boxes, frozen-tail and other exclusion annotations, and retained-video bounds.
- Added import validation, browser drafts, per-clip saves, downloadable revision history and merged latest-record export.
- Preserved the original 60-video leader relation review URL, schemas, storage keys and annotation files.
- Added a standalone synthetic demo, data schemas, source packaging and isolated HTTP/browser regression checks.

## 2.1 — retained legacy review

- Explicitly distinguished absent leaders from temporarily invisible leaders.
- Supported frame inspection, manual boxes, interval decisions, JSON exchange and append-only saves through VS Code SSH port forwarding.

The pre-v3 application and original annotation hashes are additionally backed up in the parent workspace's `archive/review_web_before_v3_20261103/`. The previous standalone development ZIP is retained in `archive/`.
