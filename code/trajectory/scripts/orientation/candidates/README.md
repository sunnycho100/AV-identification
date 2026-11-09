# Orientation candidates

Each file here is one candidate, a module exposing

    run(clip, det_dir, tracks, cfg) -> {track_id: {frame: yaw}}

`det_dir` is `outputs/object_detection/camera-data/<clip>_phase1`, `tracks` is
`{track_id: [state, ...]}` from `score_heading.load_tracks`, `cfg` comes from
`run_candidate.py --cfg key=value`. Yaws are radians in the camera ground frame;
return only the frames you changed, omit the rest.

A candidate that needs the detector rerun rather than post-processed is graded
with `--cfg suffix=<tag>`: the runner then hands it
`outputs/object_detection/camera-data/<clip>_<tag>` and the matching tracking
directory, and computes the baseline on the clips that rerun covers, dropping
the rest from both sides. `bn_stats_recalib` (suffix `bnrecal`) and `ckpt_r140`
(suffix `r140`) work that way.

Rules. A candidate may read detections, tracks, calibration and frames. It may
never open anything under `Camera data/` (GPS trajectories are held-out ground
truth), and `run_candidate.py` refuses to run a file that mentions
`Camera data` or `trajectory.csv`; it also sets `ORIENTATION_NO_GPS=1`. Do not
tune per clip: the keep rule grades the mean over all held-out clips. Run it
with `run_candidate.py <name>`, which scores it and appends one ledger row.
