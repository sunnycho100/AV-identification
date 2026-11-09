# Human-assisted data review: methods note

Suggested wording for the data-processing subsection (features implemented in v3.0.0; no new reconstruction result implied):

> We developed a browser-based review tool for roadside video preparation. Reviewers inspect native video frames, select a temporally continuous interval, and mark the ego vehicle at multiple frames to resolve missed detections and ambiguous track identities. Unusable image intervals, including frozen video tails, are explicitly annotated, while the original video is retained. The tool separately supports ego–leader relation review, distinguishing an absent leader from a leader that is temporarily outside the camera view. Source identifiers, native timestamps, bounding-box anchors, interval decisions and revision history are exported as structured JSON records for subsequent trajectory processing.

The interface, JSON schemas and a synthetic demonstration dataset can accompany the paper as software artifacts. The current source package is prepared locally; this does not mean a public repository or dataset release has taken place.

The Todd SE supplementary queue contains five videos previously producing less than three seconds of trusted reconstructed trajectory. The collector reports that all five have at least three seconds of usable visible motion, including an early interval in `HV_T_EW_2` before its frozen tail. This assertion motivates a new review round. The review outcome, reconstructed durations and accuracy must be reported after the annotations are submitted and processing is rerun, not inferred from the existence of the interface.

For a paper figure, prefer a synthetic screenshot or a reviewed field example cleared for release. Describe the interval selection, ego anchors, excluded ranges and revision history. Do not present yellow candidate boxes as manually validated ground truth.
