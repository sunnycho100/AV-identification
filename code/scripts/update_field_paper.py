"""Apply the documented field-data/method revision; original files are backed up."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'paper'
data=r'''\section{Data collection}
\label{sec:collection}
\subsection{Collected roadside video and reference trajectories}
The assembled field collection contains 60 roadside video clips from five sites along the US~12/18 corridor in the Madison area: Fish Hatchery Road, Seminole Highway, Todd Drive, Verona Road, and Whitney Way. Seven site--view combinations are represented (Table~\ref{tab:field-inventory}). The video-container durations sum to 640.139~s (approximately 10.67~min); this is a sum of clip durations, not continuous corridor coverage or instrumented-vehicle observation time. Fifty-six clips have paired processed positioning files containing 74,267 rows in total. The timestamps in these paired files fall on October~25, 2025 (2 recordings), December~13, 2025 (9), and October~29, 2026 (45), after conversion to local time. The four videos without paired files cannot be dated independently from those logs.

\begin{table}[pos=tbp]
\centering
\caption{Inventory of the collected roadside clips. ACC/HV counts are recording labels for the instrumented vehicle, not labels for all visible traffic.}
\label{tab:field-inventory}
\begin{tabular}{llrrrr}
\toprule
Site & View & Clips & Paired GPS & ACC & HV \\
\midrule
Fish Hatchery Rd & SW & 6 & 5 & 3 & 3 \\
Seminole Hwy & NW & 14 & 13 & 7 & 7 \\
Todd Dr & SE & 13 & 12 & 7 & 6 \\
Todd Dr & W & 5 & 5 & 1 & 4 \\
Verona Rd & SE & 16 & 15 & 8 & 8 \\
Whitney Way & NW & 4 & 4 & 3 & 1 \\
Whitney Way & SE & 2 & 2 & 1 & 1 \\
\midrule
Total & & 60 & 56 & 30 & 30 \\
\bottomrule
\end{tabular}
\end{table}

The supplied vehicle reference images show black and red Lincoln test vehicles with visible roof equipment and hood markers. Color is not a control-state label. The acquisition team confirms that the AV/HV prefixes in the processed filenames encode ACC-enabled and human-driven operation, respectively. These assignments identify the instrumented vehicle's intended state for a recording; they do not constitute independently checked, framewise controller-switch telemetry. Other vehicles visible in the same recording remain unlabeled. Recording counts must not be interpreted as counts of independent vehicles, and repeated passages may involve the same physical vehicle.

\subsection{Time alignment, geometric references, and quality control}
Each paired CSV contains 19 fields: epoch and local time variables; latitude, longitude, and height; north/east/up velocity; processed local positions and velocities; video-relative time; and nearest-frame indices, timestamps, and alignment residuals. These processed positioning records provide an external motion reference rather than camera-derived trajectories. Decoded presentation timestamps are retained for the video pipeline, and reference signals are compared on video-relative time. No nominal constant-frame-rate assumption is required for that comparison.

The inventory records four missing positioning files, 1,253 rows with absent nearest-frame indices across 12 files, and a trimmed-video mismatch for \texttt{AV\_T\_WE\_3}: its approximately 7.03-s video is shorter than the approximately 9.03-s span represented by its mapped reference times. Such rows cannot simply be treated as synchronized frame observations. File provenance, timestamps, and quality flags are retained in a clip catalog. The six-clip integration pilot in Section~\ref{sec:trajectory-results} uses a separate, explicit recording split and excludes this mismatched clip.

Calibration is estimated from the video using the available intrinsic, vanishing-point, and metric-scale procedures. The package does not contain a complete surveyed camera calibration for all seven views. Camera model, focal settings, pose, antenna-to-vehicle reference-point offsets, and positioning accuracy should therefore not be inferred from illustrative installation photographs. The initial Todd pilot uses image-derived intrinsics and orientation with a historical 16.26-m camera-height assumption. Its metric-scale status is recorded explicitly, and lane-width estimation is treated as an image-based diagnostic rather than survey ground truth. Target identification uses the supplied vehicle-marker method where the marker is visible; uncertain identities are excluded from labeled evaluation.
'''
method=r'''\section{Vehicle trajectory reconstruction from roadside cameras}
\label{sec:trajectory}
\subsection{Trajectory representation and calibration}
Let $I_c(t)$ denote a frame acquired by camera $c$ at presentation time $t$, and let $\mathcal C_c$ contain its intrinsic matrix, camera-to-road transformation, and ground-plane definition. The reconstruction module produces
\begin{equation}
 \mathcal R:\{I_c(t),\mathcal C_c\}_{t}\longmapsto\{\mathcal T_i\}_{i=1}^{N},
 \qquad \mathcal T_i=\{(t,\mathbf p_i(t),\mathbf v_i(t))\}_{t\in\Omega_i},
 \label{eq:trajectory}
\end{equation}
where $\mathbf p_i=[s_i,r_i]^\top$ contains longitudinal and lateral road coordinates. Each observation retains its recording, track identity, frame number, timestamp, detection status, and associated lane and leader. Derivatives and interaction features are subsequently constructed within valid observation windows; they are not extrapolated across missing tracklets.

In the implemented pilot, AnyCalib estimates pinhole intrinsics on three frames per recording, and the componentwise median is used. A line-segment detector estimates the longitudinal vanishing point on 15 distributed frames; the median accepted point determines pitch and yaw under a zero-roll assumption. A metric height fixes scale. The code additionally supports lane-width-based scale estimation and reference-distance fitting, but the reported initial pilot retains the historical Todd height of 16.26~m. Thus its positions and speeds remain conditional on a flat-road model and an incompletely validated scale. Intrinsic samples, vanishing-point diagnostics, and the exact transformation used for detection are saved for each recording.

\subsection{BEVHeight localization and temporal reconstruction}
BEVHeight \citep{bevheight2023} lifts image features using a predicted distribution of height above the ground and calibrated camera geometry, aggregates the lifted features in bird's-eye view, and predicts oriented 3D vehicle boxes. The pilot uses the supplied ResNet-50, 102.4-m configuration and the previously fine-tuned checkpoint. Its bottom-centered box convention is transformed consistently to the detector's road-ground convention; box dimensions and the ground-plane offset are retained separately from the vehicle's motion state.

Detections are filtered to the roadway and to a usable range determined by camera height, focal length, and a maximum ground footprint of one meter per image pixel. The Todd pilot uses the existing site-specific road band. It is not transferred unchanged to other sites. The AB3DMOT implementation associates 3D detections using a constant-velocity Kalman model and two association stages. Detections scoring at least 0.45 are associated first and may initialize tracks; weaker detections in the range 0.20--0.45 may only continue unmatched existing tracks. Tracks may coast for up to six frames. These parameters are fixed across the training, validation, and test recordings.

The postprocessing stage joins compatible fragments over gaps of at most one second, using longitudinal motion and a lateral displacement constraint. Implausible jumps are split using the existing 45-m/s physical gate. A Rauch--Tung--Striebel smoother estimates positions and velocities, with observation uncertainty increasing with the projected ground footprint. Coasted tails are trimmed and substantially overlapping duplicate tracks are removed. Track-derived lane centers are estimated separately by travel direction using a 0.25-m lateral histogram, a minimum 2.5-m peak separation, and moving observations above 3~m/s. Observations more than 2~m from a lane center remain unassigned. The same-direction, same-lane vehicle immediately ahead supplies a candidate leader. These automatic assignments are hypotheses, not manually verified association ground truth.

\subsection{Temporal pseudo-label refinement}
\label{sec:trajectory-optimization}
The implemented adaptation procedure constructs pseudo-labels from the baseline detections and their tracked motion. Trusted tracks require at least 15 states, mean speed above 5~m/s, median detection score of at least 0.40, at least 80\% detection matching coverage, limited heading variation, and consistency with the road plane. For their matched detections, heading is derived from the tangent of a fitted trajectory, box height above the road is constrained by the ground plane, and dimensions are replaced by their per-track medians. Postprocessed velocities are expressed in meters per second. All other car detections above score 0.30 remain in the label set, preventing confident but uncorrected cars from becoming training negatives.

Only the detection head is updated in the pilot, with the image and height backbone fixed. Training recordings supply the pseudo-labels used for optimization; distinct validation recordings supply a consistency loss for checkpoint selection. Test recordings are excluded from this process. Since labels are derived from the detector and temporal geometry, this experiment measures adaptation feasibility and consistency. It cannot establish improved recall for missed vehicles or replace independent 3D detection annotations. Reference-position and driving-state measurements, when identity and timing are resolved, provide separate downstream checks.

\subsection{Interface to trajectory-based identification}
The original trajectory export reports center-to-center spacing. The integrated adapter recovers vehicle dimensions through one-to-one matches between track states and same-frame detections within 2~m, then takes per-track median dimensions. At least five matches and 50\% matching coverage are required. For approximately lane-aligned vehicles, the primitive net gap is
\begin{equation}
 g_{i\ell}(t)=d_i\,[s_\ell(t)-s_i(t)]-\frac{L_i+L_\ell}{2},
 \label{eq:field-gap}
\end{equation}
where $d_i\in\{-1,+1\}$ is the travel direction and $L_i,L_\ell$ are estimated vehicle lengths. This conversion is necessary because the classifier was trained on bumper-to-bumper clearance rather than center spacing. Dimension and orientation errors remain sources of gap uncertainty.

The adapter requires a constant follower--leader identity, lane, and direction within each valid interval. Only observations supported by detections are used, and gaps exceeding 0.20~s break the interval. Longitudinal speeds and net gap are resampled to 10~Hz without extrapolation, yielding 40-sample windows with a 3.9-s endpoint span (the source protocol's nominal four-second observation). The original speed-above-50-mph, gap, acceleration, and gap--relative-speed consistency checks are retained. Accepted primitives are stored as $N\times40\times3$ arrays; every candidate window has a provenance row and explicit acceptance or rejection status. The original window-local eleven-feature transform and training-only normalization statistics are then applied to CNN and Transformer models. Filename labels are attached only after image-based identification of the instrumented follower; other road users retain an unknown label. Repeated-observation fusion requires independently supported identity continuity and is not inferred merely from similar motion.
'''
(ROOT/'sections/data_collection.tex').write_text(data,encoding='utf-8')
(ROOT/'sections/trajectory_reconstruction.tex').write_text(method,encoding='utf-8')
replacements={
 'Manuscript.tex': [('A dedicated field campaign is planned to collect roadside video, reference trajectories, and driving-state labels.','The assembled field collection contains 60 clips from five roadside sites, with 56 paired positioning files and recording-level ACC/HV assignments. An initial local-GPU pilot integrates image calibration, detection, tracking, temporal pseudo-label adaptation, and a quality-gated classifier interface.')],
 'sections/introduction.tex': [('The planned synchronized collection will support evaluation of the complete roadside measurement process.','The assembled roadside collection and an initial integration pilot provide a basis for evaluating the measurement process, while metric calibration and labeled-window coverage remain explicit limitations.'),('describes the data collection design','describes the collected data')],
 'sections/conclusion.tex': [('The planned collection campaign aligns roadside video with reference trajectories and driving-state labels to support evaluation of this measurement process.','The assembled field collection comprises 60 roadside clips and 56 paired positioning files, with recording-level ACC/HV assignments for the instrumented vehicle. The implemented pipeline connects image calibration, 3D detection, tracking and smoothing to quality-controlled leader--follower windows.'),('The paired collection will enable evaluation of that relationship and further development of joint visual and trajectory learning.','The initial roadside pilot establishes a reproducible integration and adaptation procedure. Verified metric scale, reliable target association, and sufficient continuous labeled following observations remain necessary before claiming field identification accuracy or cross-scene generalization.')]
}
for rel,pairs in replacements.items():
    p=ROOT/rel;s=p.read_text(encoding='utf-8')
    for old,new in pairs:s=s.replace(old,new)
    p.write_text(s,encoding='utf-8')
print('Updated data collection, trajectory method, abstract, introduction and conclusion.')
