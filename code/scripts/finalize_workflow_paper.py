"""Apply small cross-section consistency edits after expanding the methods."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
paper=ROOT/'../paper'
replacements={
 'sections/trajectory_reconstruction.tex':[(r'\subsection{Trajectory representation and calibration}',r'\subsection{Trajectory representation}')],
 'sections/field_pilot_results.tex':[
 ('The local integration pilot uses six Todd/SE recordings','The initial local integration pilot uses a camera height of 16.26~m and six Todd/SE recordings'),
 ('These single-record diagnostics retain the unvalidated camera-height assumption and do not demonstrate broad accuracy gains.','These single-record diagnostics use the same 16.26-m setting and do not demonstrate broad accuracy gains.'),
 ('Camera height remains an assumed historical value; the figure is a pilot diagnostic, not a cross-scene performance estimate.','Camera height is 16.26~m for both runs; the figure is a pilot diagnostic, not a cross-scene performance estimate.'),
 ],
 'sections/conclusion.tex':[
 ('The implemented pipeline connects image calibration, 3D detection, tracking and smoothing to quality-controlled leader--follower windows.','The implemented pipeline connects intrinsic and vanishing-point estimation, height-conditioned BEVHeight localization, two-stage multi-object tracking, trajectory stitching and smoothing, and lane/leader assignment to quality-controlled following windows. Temporal pseudo-label correction, feature masking, heading refinement, and reference-supervised losses are documented as explicit adaptation branches.'),
 ('The initial roadside pilot establishes a reproducible integration and adaptation procedure.','The initial roadside pilot establishes a reproducible integration and adaptation procedure. Additional execution checks cover masking, label correction, head adaptation, and heading refinement; GPS-supervised training remains blocked by registration. The operational 15-m camera-height setting and 17-m sensitivity setting are evaluated separately on two contiguous excerpts.'),
 ],
 'Manuscript.tex':[
 ('An initial local-GPU pilot integrates image calibration, detection, tracking, temporal pseudo-label adaptation, and a quality-gated classifier interface.','A local-GPU pilot integrates intrinsic and pose estimation, BEVHeight detection, two-stage multi-object tracking, temporal reconstruction, pseudo-label adaptation, and a quality-gated classifier interface. Additional checks examine masking and orientation refinement, with a separate 15-m versus 17-m camera-height sensitivity experiment.'),
 ]
}
for rel,pairs in replacements.items():
    p=paper/rel;text=p.read_text(encoding='utf-8')
    for old,new in pairs:
        if old not in text and new not in text:raise ValueError(f'Missing replacement in {rel}: {old[:70]}')
        text=text.replace(old,new)
    p.write_text(text,encoding='utf-8')
# Old generators are retained as archival reconstruction utilities. Their
# output must not silently replace the current methods/results revision.
