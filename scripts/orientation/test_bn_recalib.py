"""Self-check for bn_recalib: weights must not move, only BatchNorm buffers.

The whole claim of candidate 3 is "checkpoint weights unchanged", so that is what
is asserted here, twice. Once on a toy model that runs in a second, and once on
the real saved checkpoint if it is on disk, where the comparison is against the
DAIR checkpoint it was derived from.

    /Users/sunghwan_cho/miniforge/bin/python scripts/orientation/test_bn_recalib.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bn_recalib import BATCH, bn_keys, diff_keys, pick_frames, prepare_bn


class Toy(nn.Module):
    """A conv BN and a BatchNorm1d(27), the two shapes HeightNet actually uses."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn2d = nn.BatchNorm2d(4)
        self.bn1d = nn.BatchNorm1d(27)
        self.fc = nn.Linear(27, 4)

    def forward(self, img, vec):
        return self.bn2d(self.conv(img)).mean() + self.fc(self.bn1d(vec)).mean()


def test_only_bn_buffers_move():
    torch.manual_seed(0)
    model = Toy()
    # pretend the checkpoint was trained on a different distribution
    with torch.no_grad():
        model.bn2d.running_mean.fill_(5.0)
        model.bn2d.running_var.fill_(9.0)
        model.bn1d.running_mean.fill_(2183.0)
        model.bn1d.running_var.fill_(1e6)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    bns = prepare_bn(model)
    assert len(bns) == 2, bns
    assert all(m.training and m.momentum is None for m in bns)
    with torch.no_grad():
        for _ in range(3):
            model(torch.randn(BATCH, 3, 8, 8), torch.randn(BATCH, 27) * 3 + 1500)
    model.eval()

    after = model.state_dict()
    moved = diff_keys(before, after)
    allowed = bn_keys(model)
    assert moved, "nothing moved: the recalibration did not run"
    assert moved <= allowed, sorted(moved - allowed)

    params = {k for k, _ in model.named_parameters()}
    assert params & moved == set(), sorted(params & moved)
    for k in params:
        assert torch.equal(before[k], after[k]), k

    # the buffers did not merely reset to (0, 1): they took our data's statistics
    assert abs(float(after["bn1d.running_mean"].mean()) - 1500.0) < 50.0
    assert float(after["bn1d.running_var"].mean()) < 1e5
    assert float(after["bn2d.running_var"].mean()) != 9.0


def test_batch_of_one_would_have_failed():
    """Why BATCH is 2: BatchNorm1d in train mode rejects a single sample."""
    model = Toy()
    prepare_bn(model)
    try:
        model(torch.randn(1, 3, 8, 8), torch.randn(1, 27))
    except ValueError:
        return
    raise AssertionError("BatchNorm1d accepted a batch of 1; BATCH could be 1")


def test_pick_frames_spreads_over_clips():
    root = Path(__file__).resolve().parents[2] / "data/camera-data"
    dirs = [root / "AV_T_EW_3/frames_all", root / "AV_T_WE_1/frames_all"]
    if not all(d.exists() for d in dirs):
        print("skip pick_frames: frames_all missing")
        return
    frames = pick_frames(dirs, 60)
    assert len(frames) == 60, len(frames)
    clips = {c for _, c in frames}
    assert clips == {"AV_T_EW_3", "AV_T_WE_1"}, clips
    assert len(set(f for f, _ in frames)) == 60, "duplicate frames picked"


def _stripped(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    return {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}


def test_saved_checkpoint_matches_source():
    from bn_recalib import DEFAULT_OUT
    from scripts.object_detection.run_bevheight_single import CKPT_PATH
    if not Path(DEFAULT_OUT).exists():
        print(f"skip saved-checkpoint check: {DEFAULT_OUT} not built yet")
        return
    src, new = _stripped(CKPT_PATH), _stripped(DEFAULT_OUT)
    shared = set(src) & set(new)
    assert len(shared) > 100, len(shared)
    moved = {k for k in shared if not torch.equal(src[k], new[k])}
    bad = {k for k in moved
           if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    assert not bad, sorted(bad)[:10]
    assert moved, "the saved checkpoint is identical to the source"
    print(f"saved checkpoint: {len(shared)} shared tensors, {len(moved)} BN buffers moved")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
