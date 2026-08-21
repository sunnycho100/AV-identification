"""Verify the server environment can do what the Mac could not.

Run this immediately after building the conda env, before syncing data or
starting a run. Each check corresponds to a specific failure we already hit
locally, so a pass here means that failure cannot recur.
"""
import sys
import torch
import torch.nn as nn

ok = True


def check(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


check("CUDA available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU visible")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    check("compute capability >= 8.9 (Ada)", cap >= (8, 9), f"sm_{cap[0]}{cap[1]}")

# 1. The reason we are on this machine: DCN must have a working backward.
#    On the Mac this raised "DeformConv2dFunctionBackward has no attribute bufs_",
#    which forced freezing the height branch and left the depth warp untrainable.
try:
    from mmcv.ops import DeformConv2dPack
    m = DeformConv2dPack(8, 8, kernel_size=3, padding=1).cuda()
    x = torch.randn(2, 8, 16, 16, device="cuda", requires_grad=True)
    m(x).square().mean().backward()
    check("deformable conv backward", x.grad is not None and torch.isfinite(x.grad).all(),
          "height branch is trainable here")
except Exception as e:
    check("deformable conv backward", False, f"{type(e).__name__}: {str(e)[:90]}")

# 2. Voxel pooling: the CUDA extension should be compiled, not the slow fallback.
try:
    from ops.voxel_pooling import voxel_pooling
    from ops.voxel_pooling.voxel_pooling import _CUDA_AVAILABLE
    g = torch.randint(0, 32, (1, 64, 3), device="cuda", dtype=torch.int32)
    f = torch.randn(1, 64, 16, device="cuda", requires_grad=True)
    out = voxel_pooling(g.contiguous(), f.contiguous(), torch.tensor([32, 32, 1]))
    out.square().mean().backward()
    check("voxel pooling forward+backward", f.grad is not None and torch.isfinite(f.grad).all())
    check("voxel pooling CUDA extension built", _CUDA_AVAILABLE,
          "compiled" if _CUDA_AVAILABLE else "falling back to slow pure-PyTorch path")
except Exception as e:
    check("voxel pooling", False, f"{type(e).__name__}: {str(e)[:90]}")

# 3. Numerics sanity. MPS silently produced ~40x gradients on the Mac while the
#    loss appeared to fall, so device numerics are never assumed again.
try:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 3, padding=1))
    x = torch.randn(2, 3, 32, 32)
    lc = net(x).square().mean(); lc.backward()
    gc = torch.cat([p.grad.flatten() for p in net.parameters()]).norm().item()
    net.zero_grad(); netg = net.cuda()
    lg = netg(x.cuda()).square().mean(); lg.backward()
    gg = torch.cat([p.grad.flatten() for p in netg.parameters()]).norm().item()
    rel = abs(gg - gc) / max(gc, 1e-9)
    check("CPU vs GPU gradients agree", rel < 0.02,
          f"cpu {gc:.4f} vs gpu {gg:.4f} (rel {rel:.1%})")
except Exception as e:
    check("CPU vs GPU gradients agree", False, f"{type(e).__name__}: {str(e)[:90]}")

for mod in ["mmcv", "mmdet", "mmdet3d"]:
    try:
        m = __import__(mod)
        check(f"{mod} importable", True, m.__version__)
    except Exception as e:
        check(f"{mod} importable", False, str(e)[:90])

print("\n" + ("environment OK: full fine-tuning (head + height branch) is possible"
              if ok else "environment NOT ready, fix the FAILs above before syncing data"))
sys.exit(0 if ok else 1)
