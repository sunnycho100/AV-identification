import torch
from ops.voxel_pooling.voxel_pooling import _voxel_pooling_pure_pytorch

def test_autograd_pure_pytorch():
    B = 2
    N = 20
    C = 4
    vx, vy, vz = 5, 5, 1
    voxel_num = torch.tensor([vx, vy, vz])

    geom_xyz = torch.randint(0, 5, (B, N, 3), dtype=torch.int32)
    input_features = torch.randn(B, N, C, dtype=torch.float64, requires_grad=True)

    # Test autograd gradcheck
    def func(feats):
        return _voxel_pooling_pure_pytorch(geom_xyz, feats, voxel_num)

    test_passed = torch.autograd.gradcheck(func, (input_features,), eps=1e-6, atol=1e-4)
    print("PyTorch fallback gradcheck passed:", test_passed)

test_autograd_pure_pytorch()
