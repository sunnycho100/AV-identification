# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch.autograd import Function

try:
    from . import voxel_pooling_ext
    _CUDA_AVAILABLE = voxel_pooling_ext is not None
except ImportError:
    _CUDA_AVAILABLE = False


def _voxel_pooling_pure_pytorch(geom_xyz, input_features, voxel_num):
    """Pure PyTorch fallback for voxel pooling (scatter_add based).

    Args:
        geom_xyz (Tensor): Integer voxel coords [B, N, 3] (x, y, z).
        input_features (Tensor): Features [B, N, C].
        voxel_num (Tensor): [voxel_x, voxel_y, voxel_z].

    Returns:
        Tensor: (B, C, H, W) BEV feature map.
    """
    geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
    input_features = input_features.reshape(geom_xyz.shape[0], -1,
                                            input_features.shape[-1])
    batch_size = geom_xyz.shape[0]
    num_channels = input_features.shape[2]
    vx, vy, vz = int(voxel_num[0]), int(voxel_num[1]), int(voxel_num[2])

    output = input_features.new_zeros(batch_size, vy, vx, num_channels)

    for b in range(batch_size):
        coords = geom_xyz[b]  # [N, 3]
        feats = input_features[b]  # [N, C]
        # Filter out-of-bound points
        valid = ((coords[:, 0] >= 0) & (coords[:, 0] < vx) &
                 (coords[:, 1] >= 0) & (coords[:, 1] < vy) &
                 (coords[:, 2] >= 0) & (coords[:, 2] < vz))
        coords = coords[valid].long()
        feats = feats[valid]
        # Flatten y*vx + x as scatter index
        linear_idx = coords[:, 1] * vx + coords[:, 0]  # [M]
        linear_idx = linear_idx.unsqueeze(1).expand_as(feats)  # [M, C]
        output_flat = output[b].reshape(-1, num_channels)  # [vy*vx, C]
        output_flat.scatter_add_(0, linear_idx, feats)

    return output.permute(0, 3, 1, 2)  # [B, C, H, W]


class VoxelPooling(Function):
    @staticmethod
    def forward(ctx, geom_xyz: torch.Tensor, input_features: torch.Tensor,
                voxel_num: torch.Tensor) -> torch.Tensor:
        """Forward function for `voxel pooling.

        Args:
            geom_xyz (Tensor): xyz coord for each voxel with the shape
                of [B, N, 3].
            input_features (Tensor): feature for each voxel with the
                shape of [B, N, C].
            voxel_num (Tensor): Number of voxels for each dim with the
                shape of [3].

        Returns:
            Tensor: (B, C, H, W) bev feature map.
        """
        assert geom_xyz.is_contiguous()
        assert input_features.is_contiguous()
        # no gradient for input_features and geom_feats
        ctx.mark_non_differentiable(geom_xyz)
        grad_input_features = torch.zeros_like(input_features)
        geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
        input_features = input_features.reshape(
            (geom_xyz.shape[0], -1, input_features.shape[-1]))
        assert geom_xyz.shape[1] == input_features.shape[1]
        batch_size = input_features.shape[0]
        num_points = input_features.shape[1]
        num_channels = input_features.shape[2]
        output_features = input_features.new_zeros(batch_size, voxel_num[1],
                                                   voxel_num[0], num_channels)
        # Save the position of bev_feature_map for each input point.
        pos_memo = geom_xyz.new_ones(batch_size, num_points, 3) * -1
        voxel_pooling_ext.voxel_pooling_forward_wrapper(
            batch_size,
            num_points,
            num_channels,
            voxel_num[0],
            voxel_num[1],
            voxel_num[2],
            geom_xyz,
            input_features,
            output_features,
            pos_memo,
        )
        # save grad_input_features and pos_memo for backward
        ctx.save_for_backward(grad_input_features, pos_memo)
        return output_features.permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad_output_features):
        (grad_input_features, pos_memo) = ctx.saved_tensors
        kept = (pos_memo != -1)[..., 0]
        grad_input_features_shape = grad_input_features.shape
        grad_input_features = grad_input_features.reshape(
            grad_input_features.shape[0], -1, grad_input_features.shape[-1])
        grad_input_features[kept] = grad_output_features[
            pos_memo[kept][..., 0].long(), :, pos_memo[kept][..., 1].long(),
            pos_memo[kept][..., 2].long()]
        grad_input_features = grad_input_features.reshape(
            grad_input_features_shape)
        return None, grad_input_features, None


if _CUDA_AVAILABLE:
    voxel_pooling = VoxelPooling.apply
else:
    voxel_pooling = _voxel_pooling_pure_pytorch
