import torch
import numpy as np

def test_frustum_and_voxel_grid():
    final_dim = (864, 1536) # (H, W)
    downsample_factor = 16
    fH, fW = final_dim[0] // downsample_factor, final_dim[1] // downsample_factor
    print(f"Feature map dimensions: fH={fH}, fW={fW}") # (54, 96)

    # Frustum generation in LSSFPN:
    d_bound = [-2.0, 0.0, 90]
    alpha = 1.5
    d_coords = np.arange(d_bound[2]) / d_bound[2]
    d_coords = np.power(d_coords, alpha)
    d_coords = d_bound[0] + d_coords * (d_bound[1] - d_bound[0])
    d_coords = torch.tensor(d_coords, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
    
    D, _, _ = d_coords.shape
    x_coords = torch.linspace(0, final_dim[1] - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
    y_coords = torch.linspace(0, final_dim[0] - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
    
    print("x_coords[0, 0, :5]:", x_coords[0, 0, :5].numpy())
    print("x_coords step:", (x_coords[0, 0, 1] - x_coords[0, 0, 0]).item())
    print("y_coords[0, :5, 0]:", y_coords[0, :5, 0].numpy())
    print("y_coords step:", (y_coords[0, 1, 0] - y_coords[0, 0, 0]).item())
    print("d_coords[:5, 0, 0]:", d_coords[:5, 0, 0].numpy())
    print("d_coords[-5:, 0, 0]:", d_coords[-5:, 0, 0].numpy())

    # Check Voxel Grid definition in LSSFPN
    x_bound = [0, 102.4, 0.8]
    y_bound = [-51.2, 51.2, 0.8]
    z_bound = [-5, 3, 8]
    
    voxel_size = torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]])
    voxel_coord = torch.Tensor([row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]])
    voxel_num = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]])
    
    print("voxel_size:", voxel_size.numpy())
    print("voxel_coord (center of first voxel):", voxel_coord.numpy())
    print("voxel_num (grid shape [X, Y, Z]):", voxel_num.numpy())
    
    origin = voxel_coord - voxel_size / 2.0
    print("origin (voxel_coord - voxel_size/2, lower corner):", origin.numpy())

test_frustum_and_voxel_grid()
