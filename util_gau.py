import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
import util
import torch

class GaussianData6D:
    means: np.ndarray # 6d -> xyz world pos and xyz view dir
    covariances: np.ndarray # 6d covariance matrix

    # Cached cuda intermediates
    cuda_covariances_11: torch.Tensor
    cuda_covariances_12: torch.Tensor
    cuda_covariances_21: torch.Tensor
    cuda_covariances_22: torch.Tensor
    cuda_covariances_22_inv: torch.Tensor
    cuda_covariances_regr: torch.Tensor
    cuda_covariances_conditioned: torch.Tensor
    cuda_covariances_conditioned_flat: torch.Tensor
    cuda_means: torch.Tensor
    cuda_opacities: torch.Tensor

    covariances_conditioned_flat: np.ndarray



    colors: np.ndarray # 3d color
    opacities: np.ndarray # 1d opacity

    def __init__(self, means, covariances, colors, opacities):
        self.means = means
        self.covariances = covariances
        self.colors = colors
        self.opacities = opacities

        self.cuda_means = torch.tensor(self.means, device='cuda')
        self.cuda_opacities = torch.tensor(self.opacities, device='cuda')

        condition_dim = 3
        # taken from section 2 of https://www.sdiolatz.info/ndg-fitting/static/files/ndg-supp.pdf
        # todo: this can be done in a shader instead for much faster perf
        covariances = torch.tensor(self.covariances, device='cuda')
        self.cuda_covariances_11 = covariances[:, :condition_dim, :condition_dim]
        self.cuda_covariances_12 = covariances[:, :condition_dim, condition_dim:]
        self.cuda_covariances_21 = covariances[:, condition_dim:, :condition_dim]
        self.cuda_covariances_22 = covariances[:, condition_dim:, condition_dim:]

        self.cuda_covariances_22_inv = self.cuda_covariances_22.inverse()
        self.cuda_covariances_regr = torch.bmm(self.cuda_covariances_12, self.cuda_covariances_22_inv)
        self.cuda_covariances_conditioned = self.cuda_covariances_11 - torch.bmm(self.cuda_covariances_regr, self.cuda_covariances_21)
        self.cuda_covariances_conditioned_flat = self.cuda_covariances_conditioned.flatten(start_dim=1)
        self.covariances_conditioned_flat = self.cuda_covariances_conditioned_flat.cpu().numpy()

    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.means, self.covariances, self.colors, self.opacities], axis=-1)
        return np.ascontiguousarray(ret)
    def __len__(self):
        return len(self.means)
    
    def to_gaussian3d(self, camera: util.Camera):
        #print("Starting math")
        camera_pos = torch.tensor(camera.position, device='cuda')
        view_directions = self.cuda_means[:, :3] - camera_pos
        view_directions_unit = view_directions / view_directions.norm(dim=1, keepdim=True)

       
        condition_dim = 3
        means_1 = self.cuda_means[:, :condition_dim]
        means_2 = self.cuda_means[:, condition_dim:]
        x = view_directions_unit-means_2

        pdf_conditioned = torch.exp(-torch.abs(torch.bmm(torch.bmm(x.unsqueeze(-2), self.cuda_covariances_22_inv), x.unsqueeze(-1))))[:, :, 0]
        means_conditioned = means_1 + torch.bmm(self.cuda_covariances_regr, x.unsqueeze(-1))[:, :, 0]
        # flatten so it can be concatenated later in numpy
        opacities_conditioned = self.cuda_opacities * pdf_conditioned
        

        #print(f"pdf conditioned shape: {pdf_conditioned.shape}")
        #print(f"means conditioned shape: {means_conditioned.shape}")
        #print(f"covariance conditioned shape: {self.cuda_covariances_conditioned.shape}")
        #print(f"opacities conditioned shape: {opacities_conditioned.shape}")

        #print("Done with math")
        return GaussianData3D(means_conditioned.cpu().numpy(), self.covariances_conditioned_flat, self.colors, opacities_conditioned.cpu().numpy())


        # view_directions = self.means[:, :3] - np.asarray(camera.position)
        # view_directions_unit = view_directions / np.linalg.norm(view_directions, axis=1, keepdims=True)

        # condition_dim = 3

        # # taken from section 2 of https://www.sdiolatz.info/ndg-fitting/static/files/ndg-supp.pdf
        # # todo: this can be done in a shader instead for much faster perf
        # covariances_11 = self.covariances[:, :condition_dim, :condition_dim]
        # covariances_12 = self.covariances[:, :condition_dim, condition_dim:]
        # covariances_21 = self.covariances[:, condition_dim:, :condition_dim]
        # covariances_22 = self.covariances[:, condition_dim:, condition_dim:]

        # covariances_22_inv = np.linalg.inv(covariances_22)

        # covariances_regr = np.matmul(covariances_12, covariances_22_inv)

        # means_1 = self.means[:, :condition_dim]
        # means_2 = self.means[:, condition_dim:]
        # x = view_directions_unit-means_2
        # print(x, flush=True)

        # pdf_cond = np.exp(-np.abs(np.matmul(np.matmul(x.unsqueeze(-2), covariances_22_inv), x.unsqueeze(-1))))[:, :, 0]

        # m_cond = means_1 + np.matmul(covariances_regr, x.unsqueeze(-1))[:, :, 0]
        # covariances_cond = covariances_11 - torch.bmm(covariances_regr, covariances_21)


    

@dataclass
class GaussianData3D:
    xyz: np.ndarray
    cov3d: np.ndarray
    colors: np.ndarray
    opacity: np.ndarray
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.cov3d, self.colors, self.opacity], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)

@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]


def naive_gaussian():
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)
    gau_s = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = np.array([
        1, 0, 1, 
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1, 
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = (gau_c - 0.5) / 0.28209
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)
    return GaussianData(
        gau_xyz,
        gau_rot,
        gau_s,
        gau_a,
        gau_c
    )    

def load_ply_6d(path):
    plydata = PlyData.read(path)
    vertex_data = plydata['vertex'].data
    means = np.stack((np.asarray(plydata.elements[0]["m_0"]),
                      np.asarray(plydata.elements[0]["m_1"]),
                      np.asarray(plydata.elements[0]["m_2"]),
                      np.asarray(plydata.elements[0]["m_3"]),
                      np.asarray(plydata.elements[0]["m_4"]),
                      np.asarray(plydata.elements[0]["m_5"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    # extract 6d covariance matrices into (num_gaussians, 6, 6) shape np array
    cov_keys = [f'cov_{i}_{j}' for i in range(6) for j in range(6)]
    cov_values = np.array([vertex_data[key] for key in cov_keys]).T
    covariances = cov_values.reshape(-1, 6, 6)
    colors = np.stack((np.asarray(plydata.elements[0]["f_0"]),
                      np.asarray(plydata.elements[0]["f_1"]),
                      np.asarray(plydata.elements[0]["f_2"])), axis=1)
    
    return GaussianData6D(means, covariances, colors, opacities)



def load_ply(path):
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    opacities = 1/(1 + np.exp(- opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                        features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)
    return GaussianData(xyz, rots, scales, opacities, shs)

if __name__ == "__main__":
    gs = load_ply_6d("C:\\Users\\samue\\Workspace\\ndg-fitting\\models\\splatting\\20240802-181703\\point_cloud10000.ply")
    gs3d = gs.to_gaussian3d(util.Camera(720, 1080))
    flat = gs3d.flat()
    print(flat.shape)
    print(flat)
