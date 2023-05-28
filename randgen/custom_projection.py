'''
Hanjiang Hu, for VNN-COMP 2023
May 2023
'''

import argparse
import torch, pickle
import cupy as cp
import numpy as np
from torchvision import datasets
import torch.onnx
import onnxruntime as _ort
from onnxruntime_extensions import (onnx_op, get_library_path as _get_library_path)


def _load_pickle(path: str) -> dict:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def setup_dataset(onnx_file):

    proj_type = onnx_file.split('_')[1]
    dataset_path = f"../dataset/metaroom_{proj_type}/certify"
    dataset = datasets.DatasetFolder(dataset_path, _load_pickle, extensions="pkl")

    intrinsic_list = []
    extrinsic_list = []
    pc_list = []
    for i in range(len(dataset)):
        (x, label) = dataset[i]
        intrinsic_matrix = cp.asnumpy(x["intrinsic_matrix"]).astype(np.float16)
        intrinsic_matrix[0][0] /= 8
        intrinsic_matrix[1][1] /= 8
        intrinsic_matrix[0][2] /= 8
        intrinsic_matrix[1][2] /= 8

        intrinsic_matrix[0][0] = intrinsic_matrix[0][0] * 2 / 5
        intrinsic_matrix[1][1] = intrinsic_matrix[1][1]  * 4 / 9
        intrinsic_matrix[0][2] = intrinsic_matrix[0][2] * 2 / 5 - 4
        intrinsic_matrix[1][2] = intrinsic_matrix[1][2] * 4 / 9 - 4

        extrinsic_matrix = cp.asnumpy(x["pose"]).astype(np.float16)
        complete_3D_oracle = cp.asnumpy(x["point_cloud"]).astype(np.float16)

        intrinsic_list.append(intrinsic_matrix)
        extrinsic_list.append(extrinsic_matrix)
        pc_list.append(complete_3D_oracle)

    intrinsic_np = np.array(intrinsic_list)
    extrinsic_np = np.array(extrinsic_list)


    intrinsic_tensor = torch.from_numpy(intrinsic_np).to(torch.float32)
    extrinsic_tensor = torch.from_numpy(extrinsic_np).to(torch.float32)
    pc_tensor = pc_list
    return intrinsic_tensor, extrinsic_tensor, pc_tensor
def filter_frustum(x_ge_0_index, project_positions_flat, project_positions_float, project_positions, points_start,
                   colors):
    project_positions_flat = project_positions_flat[x_ge_0_index]
    project_positions_float = project_positions_float[x_ge_0_index]
    project_positions = project_positions[x_ge_0_index]
    points_start = points_start[x_ge_0_index]
    colors = colors[x_ge_0_index]
    return project_positions_flat, project_positions_float, project_positions, points_start, colors
def find_new_extrinsic_matrix(extrinsic_matrix, axis, alpha):
    if axis == 'tz':
        R = torch.tensor([[1.0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = torch.cat((torch.tensor([0.0, 0.0]), alpha)).unsqueeze(dim=0)
    elif axis == 'tx':
        R = torch.tensor([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = torch.cat((alpha, torch.tensor([0.0, 0.0]))).unsqueeze(dim=0)
    elif axis == 'ty':
        R = torch.tensor([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = torch.cat((torch.tensor([0.0]), alpha, torch.tensor([0.0]))).unsqueeze(dim=0)
    elif axis == 'rz':
        R = torch.stack([
            torch.cat((torch.cos(alpha), -torch.sin(alpha), torch.tensor([0.0]))),
            torch.cat((torch.sin(alpha), torch.cos(alpha), torch.tensor([0.0]))),
            torch.tensor([0.0, 0, 1])
        ])
        t = torch.tensor([0.0, 0, 0]).unsqueeze(dim=0)
    elif axis == 'ry':
        R = torch.stack([
            torch.cat((torch.cos(alpha), torch.tensor([0.0]), torch.sin(alpha))),
            torch.tensor([0.0, 1, 0]),
            torch.cat((-torch.sin(alpha), torch.tensor([0.0]), torch.cos(alpha)))
        ])
        t = torch.tensor([0.0, 0, 0]).unsqueeze(dim=0)
    else:
        R = torch.stack([
            torch.tensor([1.0, 0, 0]),
            torch.cat((torch.tensor([0.0]), torch.cos(alpha),  -torch.sin(alpha))),
            torch.cat((torch.tensor([0.0]), torch.sin(alpha),  torch.cos(alpha)))
        ])
        t = torch.tensor([0.0, 0, 0]).unsqueeze(dim=0)
    rel_matrix = torch.cat((torch.cat((R, t.T), dim=1), torch.tensor([0.0, 0, 0, 1]).unsqueeze(dim=0))).to(torch.float32)
    return (extrinsic_matrix @ rel_matrix)

def projection_oracle(point_cloud_npy, extrinsic_matrix, intrinsic_matrix):
    # load point cloud
    point_cloud = point_cloud_npy
    original_positions = point_cloud[:, 0: 3]
    colors = point_cloud[:, 3: 6]

    positions = torch.cat((original_positions, torch.ones((original_positions.shape[0], 1))), dim=1)
    points_start = (torch.inverse(extrinsic_matrix)[0: 3] @ positions.T).T
    project_positions = intrinsic_matrix @ torch.inverse(extrinsic_matrix)[0: 3] @ positions.T
    project_positions = project_positions.T
    project_positions_float = project_positions[:, 0:2] / project_positions[:, 2:3]

    project_positions_flat = torch.floor(project_positions_float).short()

    h, w = 2 * int(intrinsic_matrix[1][2]), 2 * int(intrinsic_matrix[0][2])

    x_ge_0_index = np.where(project_positions_flat[:, 0] >= 0)
    project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
        x_ge_0_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
    y_ge_0_index = np.where(project_positions_flat[:, 1] >= 0)
    project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
        y_ge_0_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
    x_l_w_index = np.where(project_positions_flat[:, 0] < w)
    project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
        x_l_w_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
    y_l_h_index = np.where(project_positions_flat[:, 1] < h)
    project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
        y_l_h_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)
    d_g_0_index = np.where(project_positions[:, 2] > 0)
    project_positions_flat, project_positions_float, project_positions, points_start, colors = filter_frustum(
        d_g_0_index, project_positions_flat, project_positions_float, project_positions, points_start, colors)


    return project_positions_flat, project_positions_float, project_positions, points_start, colors  # , point_cloud

def find_2d_image(project_positions_flat, project_positions, colors, intrinsic_matrix):
    # get color image
    h, w = 2 * int(intrinsic_matrix[1][2]), 2 * int(intrinsic_matrix[0][2])

    image = torch.ones((h, w, 3))
    project_positions[:, :2] = project_positions_flat
    colored_positions = torch.cat((project_positions, colors), dim=1).to(torch.float32)
    unique = torch.unique(project_positions_flat, dim=0)
    unique = unique.short()
    unique_ = torch.repeat_interleave(unique[None, :], project_positions_flat.shape[0], dim=0)
    project_positions_flat_ = torch.repeat_interleave(project_positions_flat[:, None, :], unique.shape[0], dim=1)
    colored_positions_ = torch.repeat_interleave(colored_positions[:, None, :], unique.shape[0], dim=1)
    same_positions_xy_index = (project_positions_flat_ == unique_)[:, :, 0] & (project_positions_flat_ == unique_)[:, :,
                                                                              1]
    depths_all = torch.where(same_positions_xy_index, colored_positions_[:, :, 2], torch.tensor(float('inf')))

    filtered_positions = colored_positions_[torch.argmin(depths_all, dim=0), torch.arange(depths_all.shape[1])]

    image[filtered_positions[:, 1:2].long().T[0], filtered_positions[:, :1].long().T[0], :] = filtered_positions[:, 3:6]
    return image

def custom_ort_session(onnx_file):
    intrinsic_tensor, extrinsic_tensor, pc_tensor = setup_dataset(onnx_file)
    const = int(onnx_file.split('_')[2])
    proj_type = onnx_file.split('_')[1]
    @onnx_op(op_type='ProjectionOp', domain='ai.onnx.contrib')
    def ProjectionOp(x):
        x = torch.from_numpy(x)
        x = x[0]
        extrinsic_matrix_origin = extrinsic_tensor[const]
        alpha = torch.norm(x, float('inf')).unsqueeze(0)
        complete_3D_oracle = torch.tensor(pc_tensor[const])
        intrinsic_m = intrinsic_tensor[const]
        extrinsic_m = find_new_extrinsic_matrix(extrinsic_matrix_origin, proj_type, alpha)

        project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
            complete_3D_oracle, extrinsic_m, intrinsic_m)
        now_img = find_2d_image(project_positions_flat, project_positions, colors, intrinsic_m)

        now_img = torch.transpose(now_img, 1, 2)
        now_img = torch.transpose(now_img, 0, 1)  # .type(torch.cuda.FloatTensor)
        now_img = now_img.unsqueeze(dim=0).float()
        return now_img

    so = _ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())

    ort_session = _ort.InferenceSession(onnx_file, so)

    return  ort_session
