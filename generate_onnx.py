
import os
import sys
# sys.path.append('vnn_comp2023')
# sys.path.append('')

import math

# evaluate a smoothed classifier on a dataset
import argparse
# import setGPU
from datasets_vnn import get_dataset, DATASETS, get_num_classes, get_normalize_layer
from core import SemanticSmooth
from time import time
import random
# import setproctitle
import torch
import torch.nn as nn
import torchvision
import datetime
from tensorboardX import SummaryWriter

from architectures import get_architecture
from architectures_denoise import get_architecture_denoise
from transformers_ import RotationTransformer
from transformers_ import gen_transformer, DiffResolvableProjectionTransformer
from transforms import visualize
import numpy as np
import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from collections import OrderedDict
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
from vnn_models import Models
# from auto_LiRPA.operators import *
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
from torch.onnx import register_custom_op_symbolic
import torch._C as _C

TensorProtoDataType = _C._onnx.TensorProtoDataType
OperatorExportTypes = _C._onnx.OperatorExportTypes

'''
CUDA_VISIBLE_DEVICES=3 python generate_onnx.py metaroom diff_resolvable_tz models/metaroom/cnn_4layer_new/vnn_all/tv_noise_0.0_lirpa/best_checkpoint.pth.tar     --cpu --outfile 4cnn_tz
CUDA_VISIBLE_DEVICES=3 python generate_onnx.py metaroom diff_resolvable_tz models/metaroom/cnn_6layer_new/vnn_all/tv_noise_0.0_lirpa/best_checkpoint.pth.tar     --cpu --outfile 6cnn_tz

CUDA_VISIBLE_DEVICES=2 python generate_onnx.py metaroom diff_resolvable_ry models/metaroom/cnn_4layer_new/vnn_all/tv_noise_0.0_lirpa/best_checkpoint.pth.tar     --cpu --outfile 4cnn_ry
CUDA_VISIBLE_DEVICES=2 python generate_onnx.py metaroom diff_resolvable_ry models/metaroom/cnn_6layer_new/vnn_all/tv_noise_0.0_lirpa/best_checkpoint.pth.tar     --cpu --outfile 6cnn_ry
'''

EPS = 1e-6

parser = argparse.ArgumentParser(description='Strict rotation certify')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument('transtype', type=str, help='type of projective transformations',
                    choices=['resolvable_tz', 'resolvable_tx', 'resolvable_ty', 'resolvable_rz', 'resolvable_rx',
                             'resolvable_ry', 'diff_resolvable_tz', 'diff_resolvable_tx', 'diff_resolvable_ty', 'diff_resolvable_rz', 'diff_resolvable_rx',
                             'diff_resolvable_ry'])
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")

parser.add_argument("--noise_sd", type=float,default=0.0, help="pixel gaussian noise hyperparameter")
parser.add_argument("--noise_b", type=float, default=0.0, help="noise hyperparameter for brightness shift dimension")
parser.add_argument("--noise_k", type=float, default=0.0, help="noise hyperparameter for brightness scaling dimension")
parser.add_argument("--l2_r", type=float, default=0.0, help="additional l2 magnitude to be tolerated")
parser.add_argument("--aliasfile", type=str,default='None', help='output of alias data')
parser.add_argument("--outfile", type=str, help="output onnx")
parser.add_argument("--b", type=float, default=0.0, help="brightness shift requirement")
parser.add_argument("--k", type=float, default=0.0, help="brightness scaling requirement")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--start", type=int, default=0, help="start before skipping how many examples")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", default="certify", help="train or test set")
parser.add_argument("--N0", type=int, default=500)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--slice", type=int, default=1000, help="number of angle slices")
parser.add_argument("--alpha", type=float, default=0.01, help="failure probability")
parser.add_argument("--partial", type=float, default=180.0, help="certify +-partial degrees")
parser.add_argument("--verbstep", type=int, default=10, help="output frequency")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--N_partitions", type=int, default=7001, help="number of partitions to use")
parser.add_argument("--saved_path", type=str, default="", help='output of alias data')
parser.add_argument("--factor", type=float, default=1.0, help="factors to rescale from original radii")
parser.add_argument('--denoiser', type=str, default='',
                    help='Path to a denoiser to attached before classifier during certificaiton.')
parser.add_argument('--training_type', type=str, default='vnn',
                    help='all, vnn or separate')
parser.add_argument("--cpu", action='store_true')
parser.add_argument("--tune_crown", action='store_true')
parser.add_argument("--fix_crown", action='store_true')
parser.add_argument("--loose_bound", action='store_true')
parser.add_argument("--sml_img", action='store_false')

args = parser.parse_args()
torch.set_num_threads(5)

dataset = get_dataset(args.dataset, args.split, args.transtype)
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

    # intrinsic_matrix[0][0] /= 22.5
    # intrinsic_matrix[1][1] /= 160/7.0
    # intrinsic_matrix[0][2] /= 22.5
    # intrinsic_matrix[1][2] /= 160/7.0
    intrinsic_matrix[0][0] = intrinsic_matrix[0][0] * 2 / 5
    intrinsic_matrix[1][1] = intrinsic_matrix[1][1]  * 4 / 9
    intrinsic_matrix[0][2] = intrinsic_matrix[0][2] * 2 / 5 - 4
    intrinsic_matrix[1][2] = intrinsic_matrix[1][2] * 4 / 9 - 4

    extrinsic_matrix = cp.asnumpy(x["pose"]).astype(np.float16)
    complete_3D_oracle = cp.asnumpy(x["point_cloud"]).astype(np.float16)
    # print(len(complete_3D_oracle))
    # print(intrinsic_matrix)
    intrinsic_list.append(intrinsic_matrix)
    extrinsic_list.append(extrinsic_matrix)
    pc_list.append(complete_3D_oracle)
# print(intrinsic_list)
intrinsic_np = np.array(intrinsic_list)
extrinsic_np = np.array(extrinsic_list)
# pc_np = np.array(pc_list)
if args.cpu:
    intrinsic_tensor = torch.from_numpy(intrinsic_np).to(torch.float32)
    extrinsic_tensor = torch.from_numpy(extrinsic_np).to(torch.float32)
else:
    intrinsic_tensor = torch.from_numpy(intrinsic_np).cuda().to(torch.float16)
    extrinsic_tensor = torch.from_numpy(extrinsic_np).cuda().to(torch.float16)
pc_tensor = pc_list

# if args.cpu:
#     img_test_list = torch.ones([120, 7001, 90, 160, 3]).to(torch.float32)
# else:
#     img_test_list = torch.ones([120, 7001, 90, 160, 3]).to(torch.float16)
# img_test_list = torch.transpose(img_test_list, 3, 4)
# img_test_list = torch.transpose(img_test_list, 2, 3)#.type(torch.cuda.FloatTensor)

# for i in range(120):
#     for j in range(7001):
#         image_i_j = matplotlib.image.imread(args.saved_path + '/%03d' % i + '/%05d.png' % j)
#         now_img = torch.as_tensor(image_i_j[:, :, :3]).to(torch.float16)
#         img_test_list[i][j] = now_img
# img_test_list.cuda()
class LinearBound:
    def __init__(
            self, lw=None, lb=None, uw=None, ub=None, lower=None, upper=None,
            from_input=None, x_L=None, x_U=None, offset=0, tot_dim=None):
        self.lw = lw
        self.lb = lb
        self.uw = uw
        self.ub = ub
        self.lower = lower
        self.upper = upper
        self.from_input = from_input
        self.x_L = x_L
        self.x_U = x_U
        # Offset for input variables. Used for batched forward bound
        # propagation.
        self.offset = offset
        if tot_dim is not None:
            self.tot_dim = tot_dim
        elif lw is not None:
            self.tot_dim = lw.shape[1]
        else:
            self.tot_dim = 0

    def is_single_bound(self):
        """Check whether the linear lower bound and the linear upper bound are
        the same."""
        if (self.lw is not None and self.uw is not None
                and self.lb is not None and self.ub is not None):
            return (self.lw.data_ptr() == self.uw.data_ptr()
                and self.lb.data_ptr() == self.ub.data_ptr()
                and self.x_L is not None and self.x_U is not None)
        else:
            return True

def filter_frustum(x_ge_0_index, project_positions_flat, project_positions_float, project_positions, points_start,
                   colors):  # , point_cloud=np.array(0)):
    project_positions_flat = project_positions_flat[x_ge_0_index]
    project_positions_float = project_positions_float[x_ge_0_index]
    project_positions = project_positions[x_ge_0_index]
    points_start = points_start[x_ge_0_index]
    colors = colors[x_ge_0_index]
    # if len(point_cloud.shape) != 0:
    #     point_cloud = point_cloud[x_ge_0_index]
    # print(points_start.shape)
    return project_positions_flat, project_positions_float, project_positions, points_start, colors  # , point_cloud
def find_new_extrinsic_matrix(extrinsic_matrix, axis, alpha):
    # alpha = np.asnumpy(alpha)
    if axis == 'tz':
        # R = np.array([[1, 0, 0],
        #               [0, 1, 0],
        #               [0, 0, 1]])
        # t = np.array([[0, 0, alpha]])
        R = torch.tensor([[1.0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = torch.cat((torch.tensor([0.0, 0.0]), alpha)).unsqueeze(dim=0)
    elif axis == 'tx':
        # R = np.array([[1, 0, 0],
        #               [0, 1, 0],
        #               [0, 0, 1]])
        # t = np.array([[alpha, 0, 0]])
        R = torch.tensor([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = torch.cat((alpha, torch.tensor([0.0, 0.0]))).unsqueeze(dim=0)
    elif axis == 'ty':
        # R = np.array([[1, 0, 0],
        #               [0, 1, 0],
        #               [0, 0, 1]])
        # t = np.array([[0, alpha, 0]])
        R = torch.tensor([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = torch.cat((torch.tensor([0.0]), alpha, torch.tensor([0.0]))).unsqueeze(dim=0)
    elif axis == 'rz':
        # R = np.array([[numpy.cos(alpha), -numpy.sin(alpha), 0],
        #               [numpy.sin(alpha), numpy.cos(alpha), 0],
        #               [0, 0, 1]])
        # t = np.array([[0, 0, 0]])
        R = torch.stack([
            torch.cat((torch.cos(alpha), -torch.sin(alpha), torch.tensor([0.0]))),
            torch.cat((torch.sin(alpha), torch.cos(alpha), torch.tensor([0.0]))),
            torch.tensor([0.0, 0, 1])
        ])
        t = torch.tensor([0.0, 0, 0]).unsqueeze(dim=0)
    elif axis == 'ry':
        # R = np.array([[numpy.cos(alpha), 0, numpy.sin(alpha)],
        #               [0, 1, 0],
        #               [-numpy.sin(alpha), 0, numpy.cos(alpha)]])
        # t = np.array([[0, 0, 0]])
        R = torch.stack([
            torch.cat((torch.cos(alpha), torch.tensor([0.0]), torch.sin(alpha))),
            torch.tensor([0.0, 1, 0]),
            torch.cat((-torch.sin(alpha), torch.tensor([0.0]), torch.cos(alpha)))
        ])
        t = torch.tensor([0.0, 0, 0]).unsqueeze(dim=0)
    else:
        # R = np.array([[1, 0, 0],
        #               [0, numpy.cos(alpha), -numpy.sin(alpha)],
        #               [0, numpy.sin(alpha), numpy.cos(alpha)]])
        # t = np.array([[0, 0, 0]])
        R = torch.stack([
            torch.tensor([1.0, 0, 0]),
            torch.cat((torch.tensor([0.0]), torch.cos(alpha),  -torch.sin(alpha))),
            torch.cat((torch.tensor([0.0]), torch.sin(alpha),  torch.cos(alpha)))
        ])
        t = torch.tensor([0.0, 0, 0]).unsqueeze(dim=0)
    if args.cpu:
        rel_matrix = torch.cat((torch.cat((R, t.T), dim=1), torch.tensor([0.0, 0, 0, 1]).unsqueeze(dim=0))).to(torch.float32)
    else:
        rel_matrix = torch.cat((torch.cat((R, t.T), dim=1), torch.tensor([0.0, 0, 0, 1]).unsqueeze(dim=0))).cuda().to(
            torch.float16)
    return (extrinsic_matrix @ rel_matrix)

def projection_oracle(point_cloud_npy, extrinsic_matrix, intrinsic_matrix, k_ambiguity, round=False, no_filter_frustum=False):
    # load point cloud

    point_cloud = point_cloud_npy
    original_positions = point_cloud[:, 0: 3]
    colors = point_cloud[:, 3: 6]

    positions = torch.cat((original_positions, torch.ones((original_positions.shape[0], 1))), dim=1)
    # print(intrinsic_matrix, extrinsic_matrix)
    points_start = (torch.inverse(extrinsic_matrix)[0: 3] @ positions.T).T
    project_positions = intrinsic_matrix @ torch.inverse(extrinsic_matrix)[0: 3] @ positions.T
    project_positions = project_positions.T
    project_positions_float = project_positions[:, 0:2] / project_positions[:, 2:3]
    h, w = 2 * int(intrinsic_matrix[1][2]), 2 * int(intrinsic_matrix[0][2])

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

def find_2d_image(project_positions_flat, project_positions, points_start, colors, intrinsic_matrix,
                  need_second_img=False, no_filter_frustum=False):


    # get color image
    h, w = 2 * int(intrinsic_matrix[1][2]), 2 * int(intrinsic_matrix[0][2])
    # print(h, w)
    image = torch.ones((h, w, 3))
    # dists = np.inf * np.ones((h, w))

    # pixel_points = [[[] for j in range(w)] for i in range(h)]
    # pixel_closest_point = [[-1 for j in range(w)] for i in range(h)]

    # second_dists = np.inf * np.ones((h, w))

    # second_pixel_points = [ [ [] for j in range(w) ] for i in range(h) ]
    # second_pixel_closest_point = [[-1 for j in range(w)] for i in range(h)]
    # second_image = np.ones((h, w, 3))

    project_positions[:, :2] = project_positions_flat
    if args.cpu:
        colored_positions = torch.cat((project_positions, colors), dim=1).to(torch.float32)
    else:
        colored_positions = torch.cat((project_positions, colors), dim=1).cuda().to(torch.float16)
    # img_a = project_positions[:, :2] #image[project_positions[:, :1].astype("int").T[0], project_positions[:, 1:2].astype("int").T[0], :]
    unique = torch.unique(project_positions_flat, dim=0)
    # unique = np.unique(project_positions_flat, axis=0)
    filtered_list = []
    # print(np.where(project_positions_flat == unique))
    # print("project_positions_flat", project_positions_flat.shape)
    # print("unique", unique.shape)
    # print("colored_positions", colored_positions.shape)
    # print(np.max(project_positions_flat),np.max(unique))

    unique = unique.short()
    unique_ = torch.repeat_interleave(unique[None, :], project_positions_flat.shape[0], dim=0)
    project_positions_flat_ = torch.repeat_interleave(project_positions_flat[:, None, :], unique.shape[0], dim=1)
    colored_positions_ = torch.repeat_interleave(colored_positions[:, None, :], unique.shape[0], dim=1)
    # print("project_positions_flat", project_positions_flat_.dtype)
    # print("unique", unique_.dtype)
    # print("colored_positions", colored_positions_.dtype)
    same_positions_xy_index = (project_positions_flat_ == unique_)[:, :, 0] & (project_positions_flat_ == unique_)[:, :,
                                                                              1]
    depths_all = torch.where(same_positions_xy_index, colored_positions_[:, :, 2], torch.tensor(float('inf')))

    filtered_positions = colored_positions_[torch.argmin(depths_all, dim=0), torch.arange(depths_all.shape[1])]
    '''
    for unique_item in unique:
        # print("unique_item", unique_item)
        # print("######", project_positions_flat)

        points_index_with_same_pixel_x = np.where(project_positions_flat[:, 0] == unique_item[0])[0]
        project_positions_flat_x = project_positions_flat[points_index_with_same_pixel_x]
        colored_positions_x = colored_positions[points_index_with_same_pixel_x]
        # print(unique_item[0], project_positions_flat_x, colored_positions_x)

        points_index_with_same_pixel_y = np.where(project_positions_flat_x[:, 1] == unique_item[1])[0]
        project_positions_flat_xy = project_positions_flat_x[points_index_with_same_pixel_y]
        colored_positions_xy = colored_positions_x[points_index_with_same_pixel_y]

        # print("$$$$$$$$$$$",unique_item, project_positions_flat[points_index_with_same_pixel][0])
        # print(b,unique_item,project_positions_flat.shape)
        # a = colored_positions[points_index_with_same_pixel] #numpy.unique(np.asnumpy(points_index_with_same_pixel), axis=0)
        # print("a", colored_positions_xy)
        depth = colored_positions_xy[:, 2]
        # print("depth", depth.shape)
        min_index = np.argmin(depth)
        # min_index_top_1, min_index_top_2 = np.argsort(depth)[0], np.argsort(depth)[1]

        filtered_list.append(colored_positions_xy[min_index])

        # assert depth.shape[0] == colored_positions_xy.shape[0]
        # assert colored_positions.shape[0] == project_positions_flat.shape[0]

    filtered_positions = np.array(filtered_list)
    # print(filtered_positions[:, :3])
    '''
    # print(filtered_positions)
    image[filtered_positions[:, 1:2].long().T[0], filtered_positions[:, :1].long().T[0],
    :] = filtered_positions[:, 3:6]
    # print("project_positions_flat", project_positions_flat.shape)
    # print("project_positions", project_positions.shape)
    # print("colored_positions", colored_positions.shape)
    # print("unique", unique.shape)
    # print("filtered_positions", filtered_positions.shape)



    return image




""" Step 1: Define a `torch.autograd.Function` class to declare and implement the
computation of the operator. """
class ProjectionOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, const):
        """ In this function, define the arguments and attributes of the operator.
        "custom::PlusConstant" is the name of the new operator, "x" is an argument
        of the operator, "const_i" is an attribute which stands for "c" in the operator.
        There can be multiple arguments and attributes. For attribute naming,
        use a suffix such as "_i" to specify the data type, where "_i" stands for
        integer, "_t" stands for tensor, "_f" stands for float, etc. """
        return g.op('ai.onnx.contrib::ProjectionOp', x, const_i=const)

    @staticmethod
    def forward(ctx, x, const):
        """ In this function, implement the computation for the operator, i.e.,
        f(x) = x + c in this case. """
        # x = x[0]
        # index = 7000 * ((x + args.partial) / (2 * args.partial))
        # # now_img = img_test_list[const][index.floor().long().cpu()].cuda().squeeze().squeeze()
        # this_img_list = img_test_list[const]
        # now_img = this_img_list[index.floor().long().cpu()]
        # print("%%%%%%%%%%%%%%%%%%%%", now_img.shape)

        x = x[0]
        extrinsic_matrix_origin = extrinsic_tensor[const]
        alpha = torch.norm(x, float('inf')).unsqueeze(0)
        complete_3D_oracle = torch.tensor(pc_tensor[const])
        intrinsic_m = intrinsic_tensor[const]
        extrinsic_m = find_new_extrinsic_matrix(extrinsic_matrix_origin, args.transtype[-2:], alpha)

        project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
            complete_3D_oracle, extrinsic_m, intrinsic_m, k_ambiguity=-1)  # k_ambiguity)
        now_img = find_2d_image(project_positions_flat, project_positions, points_start, colors,
                                            intrinsic_m, need_second_img=False)

        now_img = torch.transpose(now_img, 1, 2)
        now_img = torch.transpose(now_img, 0, 1)#.type(torch.cuda.FloatTensor)
        now_img = now_img.unsqueeze(dim=0).float()
        return now_img

""" Step 2: Define a `torch.nn.Module` class to declare a module using the defined
custom operator. """
class Projection(nn.Module):
    def __init__(self, const=1):
        super().__init__()
        self.const = const

    def forward(self, x):
        """ Use `PlusConstantOp.apply` to call the defined custom operator. """
        return ProjectionOp.apply(x, self.const)

# register_custom_op_symbolic('custom::Projection', ProjectionOp.symbolic, 1)
def DataParallel2CPU(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:9] == "1.module.":
            k = "1." + k[9:]
        new_state_dict[k] = v
    return new_state_dict

if __name__ == '__main__':

    orig_alpha = args.alpha
    args.alpha /= (args.slice)# * (2.0 * args.partial) / 360.0 + 1)


    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    if args.training_type == 'all':
        # load the base classifier
        checkpoint = torch.load(args.base_classifier)
        base_classifier = get_architecture(checkpoint["arch"], args.dataset,cpu=True)
        if checkpoint["arch"] == 'resnet50' or checkpoint["arch"] == 'resnet101':
            # assert 1==2
            try:
                base_classifier.load_state_dict(checkpoint['state_dict'])
            except:
                # base_classifier = torchvision.models.resnet50(pretrained=False).cuda() if checkpoint["arch"] == 'resnet50' else torchvision.models.resnet101(pretrained=False).cuda()
                #
                # # fix
                # normalize_layer = get_normalize_layer(args.dataset).cuda()
                base_classifier = torchvision.models.resnet50(pretrained=False) if checkpoint[
                                                                                              "arch"] == 'resnet50' else torchvision.models.resnet101(
                    pretrained=False)
                normalize_layer = get_normalize_layer(args.dataset, cpu=True)
                base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
                print("$$$$$$$$$$$$")
        base_classifier.load_state_dict(checkpoint['state_dict'])
        if args.denoiser != '':
            checkpoint_denoiser = torch.load(args.denoiser)
            if "off-the-shelf-denoiser" in args.denoiser:
                denoiser = get_architecture_denoise('orig_dncnn', args.dataset)
                denoiser.load_state_dict(checkpoint_denoiser)
            else:
                denoiser = get_architecture_denoise(checkpoint_denoiser['arch'], args.dataset)
                denoiser.load_state_dict(checkpoint_denoiser['state_dict'])
            base_classifier = torch.nn.Sequential(denoiser, base_classifier)
            print("denoiser added")
    elif args.training_type == 'vnn':
        checkpoint = torch.load(args.base_classifier)

        # if checkpoint["arch"] == 'resnet50':
        #     base_classifier = torchvision.models.resnet50(False)
        #     for name, module in base_classifier.named_modules():
        #         if isinstance(module, torch.nn.MaxPool2d):
        #             base_classifier._modules[name] = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        #     print("###############################3")
        if checkpoint["arch"] == 'resnet50':
            # if args.training_type != 'vnn':
            #     base_classifier = torchvision.models.resnet50(False).cuda()
            # else:
            base_classifier = Models['resnet50']()

            # for name, module in model.named_modules():
            #     if isinstance(module, torch.nn.MaxPool2d):
            #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
            print("###############################3")
        elif checkpoint["arch"] == 'resnet101':
            # if args.training_type != 'vnn':
            #     base_classifier = torchvision.models.resnet101(False).cuda()
            # else:
            base_classifier = Models['resnet101']()
            # for name, module in model.named_modules():
            #     if isinstance(module, torch.nn.MaxPool2d):
            #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
            print("###############################3")
        elif checkpoint["arch"] == 'resnet18':
            # if args.training_type != 'vnn':
            #     base_classifier = torchvision.models.resnet18(False).cuda()
            # else:
            base_classifier = Models['resnet18']()
            # for name, module in model.named_modules():
            #     if isinstance(module, torch.nn.MaxPool2d):
            #         model._modules[name] = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0).cuda()
            print("###############################3")
        elif checkpoint["arch"] == 'resnet34':
            # if args.training_type != 'vnn':
            #     base_classifier = torchvision.models.resnet34(False).cuda()
            # else:
            base_classifier = Models['resnet34']()
        elif checkpoint["arch"] == 'cnn_4layer':
            base_classifier = Models['cnn_4layer'](in_ch=3, in_dim=(32, 56))
        elif checkpoint["arch"] == 'cnn_6layer':
            base_classifier = Models['cnn_6layer'](in_ch=3, in_dim=(32, 56))
        elif checkpoint["arch"] == 'cnn_7layer':
            base_classifier = Models['cnn_7layer'](in_ch=3, in_dim=(32, 56))
        elif checkpoint["arch"] == 'cnn_7layer_bn':
            base_classifier = Models['cnn_7layer_bn'](in_ch=3, in_dim=(32, 56))

        elif checkpoint["arch"] == 'mlp_5layer':
            base_classifier = Models['mlp_5layer'](in_ch=3, in_dim=(32, 56))
        else:
            base_classifier = None
            print('not supported')
        base_classifier.load_state_dict(checkpoint['state_dict'])
        base_classifier = base_classifier.cpu()
        print(f"loaded {checkpoint['arch']}")
    else:
        # load the base classifier
        checkpoint = torch.load(args.base_classifier)
        base_classifier = get_architecture(checkpoint["arch"], args.dataset,cpu=True)
        print('arch:', checkpoint['arch'])

        if checkpoint["arch"] == 'resnet50' and args.dataset == "imagenet":
            try:
                base_classifier.load_state_dict(checkpoint['state_dict'])
            except Exception as e:
                print('direct load failed, try alternative')
                try:
                    base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
                    base_classifier.load_state_dict(checkpoint['state_dict'])
                    # fix
                    # normalize_layer = get_normalize_layer('imagenet').cuda()
                    # base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
                except Exception as e:
                    print('alternative failed again, try alternative 2')
                    base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
                    # base_classifier.load_state_dict(checkpoint['state_dict'])
                    normalize_layer = get_normalize_layer('imagenet').cuda()
                    base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
                    base_classifier.load_state_dict(checkpoint['state_dict'])
        else:
            print("#######################################")
            state_dict = DataParallel2CPU(checkpoint['state_dict'])
            base_classifier.load_state_dict(state_dict)
            # base_classifier.load_state_dict(checkpoint['state_dict'])
        if args.denoiser != '':
            checkpoint_denoiser = torch.load(args.denoiser)
            if "off-the-shelf-denoiser" in args.denoiser:
                denoiser = get_architecture_denoise('orig_dncnn', args.dataset)
                denoiser.load_state_dict(checkpoint_denoiser)
            else:
                denoiser = get_architecture_denoise(checkpoint_denoiser['arch'], args.dataset)
                denoiser.load_state_dict(checkpoint_denoiser['state_dict'])
            base_classifier = torch.nn.Sequential(denoiser, base_classifier)
            print("denoiser added")

    dataset = get_dataset(args.dataset, "proj_test", args.transtype)
    for i in range(len(dataset)):
        # print(len(dataset))

        '''the following is only necessary if run onnx_runtime'''
        if i == 1:
            class ProjectionOp(torch.autograd.Function):
                @staticmethod
                def symbolic(g, x, const):
                    """ In this function, define the arguments and attributes of the operator.
                    "custom::PlusConstant" is the name of the new operator, "x" is an argument
                    of the operator, "const_i" is an attribute which stands for "c" in the operator.
                    There can be multiple arguments and attributes. For attribute naming,
                    use a suffix such as "_i" to specify the data type, where "_i" stands for
                    integer, "_t" stands for tensor, "_f" stands for float, etc. """
                    return g.op('ai.onnx.contrib::ProjectionOp', x, const_i=const)

                @staticmethod
                def forward(ctx, x, const):
                    """ In this function, implement the computation for the operator, i.e.,
                    f(x) = x + c in this case. """
                    # x = x[0]
                    # index = 7000 * ((x + args.partial) / (2 * args.partial))
                    # # now_img = img_test_list[const][index.floor().long().cpu()].cuda().squeeze().squeeze()
                    # this_img_list = img_test_list[const]
                    # now_img = this_img_list[index.floor().long().cpu()]
                    # print("%%%%%%%%%%%%%%%%%%%%", now_img.shape)

                    x = x[0]
                    extrinsic_matrix_origin = extrinsic_tensor[const]
                    alpha = torch.norm(x, float('inf')).unsqueeze(0)
                    complete_3D_oracle = torch.tensor(pc_tensor[const])
                    intrinsic_m = intrinsic_tensor[const]
                    extrinsic_m = find_new_extrinsic_matrix(extrinsic_matrix_origin, args.transtype[-2:], alpha)

                    project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
                        complete_3D_oracle, extrinsic_m, intrinsic_m, k_ambiguity=-1)  # k_ambiguity)
                    now_img = find_2d_image(project_positions_flat, project_positions, points_start, colors,
                                            intrinsic_m, need_second_img=False)

                    now_img = torch.transpose(now_img, 1, 2)
                    now_img = torch.transpose(now_img, 0, 1)  # .type(torch.cuda.FloatTensor)
                    now_img = now_img.unsqueeze(dim=0).float()
                    return now_img

        if i < args.start:
            continue

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i >= args.max >= 0:
            break

        (x, label) = dataset[i]
        # base_classifier = Models['resnet18']()
        model = nn.Sequential(
            Projection(const=i),
            # ProjectionOpt(const=i),
            # Flatten(),
            # nn.Linear(3*90*160, 256),
            # nn.Linear(256, 20),
            base_classifier,
        )
        # model = Models['cnn_6layer'](in_ch=3, in_dim=(32,56))
        # print(model)

        # clean, cert, good = (cAHat == label), True, True
        gap = -1.0

        camera_motion_base = torch.zeros([1, 1])
        # camera_motion_base = torch.zeros([1, 3, 32, 56])


        # wl = wl_list[i].unsqueeze(0)
        # bl = bl_list[i].unsqueeze(0)
        # wu = wu_list[i].unsqueeze(0)
        # bu = bu_list[i].unsqueeze(0)
        # h_L = -args.partial
        # h_U = args.partial
        # lower = h_L * wl + bl
        # upper = h_U * wu + bu
        # # diff_l = now_img - lower.reshape(now_img.shape)
        # # diff_u = -now_img + upper.reshape(now_img.shape)
        #
        # # camera_motion_base = lower.reshape([1, 3, 90, 160])
        # # camera_motion_base = torch.nn.functional.interpolate(camera_motion_base, (32,56), mode='bicubic')
        # # print("$$$$",x.shape, camera_motion_base.shape)
        # camera_motion_base = x.unsqueeze(0)
        # camera_motion_base = torch.ones([1, 3, 32, 32])
        camera_motion_base = camera_motion_base.cpu()
        model = model.cpu()
        model.eval()
        # model(torch.zeros_like(camera_motion_base))
        # Input to the model
        # x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
        torch_out = model(camera_motion_base)
        print(
            f"finish {i}, GT {label}, predict {torch.argmax(torch_out[0])}")
        if not label == torch.argmax(torch_out[0]):
            print("wrong...", i)
            continue
        # Export the model
        torch.onnx.export(model,  # model being run
                          camera_motion_base,  # model input (or a tuple for multiple inputs)
                          f"{args.outfile}_{i}_{label}.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=14,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}},
                          custom_opsets={"ai.onnx.contrib": 14}
                          # operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK
                          )
        import onnx

        onnx_model = onnx.load(f"{args.outfile}_{i}_{label}.onnx")
        onnx.checker.check_model(onnx_model)

        if i == 0:
            import onnxruntime

            import onnxruntime as _ort
            from onnxruntime_extensions import (
                onnx_op, PyCustomOpDef,
                get_library_path as _get_library_path)


            # from onnxruntime.tools import pytorch_export_contrib_ops
            # pytorch_export_contrib_ops.register()
            const = i
            @onnx_op(op_type='ProjectionOp', domain='ai.onnx.contrib',
                     # inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float], outputs=[PyCustomOpDef.dt_float]
                     )
            def ProjectionOp(x):
                x = torch.from_numpy(x)
                x = x[0]
                extrinsic_matrix_origin = extrinsic_tensor[const]
                alpha = torch.norm(x, float('inf')).unsqueeze(0)
                complete_3D_oracle = torch.tensor(pc_tensor[const])
                intrinsic_m = intrinsic_tensor[const]
                extrinsic_m = find_new_extrinsic_matrix(extrinsic_matrix_origin, args.transtype[-2:], alpha)

                project_positions_flat, project_positions_float, project_positions, points_start, colors = projection_oracle(
                    complete_3D_oracle, extrinsic_m, intrinsic_m, k_ambiguity=-1)  # k_ambiguity)
                now_img = find_2d_image(project_positions_flat, project_positions, points_start, colors,
                                        intrinsic_m, need_second_img=False)

                now_img = torch.transpose(now_img, 1, 2)
                now_img = torch.transpose(now_img, 0, 1)  # .type(torch.cuda.FloatTensor)
                now_img = now_img.unsqueeze(dim=0).float()
                return now_img


            so = _ort.SessionOptions()
            so.register_custom_ops_library(_get_library_path())

            ort_session = _ort.InferenceSession(f"{args.outfile}_{i}_{label}.onnx", so)

            # ort_session = onnxruntime.InferenceSession(f"{args.outfile}_{i}.onnx")



            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


            # compute ONNX Runtime output prediction
            # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
            ort_inputs = {"input": np.zeros((1,1)).astype(np.float32)}
            ort_outs = ort_session.run(None, ort_inputs)

            # compare ONNX Runtime and PyTorch results
            np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
            print(f"finish {i}, GT {label}, predict {torch.argmax(torch_out[0])}, {torch.argmax(torch.tensor(ort_outs[0]))}")

            print("Exported model has been tested with ONNXRuntime, and the result looks good!")


