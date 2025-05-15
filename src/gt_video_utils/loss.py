import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from pytorch3d.loss import chamfer_distance
import os
import trimesh as tm
import numpy as np
from tqdm.autonotebook import tqdm
import random
import scipy


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()



def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def load_gt_pcds(path):
    gts = []
    indices = [ply.split('.')[0] for ply in os.listdir(path)]
    print(indices)
    indices.sort()
    n_digits = len(indices[0])
    indices = [int(idx) for idx in indices if idx.strip() != '']
    indices.sort()
    for idx in range(len(indices)):
        print(os.path.join(path, f"{idx:0{n_digits}d}.ply"))
        if not os.path.exists(os.path.join(path, f"{idx:0{n_digits}d}.ply")):
            print("跳过，文件不存在")
            continue
        pcd = tm.load(os.path.join(path, f"{idx:0{n_digits}d}.ply"), process=False)
        print(pcd.vertices)
        np_pcd = np.array(pcd.vertices)
        gts.append(torch.from_numpy(np_pcd).to('cuda',dtype=torch.float32).contiguous())
    return gts



def evaluate(preds, gts, loss_type='CD'):
    # print(f"Prediction sequence {len(preds)}, gts sequence {len(gts)}")
    # if len(preds) != len(gts):
    #     print("[Error]: The prediction sequence is not align with the gt sequence.")
    #     return
    # print(f"Prediction pcd particles cnt {preds[0].shape[0]}, gt pcd particles cnt {gts[0].shape[0]}")
    max_f = len(preds)
    fit_loss = 0.0
    predict_loss = 0.0
    for f in tqdm(range(max_f), desc=f"Evaluate {loss_type} Loss"):
        # pcd1 = discretize(preds[f], 0.02)
        # pcd2 = discretize(gts[f], 0.02)
        # cd = chamfer_distance(preds[f], gts[f])
        # print(f"frames: {f}, cd: {cd}")

        # cd align with https://zlicheng.com/spring_gaus/
        n_sample = 2048 if loss_type == 'EMD' else 8192
        pcd0 = preds[f]
        pcd1 = gts[f]
        
        n_sample = min(n_sample, pcd0.shape[0], pcd1.shape[0])

        pcd0 = pcd0[random.sample(range(pcd0.shape[0]), n_sample), :]
        pcd1 = pcd1[random.sample(range(pcd1.shape[0]), n_sample), :]
        
        print("pcd0",pcd0.shape)
        print("pcd1",pcd1.shape)

        if loss_type == "CD":
            loss = (chamfer_distance(pcd0[None], pcd1[None])[0] * 1e3).item()
            # loss = (chamfer_distance(pcd0[None], pcd1[None])[0]).item()
        elif loss_type == "EMD":
            loss = emd_func(pcd0, pcd1).item()
        else:
            print("[Error]: undefined error type.")
        
        fit_loss+= loss

    fit_loss /= max_f
    print(f"{loss_type} loss train: {fit_loss}")
    return fit_loss


def emd_func(x, y, pkg="torch"):
    if pkg == "numpy":
        # numpy implementation
        x_ = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y_ = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        cost_matrix = np.linalg.norm(x_ - y_, 2, axis=2)
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(
                cost_matrix, maximize=False
            )
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = np.mean(np.linalg.norm(x[ind1] - y[ind2], 2, axis=1))
    else:
        # torch implementation
        x_ = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y_ = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x_, -y_), 2, dim=2)  # dis: [N, M]
        cost_matrix = dis.detach().cpu().numpy()
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(
                cost_matrix, maximize=False
            )
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = torch.mean(torch.norm(torch.add(x[ind1], -y[ind2]), 2, dim=1))

    return emd