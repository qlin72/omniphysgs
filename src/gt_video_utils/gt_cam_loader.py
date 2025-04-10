import numpy as np
from typing import NamedTuple, Optional
import os
import math
import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
import sys
sys.path.append("third_party/gaussian-splatting")
from scene.cameras import Camera


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    alpha: Optional[np.array] = None
    
def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def readCamerasFromAllData(path, white_background):
    cam_infos = []

    with open(os.path.join(path, "all_data.json")) as json_file:
        frames = json.load(json_file)

        for idx, frame in enumerate(tqdm(frames)):
            cam_id, frame_id = frame["file_path"].split("/")[-1].rstrip(".png").lstrip("r_").split("_")
            if frame_id == '-1':
                continue
            file_path = frame["file_path"].replace('r_', 'm_')
            image_path = os.path.join(path, file_path)
            image_name = Path(image_path).stem
            image = np.asarray(Image.open(image_path))
            
            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = image / 255.0
            mask = (image.astype(int).sum(-1, keepdims=True) != 255 * 3).astype(float)
            arr = norm_data * mask + bg * (1 - mask)
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")
            
            frame_time = frame['time']
            
            c2w = frame["c2w"]
            c2w.append([0.0, 0.0, 0.0, 1.0])
            matrix = np.linalg.inv(np.array(c2w))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            intrinsic = frame['intrinsic']
            ori_h, ori_w = image.size[0], image.size[1]
            # print("ori_h:", ori_h,"ori_w:", ori_w)
            fovy = focal2fov(intrinsic[1][1], ori_h)
            fovx = focal2fov(intrinsic[0][0], ori_w)
            FovY = fovx
            FovX = fovy
            if ori_w != 800:
                image = image.resize((800, 800), Image.BILINEAR)
                if white_background:
                    print("white_background")
                    mask = (np.asarray(image).astype(int).sum(-1, keepdims=True) != 255 * 3).astype(float)
                else:
                    mask = (np.asarray(image).astype(int).sum(-1, keepdims=True) != 0).astype(float)

            cam_infos.append(CameraInfo(uid=int(cam_id), 
                                        R=R, T=T, 
                                        FovY=FovY, FovX=FovX, 
                                        image=image,
                                        image_path=image_path, image_name=image_name, 
                                        width=image.size[0], height=image.size[1], 
                                        fid=frame_time, alpha=mask,
                                        ))

    return cam_infos

def group_cameras_by_time(camera_infos):
    from collections import defaultdict
    fid_to_cams = defaultdict(list)
    for cam in camera_infos:
        fid_to_cams[cam.fid].append(cam)
    sorted_fids = sorted(fid_to_cams.keys())
    return fid_to_cams, sorted_fids


def loadCam(cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    # if args.resolution in [1, 2, 4, 8]:
    #     resolution = round(orig_w / (resolution_scale * args.resolution)), round(
    #         orig_h / (resolution_scale * args.resolution))
    # else:  # should be a type that converts to float
    # if args.resolution == -1:
    if orig_w > 1600:
        global WARNED
        if not WARNED:
            print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                "If this is not desired, please explicitly specify '--resolution/-r' as 1")
            WARNED = True
        global_down = orig_w / 1600
    else:
        global_down = 1
    # else:
    #     global_down = orig_w / args.resolution

    scale = float(global_down) * float(resolution_scale)
    resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    elif cam_info.alpha is not None:
        alpha = Image.fromarray((cam_info.alpha[..., 0] * 255).astype(np.uint8)).resize(resolution)
        alpha = np.asarray(alpha).astype(float) / 255
        loaded_mask = alpha[None]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=cam_info.uid,
                  data_device='cuda')

    
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


