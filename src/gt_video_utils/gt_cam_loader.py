import numpy as np
from typing import NamedTuple, Optional
import os
import math
import json
from PIL import Image
from pathlib import Path
import tqdm


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
            fovy = focal2fov(intrinsic[1][1], ori_h)
            fovx = focal2fov(intrinsic[0][0], ori_w)
            FovY = fovx
            FovX = fovy
            if ori_w != 800:
                image = image.resize((800, 800), Image.BILINEAR)
                if white_background:
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


