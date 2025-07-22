import random
import time
import os
import shutil
import argparse
import math

import numpy as np
import warp as wp
import taichi as ti
import torch
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.mpm_core import MPMModel, set_boundary_conditions
from src.physics_guided_network import PhysicsNetwork
from src.video_distillation.ms_guidance import ModelscopeGuidance
from src.video_distillation.prompt_processors import ModelscopePromptProcessor
from src.utils.render_utils import *
from src.utils.misc_utils import *
from src.utils.camera_view_utils import load_camera_params

from src.gt_video_utils.gt_cam_loader import readCamerasFromAllData,  group_cameras_by_time, split_test_train_cams
from src.gt_video_utils.loss import *

import copy

def init_training(cfg, args=None):
    
    # get export folder
    export_path = cfg.train.export_path if cfg.train.export_path else './outputs'
    if cfg.train.train_tag is None:
        cfg.train.train_tag = time.strftime("%Y%m%d_%H_%M_%S")
    export_path = os.path.join(export_path, cfg.train.train_tag)
    if os.path.exists(export_path):
        # if args is not None and not args.overwrite:
        #     overwrite = input(f'Warning: export path {export_path} already exists. Exit?(y/n)')
        #     if overwrite.lower() == 'y':
        #         exit()
        print(f'Warning: export path {export_path} already exists')
    else:
        os.makedirs(export_path)
        os.makedirs(os.path.join(export_path, 'images'))
        os.makedirs(os.path.join(export_path, 'videos'))
        os.makedirs(os.path.join(export_path, 'checkpoints'))
    
   
    
    
    # set seed
    seed = cfg.train.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # init warp
    device = f'cuda:{cfg.train.gpu}'
    wp.init()
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})
    
    # init taichi
    if cfg.preprocessing.particle_filling is not None:
        ti.init(arch=ti.cuda, device_memory_GB=8.0)
    
    # init torch
    torch_device = torch.device(device)
    torch.cuda.set_device(cfg.train.gpu)
    torch.backends.cudnn.benchmark = False
    print(f'\n using device: {device}\n')
    
    # export config
    print(f'exporting to: {export_path}\n')
    with open(os.path.join(export_path, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    
    # init writer
    writer_path = os.path.join(export_path, 'writers', 'writer_' + time.strftime("%Y%m%d_%H_%M_%S"))
    os.makedirs(writer_path)
    writer = SummaryWriter(writer_path)
    
    return torch_device, export_path, writer
    
def main(cfg, args=None):
    
    train_params = cfg.train
    preprocessing_params = cfg.preprocessing
    render_params = cfg.render
    material_params = cfg.material
    model_params = cfg.model
    sim_params = cfg.sim
    guidance_params = cfg.guidance
    prompt_params = cfg.prompt_processor
    prompt_params.prompt = train_params.prompt

    device_material = torch.device("cuda:0")
    device_mpm = torch.device("cuda:1")

    
    
    #set camera infos of training dataset
    
    camera_infos = readCamerasFromAllData(args.gt_video_folder,False)
    
    
    test_camera_id = 10
    
    test_cams, train_cams = split_test_train_cams(camera_infos,test_camera_id)
    
    # fid_to_cams 
    
    # sorted_fids
    
    if(train_params.enable_train):
        
        fid_to_cams, sorted_fids = group_cameras_by_time(camera_infos)
        
    else:
        
        fid_to_cams, sorted_fids = group_cameras_by_time(test_cams)
         
    

    # init training
    torch_device, export_path, writer = init_training(cfg, args)
    
    
    # init gaussians
    print("Initializing gaussian scene and pre-processing...")
    model_path = train_params.model_path
    gaussians = load_gaussian_ckpt(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=torch_device)
        if render_params.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=torch_device)
    )
    (
        mpm_params,
        init_e_cat, init_p_cat,
        unselected_params,
        translate_params,
        screen_points
    ) = load_params(
        gaussians, pipeline, 
        preprocessing_params, material_params, model_params,
        export_path=export_path
    )
    
    # get preprocessed gaussian params
    trans_pos = mpm_params['pos']
    print("trans_pos:",trans_pos)
    trans_cov = mpm_params['cov']
    trans_opacity = mpm_params['opacity']
    trans_shs = mpm_params['shs']
    trans_features = mpm_params['features']
    
    # get translation params
    rotation_matrices = translate_params['rotation_matrices']
    scale_origin = translate_params['scale_origin']
    original_mean_pos = translate_params['original_mean_pos']
    
    gs_num = trans_pos.shape[0]
    
    print(f'Built gaussian particle number: {gs_num}\n')
 
    # camera setting
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = load_camera_params(render_params, rotation_matrices, scale_origin, original_mean_pos)

    # export static gaussian rendering
    print(f'Exporting static gaussian rendering to {export_path}/static.png\n')
    export_static_gaussian_rendering(
        trans_pos, trans_cov, trans_shs, trans_opacity,
        unselected_params, rotation_matrices, scale_origin, original_mean_pos,
        model_path, pipeline, render_params, viewpoint_center_worldspace, observant_coordinates, gaussians, background, screen_points,
        export_path
    )

    # init mpm model and material
    print('Building MPM simulator and setting boundary conditions\n')    
    mpm_model = MPMModel(
        sim_params, 
        material_params, 
        init_pos=trans_pos, 
        enable_train=train_params.enable_train, 
        device=0
    )
    
    set_boundary_conditions(mpm_model, sim_params.boundary_conditions)
    
    mpm_model_copy = MPMModel(
        sim_params, 
        material_params, 
        init_pos=trans_pos, 
        enable_train=train_params.enable_train, 
        device=1
    )
    
    set_boundary_conditions(mpm_model_copy, sim_params.boundary_conditions)
    
    
    
    material = PhysicsNetwork(
        elasticity_physicals=material_params.elasticity_physicals,
        plasticity_physicals=material_params.plasticity_physicals,
        in_channels=trans_features.shape[1],
        params=model_params,
        n_particles=gs_num,
        export_path=export_path
    ).to(torch_device).requires_grad_(train_params.enable_train)


    
    # material
    print('Loading neural constitutive model\n')
    elasticity, plasticity = init_constitute(
        material_params.elasticity,
        material_params.plasticity,
        elasticity_physicals=material_params.elasticity_physicals,
        plasticity_physicals=material_params.plasticity_physicals,
        requires_grad=train_params.enable_train,
        device=torch_device
    )
    save_model_dict_arch(
        {
            'material': material,
            'elasticity': elasticity, 
            'plasticity': plasticity, 
        }, 
        export_path
    )
    
    start_epoch = 0
    epochs = train_params.epochs
    num_skip_frames = sim_params.num_skip_frames
    num_frames = sim_params.num_frames
    frames_per_stage = sim_params.frames_per_stage
    assert (num_frames - num_skip_frames) % frames_per_stage == 0
    num_stages = (num_frames - num_skip_frames) // frames_per_stage
    steps_per_frame = sim_params.steps_per_frame
    
    # material_opt = torch.optim.Adam(material.parameters(), lr=train_params.learning_rate)
    if train_params.enable_train:
        material_opt = torch.optim.Adam(material.parameters(), lr=train_params.learning_rate)
        start_epoch = load_material_checkpoint(
            material,
            material_optimizer=material_opt,
            ckpt_dir=os.path.join(export_path, 'checkpoints'),
            epoch=train_params.ckpt_epoch,
            device=torch_device
        )
        print(f'\nStart training with\n{elasticity.name()}\n{plasticity.name()}')
        print(f'The prompt is: {train_params.prompt}\n')
    else:
        epochs = 1
        internal_epochs = 1
        load_material_checkpoint(
            material,
            ckpt_dir=os.path.join(export_path, 'checkpoints'),
            # epoch=train_params.ckpt_epoch,
            epoch = 29,
            device=torch_device,
        )
        print('\nTraining is disabled.')
        print(f'Setting epochs to 1 and start rendering with\n{elasticity.name()}\n{plasticity.name()}\n')

    # init params
    requires_grad = train_params.enable_train
    x = trans_pos.detach()
    v = torch.stack([torch.tensor([0.0, 0.0, 0.0], device=torch_device) for _ in range(gs_num)]) # a default vertical velocity is set
    C = torch.zeros((gs_num, 3, 3), device=torch_device)
    F = torch.eye(3, device=torch_device).unsqueeze(0).repeat(gs_num, 1, 1)
    
    x = x.requires_grad_(False)
    v = v.requires_grad_(False)
    C = C.requires_grad_(False)
    F = F.requires_grad_(False)

    
    # skip first few frames to accelerate training
    # this frames are meaningless when there is no contact or collision
    with torch.no_grad():
        mpm_model.reset()
        if material_params.elasticity == 'neural' or material_params.plasticity == 'neural':
            e_cat, p_cat = material(trans_pos, trans_features)
            e_cat = e_cat.detach().requires_grad_(False)
            p_cat = p_cat.detach().requires_grad_(False)
        else:
            e_cat, p_cat = init_e_cat, init_p_cat
            
        pred_pcds = []
        
        epoch = -1
        
        save_gt_image = True
    
        for idx in tqdm(range(num_skip_frames), desc='Skip Frames'):
                
            
            fid = sorted_fids[idx]
            
            next_fid = sorted_fids[idx + 1]
           
            
            cams = fid_to_cams[fid]
            
            
            
            cam = random.choice(cams)
            # for cam in cams_selected:
            # cam =  cams[7]
            # print(cam.image)
            
            # render
            frame_id = cam.uid
            print(cam.image_name)
            
            fid = sorted_fids[idx]
            
            cur_cam = loadCam(cam,1.0)
            
            # get rendering params
            (
                render_pos,
                render_cov,
                render_shs,
                render_opacity,
                render_rot
            ) = get_mpm_gaussian_params(
                pos=x, cov=trans_cov, shs=trans_shs, opacity=trans_opacity,
                F=F,
                unselected_params=unselected_params,
                rotation_matrices=rotation_matrices,
                scale_origin=scale_origin,
                original_mean_pos=original_mean_pos
            )
            
            rendering = render_mpm_gaussian_with_gt_camera_info(
                pipeline=pipeline,
                gaussians=gaussians,background=background, 
                pos=render_pos, cov=render_cov,shs=render_shs, opacity=render_opacity, rot=render_rot,
                screen_points=screen_points,
                camera_info = cam,
                logits=None
            )
            # if not train_params.enable_train:
            os.makedirs(export_path + '/plys', exist_ok=True)
            particle_position_tensor_to_ply(render_pos,export_path +'/plys/'+ str(idx) + '.ply')
            
            pred_pcds.append(render_pos)
            
            # if(epoch == 9):
            gt_image = cur_cam.original_image   
            
            if train_params.enable_train:
                
                os.makedirs(export_path + '/images/' + str(epoch), exist_ok=True)
    
                
                export_rendering_abs_path(rendering,export_path +'/images/'+ str(epoch) + '/' + cam.image_name +'.png')
                
                
                # if(epoch == 9):
                if(save_gt_image):
                    export_rendering_abs_path(gt_image,export_path + '/images/' + str(epoch) + '/' + 'gt_' + cam.image_name +'.png')
                
            
            else:
                os.makedirs(export_path + '/images/eval_' + str(cam.uid), exist_ok=True)
                os.makedirs(export_path + '/images/eval_gt_' + str(cam.uid), exist_ok=True)
                export_rendering_abs_path(rendering,export_path +'/images/eval_' + str(cam.uid) + '/' + cam.image_name +'.png')
                export_rendering_abs_path(gt_image,export_path + '/images/eval_gt_' + str(cam.uid) + '/' + cam.image_name +'.png') 
                
                # evaluate([render_pos], [gt_pcds[idx]], 'CD')
                # evaluate([render_pos], [gt_pcds[idx]], 'EMD')
                
                
            # mpm step
            for step in tqdm(range(round((next_fid-fid)/sim_params.dt)), leave=False):
                
                # print("next_fid-fid:",next_fid-fid)
                
                # mpm step, using checkpoint to save memory
                stress = checkpoint(elasticity, F, e_cat)
                
                assert torch.all(torch.isfinite(stress))
                
                
                # if(idx < 0):
                
                x, v, C, F = checkpoint(mpm_model, x, v, C, F, stress)

                
                assert torch.all(torch.isfinite(x))
                assert torch.all(torch.isfinite(F))
                
                F = checkpoint(plasticity, F, p_cat)
                
                assert torch.all(torch.isfinite(F))    
        
                # if (idx+1) % 1 == 0 or idx == len(sorted_fids) - 1:    
                
            
    
    x_skip = x.detach()
    v_skip = v.detach()
    C_skip = C.detach()
    F_skip = F.detach()
    time_skip = mpm_model.time
    
    
    print(time_skip)
    
    # torch.autograd.set_detect_anomaly(True)


    
    # gt_pcds = load_gt_pcds(args.gt_ply_folder)
   
    
    for epoch in range(start_epoch, epochs):
    # for epoch in range(0, 100):
        
       
        
        # recover ckpt status to the skip stage
        
            
        
        time_ckpt = time_skip
        print("time_ckpt",time_ckpt)
        
               
        
        x = x_skip.requires_grad_(requires_grad)
        v = v_skip.requires_grad_(requires_grad)
        C = C_skip.requires_grad_(requires_grad)
        F = F_skip.requires_grad_(requires_grad)
        
        # init optimizer
        if train_params.enable_train:
            
            material_opt.zero_grad()
            
            loss = torch.tensor(0.0, device='cuda')
        
        cam_count = 0
                
        # rand_num = np.random.randint(0, 11)
        
        
        save_gt_image = True
                
                
        psnr_list = []
        ssim_list = []
        
        
        
        # for pcd in gt_pcds:
        #     print(pcd.shape)
        
        pred_pcds = []
        
        trans_pos = trans_pos.detach().requires_grad_(requires_grad)
        trans_cov = trans_cov.detach().requires_grad_(requires_grad)
        trans_shs = trans_shs.detach().requires_grad_(requires_grad)
        trans_opacity = trans_opacity.detach().requires_grad_(requires_grad)
        trans_features = trans_features.detach().requires_grad_(requires_grad)
            
        
        for idx in range(num_skip_frames,len(sorted_fids)):
            
            
            # if idx % 1 == 0:
            print("idx:",idx)
            
            
            fid = sorted_fids[idx]
            if idx == len(sorted_fids) - 1:
                next_fid = fid
            else:
                next_fid = sorted_fids[idx + 1]
            
            
            cams = fid_to_cams[fid]
            
            
            for i in unselected_params:
                if unselected_params[i] is not None:
                    unselected_params[i] = unselected_params[i].detach()
            scale_origin = scale_origin.detach()
            original_mean_pos = original_mean_pos.detach()
            screen_points = screen_points.detach()
            assert x.requires_grad == requires_grad
            
            if material_params.elasticity == 'neural' or material_params.plasticity == 'neural':
                # print("this is neural")
                # extract feature
                e_cat, p_cat = material(trans_pos, trans_features)
            else:
                e_cat, p_cat = init_e_cat, init_p_cat
            
             
            # cams_selected = [cam for cam in cams 
            # if int(cam.image_name.split('_')[1]) == rand_num]
            
            cam = random.choice(cams)
            
            # render
            frame_id = cam.uid
            print(cam.image_name)
            
            cur_cam = loadCam(cam,1.0)
            
            # get rendering params
            (
                render_pos,
                render_cov,
                render_shs,
                render_opacity,
                render_rot
            ) = get_mpm_gaussian_params(
                pos=x, cov=trans_cov, shs=trans_shs, opacity=trans_opacity,
                F=F,
                unselected_params=unselected_params,
                rotation_matrices=rotation_matrices,
                scale_origin=scale_origin,
                original_mean_pos=original_mean_pos
            )
            
            rendering = render_mpm_gaussian_with_gt_camera_info(
                pipeline=pipeline,
                gaussians=gaussians,background=background, 
                pos=render_pos, cov=render_cov,shs=render_shs, opacity=render_opacity, rot=render_rot,
                screen_points=screen_points,
                camera_info = cam,
                logits=None
            )
            # if not train_params.enable_train:
            os.makedirs(export_path + '/plys', exist_ok=True)
            particle_position_tensor_to_ply(render_pos,export_path +'/plys/'+ str(idx) + '.ply')
            
            pred_pcds.append(render_pos)
            
            # if(epoch == 9):
            gt_image = cur_cam.original_image   
            
            if train_params.enable_train:
                
                os.makedirs(export_path + '/images/' + str(epoch), exist_ok=True)
    
                
                export_rendering_abs_path(rendering,export_path +'/images/'+ str(epoch) + '/' + cam.image_name +'.png')
                
                lambda_dssim = 0.6
                
                # Loss
                
                
                # if(epoch == 9):
                if(save_gt_image):
                    export_rendering_abs_path(gt_image,export_path + '/images/' + str(epoch) + '/' + 'gt_' + cam.image_name +'.png')
                
                    
                cam_count = cam_count + 1
            
            
                Ll1 = l1_loss(rendering, gt_image)
                loss = loss + (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(rendering, gt_image)) 
                    
            
            else:
                os.makedirs(export_path + '/images/eval_' + str(cam.uid), exist_ok=True)
                os.makedirs(export_path + '/images/eval_gt_' + str(cam.uid), exist_ok=True)
                export_rendering_abs_path(rendering,export_path +'/images/eval_' + str(cam.uid) + '/' + cam.image_name +'.png')
                export_rendering_abs_path(gt_image,export_path + '/images/eval_gt_' + str(cam.uid) + '/' + cam.image_name +'.png') 
                psnr_list.append(psnr(rendering, gt_image))
                ssim_list.append(ssim(rendering, gt_image))
                
                # evaluate([render_pos], [gt_pcds[idx]], 'CD')
                # evaluate([render_pos], [gt_pcds[idx]], 'EMD')
                
            
            # mpm step
            for step in tqdm(range(round((next_fid-fid)/sim_params.dt)), leave=False):
                
                # print("next_fid-fid:",next_fid-fid)
                
                # mpm step, using checkpoint to save memory
                stress = checkpoint(elasticity, F, e_cat)
                
                assert torch.all(torch.isfinite(stress))
                
                
                # if(idx < 0):
                
                # x, v, C, F = checkpoint(mpm_model, x, v, C, F, stress)

                # else:
                x_mpm = x.to(device_mpm)
                v_mpm = v.to(device_mpm)
                C_mpm = C.to(device_mpm)
                F_mpm = F.to(device_mpm)
                stress_mpm = stress.to(device_mpm)

            
                x_mpm, v_mpm, C_mpm, F_mpm = checkpoint(mpm_model_copy, x_mpm, v_mpm, C_mpm, F_mpm, stress_mpm)

                x = x_mpm.to(device_material)
                v = v_mpm.to(device_material)
                C = C_mpm.to(device_material)
                F = F_mpm.to(device_material)

                
                assert torch.all(torch.isfinite(x))
                assert torch.all(torch.isfinite(F))
                
                F = checkpoint(plasticity, F, p_cat)
                
                assert torch.all(torch.isfinite(F))    
        
                # if (idx+1) % 1 == 0 or idx == len(sorted_fids) - 1:
                    
            
        if train_params.enable_train:
        
            loss = loss/cam_count
            print("loss:",loss)
            print("cam_count",cam_count)
            loss.backward()
            
            # convert non-finite gradients to zero
            
            for p in material.parameters():
                # print(material.parameters())
                if p is not None and p.grad is not None:
                    torch.nan_to_num_(p.grad, 0.0, 0.0, 0.0)
            torch.nn.utils.clip_grad_norm_(material.parameters(), 1.0)
            material_opt.step()
            
            with open(os.path.join(export_path, 'log.txt'), 'a', encoding='utf-8') as f:
                lr = material_opt.param_groups[0]['lr']
                # f.write(f'epoch: {epoch}, idx: {idx}, loss: {loss}, lr: {lr}, e_cat: {e_cat.mean(dim=0).tolist()}, p_cat: {p_cat.mean(dim=0).tolist()}\n')
                f.write(f'epoch: {epoch}, loss: {loss}, lr: {lr}, e_cat: {e_cat.mean(dim=0).tolist()}, p_cat: {p_cat.mean(dim=0).tolist()}\n')

            # save_video_by_last_idx(f'{export_path}/images/'+ str(epoch),f'{export_path}/videos/video_{epoch:04d}.mp4')
            
            # if(save_gt_image):
            #     save_video_by_last_idx(f'{export_path}' + '/images/' + 'gt_' + str(rand_num),f'{export_path}'+'/videos/gt_video_'+str(rand_num)+'.mp4')
            
            # x_ckpt = x.detach()
            # v_ckpt = v.detach()
            # C_ckpt = C.detach()
            # F_ckpt = F.detach()
            # time_ckpt = mpm_model.time
            
            if (epoch+1) % 10 == 0 or epoch == 0:
                save_material_checkpoint(
                    material,
                    material_opt,
                    ckpt_dir=os.path.join(export_path, 'checkpoints'),
                    epoch=epoch
                )
        else:
            
            save_video_by_last_idx(f'{export_path}'+'/images/eval_' + str(test_camera_id),f'{export_path}'+'/videos/video_'+str(test_camera_id)+'.mp4')
            
        
            save_video_by_last_idx(f'{export_path}'+'/images/eval_gt_'+ str(test_camera_id),f'{export_path}'+'/videos/gt_video_'+str(test_camera_id)+'.mp4')
            
            mean_psnr = torch.mean(torch.stack(psnr_list))
            mean_ssim = torch.mean(torch.stack(ssim_list)) 
            print(f'average psnr: {mean_psnr}')
            print(f'average ssim: {mean_ssim}')
            
            # cd = evaluate(pred_pcds, gt_pcds, 'CD')
            # emd = evaluate(pred_pcds, gt_pcds, 'EMD')
            
            
            test_log = {
                "psnr": mean_psnr.item(),
                "ssim": mean_ssim.item(),
                # "cd": cd,
                # "emd": emd
            }

            # 写入 test_log.json
            with open(export_path + "/test_log.json", "w") as f:
                json.dump(test_log, f, indent=4)

            print("Test log saved to test_log.json")
            
                
            

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the config file."
    )
    
    parser.add_argument(
        "--guidance_config", 
        type=str, 
        default='./configs/ms_guidance.yaml', 
        help="Path to the SDS guidance config file."
    )
    
    parser.add_argument(
        "--gt_video_folder", 
        type=str, 
        default='/home/qingran/Desktop/omniphysgs/data/bird', 
        help="Path to the ground truth video data folder."
    )
    
    parser.add_argument(
        "--gt_ply_folder", 
        type=str, 
        default='/workspace/bird', 
        help="Path to the ground truth ply folder."
    )
    
    

    parser.add_argument(
        "--test", 
        action='store_true', 
        help="Test mode."
    )

    parser.add_argument(
        "--gpu", 
        type=int, 
        help="GPU index."
    )

    parser.add_argument(
        "--tag", 
        type=str,
        help="Training tag."
    )

    parser.add_argument(
        "--overwrite", 
        '-o', 
        action='store_true', 
        help="Overwrite the existing export folder."
    )

    parser.add_argument(
        "--output", 
        type=str, 
        help="Output folder."
    )

    parser.add_argument(
        "--save_internal", 
        action='store_true', 
        help="Save internal checkpoints."
    )

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    guidance_cfg = OmegaConf.load(args.guidance_config)
    cfg = OmegaConf.merge(cfg, guidance_cfg)

    if args.gpu is not None:
        cfg.train.gpu = args.gpu
    if args.test:
        cfg.train.enable_train = False
    if args.tag is not None:
        cfg.train.train_tag = args.tag
    if args.output is not None:
        cfg.train.export_path = args.output

    main(cfg, args)
