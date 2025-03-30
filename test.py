from src.gt_video_utils.gt_cam_loader import readCamerasFromAllData, group_cameras_by_time



camera_infos = readCamerasFromAllData('/home/qingran/Desktop/gic/data/pacnerf/bird', True)

for camera_info in camera_infos:
    print(camera_info.fid)
    
    
fid_to_cams, sorted_fids = group_cameras_by_time(camera_infos)

print(sorted_fids)