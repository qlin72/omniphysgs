import os
import json

def collect_metrics(root_dir):
    psnr_list, ssim_list, cd_list, emd_list = [], [], [], []

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        json_path = os.path.join(subdir_path, 'test_log.json')
        
        if os.path.isdir(subdir_path) and os.path.isfile(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    psnr_list.append(data['psnr'])
                    ssim_list.append(data['ssim'])
                    cd_list.append(data['cd'])
                    emd_list.append(data['emd'])
            except Exception as e:
                print(f"Failed to read {json_path}: {e}")

    def average(lst):
        # print(sum(lst))
        # print(len(lst))
        return sum(lst) / len(lst) if lst else float('nan')

    print("Average Results Across Subfolders:")
    print(f"PSNR: {average(psnr_list):.4f}")
    print(f"SSIM: {average(ssim_list):.4f}")
    print(f"CD:   {average(cd_list):.4f}")
    print(f"EMD:  {average(emd_list):.4f}")

# 修改为你的根目录路径
root_folder = "/workspace/omniphysgs/outputs/elastic"
collect_metrics(root_folder)