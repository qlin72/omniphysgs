# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from skimage.metrics import structural_similarity
# from PIL import Image
# import numpy as np
# import os
# from pytorch_msssim import ms_ssim

# from src.gt_video_utils.loss import *

# def load_image(path):
#     img = Image.open(path).convert('RGB').resize((256, 256))
#     transform = transforms.ToTensor()
#     return transform(img).unsqueeze(0)  # shape: (1, 3, H, W)

# def compute_l1(img1, img2):
#     return F.l1_loss(img1, img2)

# def compute_ssim(img1, img2):
#     # Convert to numpy for SSIM (use grayscale for simplicity)
#     img1_np = img1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#     img2_np = img2.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#     img1_gray = np.mean(img1_np, axis=2)
#     img2_gray = np.mean(img2_np, axis=2)
#     return structural_similarity(img1_gray, img2_gray, data_range=1.0)

# def compute_total_loss(img1, img2, lambda_ssim=0.5):
#     l1 = compute_l1(img1, img2)
#     ssim_score = compute_ssim(img1, img2)
#     total_loss = l1 + lambda_ssim * (1 - ssim_score)
#     return total_loss.item(), l1.item(), ssim_score

# def compute_multi_scale_loss(img1,img2):
    
#     loss = 1 - ms_ssim(img1, img2, data_range=1.0, size_average=True)
#     print(loss)

# if __name__ == '__main__':
#     # 路径替换成你的图像路径
#     gt = load_image("/home/qingran/Desktop/omniphysgs/outputs/dataset_45_07/images/20/gt_m_8_12.png")
#     pred1 = load_image("/home/qingran/Desktop/omniphysgs/outputs/dataset_45_07/images/20/m_8_12_.png")
#     pred2 = load_image("/home/qingran/Desktop/omniphysgs/outputs/dataset_45_07/images/20/m_8_12.png")

#     loss1, l1_1, ssim_1 = compute_total_loss(pred1, gt)
#     loss2, l1_2, ssim_2 = compute_total_loss(pred2, gt)

#     print("===> Loss: m_10_10 vs GT")
#     print(f"Total: {loss1:.4f}, L1: {l1_1:.4f}, SSIM: {ssim_1:.4f}")

#     print("===> Loss: m_9_10 vs GT")
#     print(f"Total: {loss2:.4f}, L1: {l1_2:.4f}, SSIM: {ssim_2:.4f}")
    
#     compute_multi_scale_loss(pred1,gt)
#     compute_multi_scale_loss(pred2,gt)
    
    

#     if loss1 < loss2:
#         print("\n⚠️ 注意：虽然 loss 更低，但这可能只是颜色更接近。SSIM 能捕捉结构差异，推荐加强感知监督。")
        
        
import lpips
import torch
from PIL import Image
from torchvision import transforms

# 加载模型（默认用 AlexNet，最快）
loss_fn = lpips.LPIPS(net='alex').cuda()

# 预处理函数
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # LPIPS expects [-1,1]
])

# 加载图像并转为 [-1, 1] Tensor
def load_tensor(path):
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).cuda()

# 加载图像
gt = load_tensor("/home/qingran/Desktop/omniphysgs/outputs/dataset_45_07/images/20/gt_m_8_12.png")
pred1 = load_tensor("/home/qingran/Desktop/omniphysgs/outputs/dataset_45_07/images/20/m_8_12_.png")
pred2 = load_tensor("/home/qingran/Desktop/omniphysgs/outputs/dataset_45_07/images/20/m_8_12_sticky.png")

# 计算 LPIPS 距离（越小越好）
dist1 = loss_fn(gt, pred1)
dist2 = loss_fn(gt, pred2)

print("LPIPS (sticky vs GT):", dist1.item())
print("LPIPS (plain vs GT):", dist2.item())

if dist1 < dist2:
    print("✅ sticky 更接近 Ground Truth")
else:
    print("✅ plain 更接近 Ground Truth（可能不太对）")