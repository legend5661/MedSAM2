import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from skimage import measure

# 假设您的 predictor 和其他辅助函数都在这个文件中
from sam2.build_sam import build_sam2_video_predictor_text
# from your_utils import (
#     resize_grayscale_to_rgb_and_resize, 
#     getLargestCC, 
#     dice_multi_class
# )

# ===============================================================
#  注意：请确保以下辅助函数已定义，我从您上一个问题中复制过来
# ===============================================================

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    if labels.max() == 0: # 如果没有前景
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def dice_multi_class(preds, targets):
    smooth = 1.0
    assert preds.shape == targets.shape
    # 假设二分类，标签为 0 和 1
    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

def resize_grayscale_to_rgb_and_resize(array, image_size):
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size), Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1)
        resized_array[i] = img_array
    return resized_array

# ===============================================================
#  主测试脚本
# ==================================s=============================

def run_test(args):
    # 初始化模型
    # ！！！注意：这里需要您根据实际情况修改，特别是 build_sam2_video_predictor 的导入路径
    # 并且 predictor 类需要包含我们上面定义的 add_new_text 函数
    print("正在加载模型...")
    predictor = build_sam2_video_predictor_text(args.cfg, args.checkpoint)
    # predictor.add_new_text = add_new_text.__get__(predictor) # 动态绑定新方法
    print("模型加载完毕。")
    
    # 数据集路径
    test_img_base = os.path.join(args.data_dir, "reorganized_test")
    test_mask_base = os.path.join(args.data_dir, "reorganized_test_mask")
    os.makedirs(args.pred_save_dir, exist_ok=True)
    
    sample_ids = [d for d in os.listdir(test_img_base) if os.path.isdir(os.path.join(test_img_base, d))]
    
    all_dices = []
    
    pbar = tqdm(sample_ids, desc="正在测试 3D 样本")
    for sample_id in pbar:
        sample_img_dir = os.path.join(test_img_base, sample_id)
        sample_mask_dir = os.path.join(test_mask_base, sample_id)
        prompt_path = os.path.join(sample_img_dir, "prompt.json")
        
        if not os.path.exists(prompt_path):
            continue
            
        with open(prompt_path) as f:
            prompt_data = json.load(f)
            
        # 1. 加载图像和真值掩码 (Ground Truth Mask)
        slices_info = sorted(prompt_data['slices'], key=lambda x: x['slice_number'])
        
        img_3D_list = []
        gt_3D_list = []
        
        H, W = slices_info[0]['height'], slices_info[0]['width']
        
        for slice_info in slices_info:
            # 加载图像
            img_path = os.path.join(sample_img_dir, slice_info['slice_file'])
            img = np.array(Image.open(img_path).convert('L'))
            img_3D_list.append(img)
            
            # 加载并合并该切片的所有真值掩码
            slice_gt = np.zeros((H, W), dtype=np.uint8)
            for ann in slice_info['annotations']:
                mask_path = os.path.join(sample_mask_dir, ann['mask_file'])
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path).convert('L'))
                    slice_gt[mask > 0] = 1 # 合并所有对象的掩码
            gt_3D_list.append(slice_gt)
            
        img_3D_ori = np.stack(img_3D_list, axis=0) # Shape: [D, H, W]
        gt_3D = np.stack(gt_3D_list, axis=0)      # Shape: [D, H, W]
        
        # 2. 预处理图像
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
        img_resized = img_resized / 255.0
        img_resized = torch.from_numpy(img_resized).cuda()
        
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].cuda()
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].cuda()
        img_resized = (img_resized - img_mean) / img_std
        
        # 3. 确定关键帧和文本提示
        num_slices = len(slices_info)
        key_frame_idx = num_slices // 2  # 中间帧
        key_slice_info = slices_info[key_frame_idx]
        
        # 提取第一个标注的第一个句子作为提示
        text_prompt = "A lesion" # 默认提示
        if key_slice_info['annotations'] and key_slice_info['annotations'][0]['sentences']:
            text_prompt = key_slice_info['annotations'][0]['sentences'][0]
        
        pbar.set_postfix_str(f"样本: {sample_id}, 提示: '{text_prompt}'")
        
        # 4. 执行推理
        pred_3D = np.zeros_like(gt_3D)
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # 初始化状态
            inference_state = predictor.init_state(img_resized, video_height=H, video_width=W)

            # 使用文本提示
            predictor.add_new_text(
                inference_state=inference_state,
                frame_idx=key_frame_idx,
                obj_id=1,
                text_prompt=text_prompt
            )
            
            # 正向传播
            for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                pred_3D[out_frame_idx, :, :] = np.logical_or(pred_3D[out_frame_idx, :, :], mask)

            # 重置并反向传播
            predictor.reset_state(inference_state)
            predictor.add_new_text(
                inference_state=inference_state,
                frame_idx=key_frame_idx,
                obj_id=1,
                text_prompt=text_prompt
            )
            for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                pred_3D[out_frame_idx, :, :] = np.logical_or(pred_3D[out_frame_idx, :, :], mask)
                
        # 5. 后处理
        if np.max(pred_3D) > 0:
            pred_3D = getLargestCC(pred_3D)
        pred_3D = pred_3D.astype(np.uint8)
        
        # 6. 评估
        dice = dice_multi_class(pred_3D, gt_3D)
        all_dices.append(dice)
        pbar.set_postfix_str(f"样本: {sample_id}, Dice: {dice:.4f}")

        # 7. 保存预测结果
        sample_pred_dir = os.path.join(args.pred_save_dir, sample_id)
        os.makedirs(sample_pred_dir, exist_ok=True)
        for i in range(num_slices):
            pred_mask_img = Image.fromarray(pred_3D[i] * 255)
            # 沿用真值掩码的文件命名方式
            original_mask_name = slices_info[i]['annotations'][0]['mask_file'] if slices_info[i]['annotations'] else f"pred_mask_{i:03d}.png"
            pred_mask_img.save(os.path.join(sample_pred_dir, original_mask_name))
            
    # 计算并打印平均 Dice
    avg_dice = np.mean(all_dices)
    print(f"\n所有样本测试完成。")
    print(f"平均 Dice 分数: {avg_dice:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM2 Text-Prompted Testing Script")
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='包含 reorganized_test 和 reorganized_test_mask 的数据集根目录。'
    )
    parser.add_argument(
        '--pred_save_dir',
        type=str,
        default="./text_prompt_results",
        help='保存分割结果的路径。'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="/staff/wangtiantong/MedSAM2/exp_log/all_dataset_20250624-1617/checkpoints/checkpoint.pt",
        help='您的模型检查点路径。'
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default="/configs/sam2.1_hiera_t512_text_predict.yaml", # 示例配置
        help='模型配置文件路径。'
    )
    
    args = parser.parse_args()
    run_test(args)