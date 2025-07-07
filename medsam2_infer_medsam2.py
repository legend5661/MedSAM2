import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from skimage import measure
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt

# 设置 PyTorch 和 NumPy 的随机种子和精度
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

# --- 辅助函数 --- 

def getLargestCC(segmentation):
    """保留分割结果中最大的连通分量。"""
    labels = measure.label(segmentation)
    if labels.max() == 0:  # 如果没有前景对象
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def dice_multi_class(preds, targets):
    """计算多类别的 Dice 分数。"""
    smooth = 1.0
    assert preds.shape == targets.shape
    labels = np.unique(targets)[1:]  # 排除背景0
    dices = []
    for label in labels:
        pred = (preds == label).astype(np.uint8)
        target = (targets == label).astype(np.uint8)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        dices.append(dice)
    return np.mean(dices)

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """将 3D 灰度 NumPy 数组转换为 RGB 并调整大小。"""
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size), Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1)
        resized_array[i] = img_array
    return resized_array

def mask2D_to_bbox(gt2D, max_shift=5):
    """从 2D 掩码生成一个略微抖动的边界框。"""
    y_indices, x_indices = np.where(gt2D > 0)
    if len(y_indices) == 0:
        return None  # 如果掩码为空，则返回 None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape
    
    # 添加随机抖动，模拟真实的手动标注
    bbox_shift = np.random.randint(-max_shift, max_shift + 1, 4)
    x_min = max(0, x_min - bbox_shift[0])
    y_min = max(0, y_min - bbox_shift[1])
    x_max = min(W - 1, x_max + bbox_shift[2])
    y_max = min(H - 1, y_max + bbox_shift[3])
    
    # 确保 min < max
    if x_min >= x_max: x_max = x_min + 1
    if y_min >= y_max: y_max = y_min + 1

    return np.array([x_min, y_min, x_max, y_max])

def find_key_frame_for_class(class_gt_3D):
    """为特定类别找到最合适的关键帧（中间有标注的帧）"""
    num_slices = class_gt_3D.shape[0]
    mid_idx = num_slices // 2
    
    # 检查中间帧是否有标注
    if np.any(class_gt_3D[mid_idx] > 0):
        return mid_idx
    
    # 向两侧搜索最近的标注帧
    for offset in range(1, num_slices // 2 + 1):
        upper_idx = mid_idx + offset
        lower_idx = mid_idx - offset
        
        if upper_idx < num_slices and np.any(class_gt_3D[upper_idx] > 0):
            return upper_idx
        if lower_idx >= 0 and np.any(class_gt_3D[lower_idx] > 0):
            return lower_idx
    
    # 如果整个序列都没有标注，返回中间帧（虽然不太可能）
    return mid_idx

def visualize_results(slice_img, gt_mask, pred_mask, bbox, save_path):
    """可视化结果：原始图像、真值掩码和预测掩码"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(slice_img, cmap='gray')
    axes[0].set_title("Original Image")
    
    # 真值掩码
    axes[1].imshow(slice_img, cmap='gray')
    axes[1].imshow(gt_mask, alpha=0.5, cmap='jet')
    axes[1].set_title("Ground Truth")
    
    # 预测结果
    axes[2].imshow(slice_img, cmap='gray')
    axes[2].imshow(pred_mask, alpha=0.5, cmap='jet')
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        w, h = x_max - x_min, y_max - y_min
        rect = plt.Rectangle((x_min, y_min), w, h, 
                             edgecolor='red', facecolor='none', linewidth=2)
        axes[2].add_patch(rect)
    axes[2].set_title("Prediction with BBox")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# --- 主程序 ---

def main():
    parser = argparse.ArgumentParser("MedSAM2 Pre-trained Model Test Script for Multi-Class PNG Dataset")
    
    # --- 参数定义 ---
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        default="/staff/wangtiantong/SAM2_new/dataset/ACDC",
        help='包含 reorganized_test 和 reorganized_test_mask 的数据集根目录。'
    )
    parser.add_argument(
        '--pred_save_dir',
        type=str,
        default="./MedSAM2_results/ACDC",
        help='保存分割结果的路径。'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="/staff/wangtiantong/MedSAM2/checkpoints/MedSAM2_latest.pt",
        help='MedSAM2 预训练模型检查点路径。'
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default="/configs/sam2.1_hiera_t512.yaml",
        help='模型配置文件路径。'
    )
    parser.add_argument(
        '--propagate_with_box',
        action='store_true',
        default=True,
        help='使用边界框作为提示进行传播。'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=False,
        help='保存可视化结果（会显著增加处理时间）。'
    )
    
    args = parser.parse_args()
    os.makedirs(args.pred_save_dir, exist_ok=True)

    # --- 初始化模型 ---
    print("正在加载 MedSAM2 预训练模型...")
    from sam2.build_sam import build_sam2_video_predictor_npz
    predictor = build_sam2_video_predictor_npz(args.cfg, args.checkpoint)
    print("模型加载完毕。")
    
    # --- 数据集路径 ---
    test_img_base = os.path.join(args.data_dir, "reorganized_test")
    test_mask_base = os.path.join(args.data_dir, "reorganized_test_mask")
    
    sample_ids = [d for d in os.listdir(test_img_base) if os.path.isdir(os.path.join(test_img_base, d))]
    
    # --- 结果记录 ---
    results_summary = OrderedDict()
    results_summary['sample_id'] = []
    results_summary['dice_score'] = []
    results_summary['num_classes'] = []
    
    # --- 主循环 ---
    pbar = tqdm(sample_ids, desc="正在测试 3D 样本")
    for sample_id in pbar:
        pbar.set_description(f"处理中: {sample_id}")
        sample_img_dir = os.path.join(test_img_base, sample_id)
        sample_mask_dir = os.path.join(test_mask_base, sample_id)
        prompt_path = os.path.join(sample_img_dir, "prompt.json")

        if not os.path.exists(prompt_path):
            print(f"警告：在 {sample_img_dir} 中未找到 prompt.json，跳过此样本。")
            continue
        
        with open(prompt_path) as f:
            prompt_data = json.load(f)
        
        # 获取类别映射
        class_mapping = prompt_data.get('all_obj', {})
        if not class_mapping:
            print(f"警告：在 {prompt_path} 中未找到类别映射，跳过此样本。")
            continue
        
        # 1. 加载图像和真值掩码（多类别）
        slices_info = sorted(prompt_data['slices'], key=lambda x: x['slice_number'])
        img_3D_list = []
        gt_3D_multiclass = []  # 存储多类别真值掩码
        
        # 初始化真值数组
        first_img = np.array(Image.open(os.path.join(sample_img_dir, slices_info[0]['slice_file'])).convert('L'))
        H, W = first_img.shape
        num_slices = len(slices_info)
        
        # 初始化真值掩码（多类别）
        gt_3D_multiclass = np.zeros((num_slices, H, W), dtype=np.uint8)
        
        for i, slice_info in enumerate(slices_info):
            # 加载图像
            img_path = os.path.join(sample_img_dir, slice_info['slice_file'])
            img = np.array(Image.open(img_path).convert('L'))
            img_3D_list.append(img)
            
            # 初始化当前切片的多类别掩码
            slice_gt = np.zeros((H, W), dtype=np.uint8)
            
            # 处理每个标注
            for ann in slice_info.get('annotations', []):
                mask_path = os.path.join(sample_mask_dir, ann['mask_file'])
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path).convert('L'))
                    category_name = ann['category']
                    class_id = class_mapping.get(category_name, 0)
                    if class_id > 0:
                        # 将当前类别的掩码设置为对应的类别ID
                        slice_gt[mask > 0] = class_id
            
            gt_3D_multiclass[i] = slice_gt
            
        img_3D_ori = np.stack(img_3D_list, axis=0)
        
        # 2. 初始化多类别预测结果
        pred_3D_multiclass = np.zeros_like(gt_3D_multiclass)
        
        # 3. 获取存在的类别
        existing_classes = []
        for class_name, class_id in class_mapping.items():
            if np.any(gt_3D_multiclass == class_id):
                existing_classes.append(class_id)
        
        # 记录处理的类别数量
        num_classes_processed = 0
        
        # 4. 对每个存在的类别进行分割
        for class_id in existing_classes:
            # 提取当前类别的二值掩码
            class_gt = (gt_3D_multiclass == class_id).astype(np.uint8)
            
            # 找到该类别最合适的关键帧
            key_frame_idx = find_key_frame_for_class(class_gt)
            key_frame_gt_mask = class_gt[key_frame_idx]
            
            if np.sum(key_frame_gt_mask) == 0:
                # 如果关键帧没有该类别，跳过
                continue
            
            # 生成边界框提示
            bbox_prompt = mask2D_to_bbox(key_frame_gt_mask)
            if bbox_prompt is None:
                continue
            
            # 3. 预处理图像
            img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)
            img_resized = img_resized / 255.0
            img_resized = torch.from_numpy(img_resized).cuda()

            img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].cuda()
            img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].cuda()
            img_resized = (img_resized - img_mean) / img_std

            # 4. 执行推理（当前类别）
            class_pred = np.zeros_like(class_gt)
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = predictor.init_state(img_resized, video_height=H, video_width=W)

                # 使用动态生成的 bbox 进行提示
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=key_frame_idx,
                    obj_id=1,
                    box=bbox_prompt,
                )

                # 正向传播
                for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                    class_pred[out_frame_idx] = np.logical_or(class_pred[out_frame_idx], mask)

                # 重置并反向传播
                predictor.reset_state(inference_state)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=key_frame_idx,
                    obj_id=1,
                    box=bbox_prompt,
                )
                for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
                    class_pred[out_frame_idx] = np.logical_or(class_pred[out_frame_idx], mask)
            
            # 后处理：保留最大连通分量
            if np.max(class_pred) > 0:
                class_pred = getLargestCC(class_pred).astype(np.uint8)
            
            # 将当前类别的预测结果合并到多类别预测中
            pred_3D_multiclass[class_pred > 0] = class_id
            num_classes_processed += 1
            
            # 可视化关键帧结果
            if args.visualize and key_frame_idx == num_slices // 2:
                vis_dir = os.path.join(args.pred_save_dir, "visualizations", sample_id)
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"class_{class_id}_keyframe.png")
                visualize_results(
                    img_3D_ori[key_frame_idx],
                    gt_3D_multiclass[key_frame_idx],
                    pred_3D_multiclass[key_frame_idx],
                    bbox_prompt,
                    vis_path
                )
        
        # 如果没有处理任何类别，跳过评估
        if num_classes_processed == 0:
            print(f"警告：样本 {sample_id} 没有处理任何类别，跳过。")
            continue
        
        # 6. 评估与记录
        dice = dice_multi_class(pred_3D_multiclass, gt_3D_multiclass)
        results_summary['sample_id'].append(sample_id)
        results_summary['dice_score'].append(dice)
        results_summary['num_classes'].append(num_classes_processed)
        pbar.set_postfix_str(f"Dice: {dice:.4f}, Classes: {num_classes_processed}")

        # 7. 保存预测结果 (PNG格式)
        sample_pred_dir = os.path.join(args.pred_save_dir, sample_id)
        os.makedirs(sample_pred_dir, exist_ok=True)
        for i in range(num_slices):
            # 创建彩色预测图
            pred_mask = pred_3D_multiclass[i]
            color_mask = np.zeros((H, W, 3), dtype=np.uint8)
            
            # 为每个类别分配颜色
            for class_id in existing_classes:
                color_mask[pred_mask == class_id] = plt.cm.tab10(class_id % 10)[:3] * 255
            
            pred_mask_img = Image.fromarray(color_mask)
            
            # 使用原始切片文件名来命名预测掩码
            slice_filename = slices_info[i]['slice_file']
            pred_mask_img.save(os.path.join(sample_pred_dir, slice_filename))

    # --- 保存总结报告 ---
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(args.pred_save_dir, 'results_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    avg_dice = summary_df['dice_score'].mean()
    total_classes = summary_df['num_classes'].sum()
    print(f"\n所有样本测试完成。")
    print(f"结果已保存至: {args.pred_save_dir}")
    print(f"平均 Dice 分数: {avg_dice:.4f}")
    print(f"处理的类别总数: {total_classes}")


if __name__ == "__main__":
    main()