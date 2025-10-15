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
import re

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

def getLargestCC(segmentation):
    """保留分割结果中最大的连通分量。"""
    labels = measure.label(segmentation)
    if labels.max() == 0:
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def dice_multi_class(preds, targets):
    """计算多类别的 Dice 分数（排除背景）。"""
    smooth = 1.0
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    assert preds.shape == targets.shape
    # 查找预测或目标中存在的所有唯一标签
    labels = np.union1d(np.unique(preds), np.unique(targets))
    labels = labels[labels != 0]  # 排除背景0
    
    if len(labels) == 0:
        return 1.0  # 如果两者都为空，则得分为1

    dices = []
    for label in labels:
        pred = (preds == label).astype(np.uint8)
        target = (targets == label).astype(np.uint8)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        dices.append(dice)
    return np.mean(dices) if dices else 1.0

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

def mask2D_to_bbox(gt2D, padding=0):
    """从 2D 掩码生成一个紧密的边界框。为保证评估的可复现性，移除了随机抖动。"""
    y_indices, x_indices = np.where(gt2D > 0)
    if len(y_indices) == 0:
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape

    # 修改：为评估使用确定性的边界框
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(W - 1, x_max + padding)
    y_max = min(H - 1, y_max + padding)
    
    if x_min >= x_max: x_max = x_min + 1
    if y_min >= y_max: y_max = y_min + 1
    return np.array([x_min, y_min, x_max, y_max])

def find_best_key_frame(class_gt_3D):
    """为特定类别找到最佳关键帧（定义为具有最大分割区域的帧）。"""
    if class_gt_3D.max() == 0:
        return class_gt_3D.shape[0] // 2
    areas = np.sum(class_gt_3D, axis=(1, 2))
    return np.argmax(areas)

def visualize_results(slice_img, gt_mask, pred_mask, save_path):
    """将真实标签和预测结果分别叠加展示在原图像的左右子图中，并保存。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    gt_empty = gt_mask.max() == 0
    pred_empty = pred_mask.max() == 0
    
    axes[0].imshow(slice_img, cmap='gray')
    if not gt_empty:
        axes[0].imshow(gt_mask, alpha=0.5, cmap='jet')
    axes[0].set_title(f"GT Overlay {'(Empty)' if gt_empty else ''}")
    axes[0].axis('off')
    
    axes[1].imshow(slice_img, cmap='gray')
    if not pred_empty:
        axes[1].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[1].set_title(f"Pred Overlay {'(Empty)' if pred_empty else ''}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    # print(f"保存可视化: {save_path}, GT非空: {not gt_empty}, Pred非空: {not pred_empty}")

def main():
    parser = argparse.ArgumentParser("MedSAM2 Multi-class Evaluation Script (Revised)")
    parser.add_argument('--data_dir', type=str,
                        default="/staff/wangtiantong/SAM2_new/dataset/amos22/MRI",
                        help='数据集根目录，包含 reorganized_test 和 reorganized_test_mask。')
    parser.add_argument('--pred_save_dir', type=str, default="./MedSAM2_results/AMOS-MRI-Revised",
                        help='保存分割结果的目录路径。')
    parser.add_argument('--checkpoint', type=str,
                        default="/staff/wangtiantong/MedSAM2/checkpoints/MedSAM2_latest.pt",
                        help='MedSAM2 预训练模型权重路径。')
    parser.add_argument('--cfg', type=str, default="/configs/sam2.1_hiera_t512.yaml",
                        help='模型配置文件路径。')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='是否保存可视化结果。')
    parser.add_argument('--use_largest_cc', action='store_true', default=False,
                        help='是否在每个类别上应用“最大连通分量”后处理。默认为False。')
    args = parser.parse_args()
    os.makedirs(args.pred_save_dir, exist_ok=True)

    print("正在加载 MedSAM2 预训练模型...")
    from sam2.build_sam import build_sam2_video_predictor_npz
    predictor = build_sam2_video_predictor_npz(args.cfg, args.checkpoint)
    print("模型加载完毕。")

    test_img_base = os.path.join(args.data_dir, "reorganized_test")
    test_mask_base = os.path.join(args.data_dir, "reorganized_test_mask")
    sample_ids = [d for d in os.listdir(test_img_base) if os.path.isdir(os.path.join(test_img_base, d))]

    results_summary = OrderedDict(sample_id=[], dice_score=[], num_classes=[])

    pbar = tqdm(sample_ids, desc="正在测试 3D 样本")
    for sample_id in pbar:
        pbar.set_description(f"处理中: {sample_id}")
        sample_img_dir = os.path.join(test_img_base, sample_id)
        sample_mask_dir = os.path.join(test_mask_base, sample_id)
        prompt_path = os.path.join(sample_img_dir, "prompt.json")
        if not os.path.exists(prompt_path):
            print(f"跳过样本 {sample_id}: 缺少 prompt.json")
            continue
        
        with open(prompt_path) as f:
            prompt_data = json.load(f)
        class_mapping = prompt_data.get('all_obj', {})
        if not class_mapping:
            print(f"跳过样本 {sample_id}: 缺少类别映射")
            continue

        name2id = {name.lower().strip(): cid for name, cid in class_mapping.items()}
        id2name = {cid: name for name, cid in class_mapping.items()}
        
        slices_info = sorted(prompt_data['slices'], key=lambda x: x['slice_number'])
        if not slices_info:
            print(f"跳过样本 {sample_id}: 没有切片信息")
            continue
        
        first_img = np.array(Image.open(os.path.join(sample_img_dir, slices_info[0]['slice_file'])).convert('L'))
        H, W = first_img.shape
        num_slices = len(slices_info)

        img_3D_ori = np.zeros((num_slices, H, W), dtype=np.uint8)
        for i, si in enumerate(slices_info):
            img_path = os.path.join(sample_img_dir, si['slice_file'])
            if os.path.exists(img_path):
                img_3D_ori[i] = np.array(Image.open(img_path).convert('L'))

        img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512) / 255.0
        img_resized = torch.from_numpy(img_resized).cuda()
        img_mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None].cuda()
        img_std = torch.tensor([0.229, 0.224, 0.225])[:,None,None].cuda()
        img_resized = (img_resized - img_mean) / img_std

        gt_3D_multiclass = np.zeros((num_slices, H, W), dtype=np.uint8)
        all_mask_files = os.listdir(sample_mask_dir)
        for i, si in enumerate(slices_info):
            slice_gt = np.zeros((H, W), dtype=np.uint8)
            slice_base_name = os.path.splitext(si['slice_file'])[0]
            corresponding_masks = [f for f in all_mask_files if f.startswith(slice_base_name)]
            for mask_file in corresponding_masks:
                mask = np.array(Image.open(os.path.join(sample_mask_dir, mask_file)).convert('L'))
                category_part = mask_file.split('_')[-1]
                category_name = os.path.splitext(category_part)[0].replace('+', ' ').lower().strip()
                cid = name2id.get(category_name, 0)
                if cid > 0:
                    slice_gt[mask > 0] = cid
            gt_3D_multiclass[i] = slice_gt
        
        existing_classes = sorted([cid for cid in class_mapping.values() if np.any(gt_3D_multiclass == cid)])
        if not existing_classes:
            print(f"警告: 样本 {sample_id} 在GT中没有任何有效类别。跳过。")
            continue

        print(f"样本 {sample_id} 存在的类别: {[id2name.get(c, f'未知{c}') for c in existing_classes]}")
        num_classes_processed = len(existing_classes)
        
        # --- 修正重叠问题的逻辑 ---
        # 1. 为每个类别的 logits 创建一个占位符
        pred_logits_4D = np.full((num_classes_processed, num_slices, H, W), -100.0, dtype=np.float32)
        class_id_to_idx = {cid: i for i, cid in enumerate(existing_classes)}

        for i, class_id in enumerate(existing_classes):
            class_gt = (gt_3D_multiclass == class_id).astype(np.uint8)
            
            # 修改：使用更稳健的关键帧选择策略
            kf = find_best_key_frame(class_gt)
            class_name = id2name.get(class_id, f'Class {class_id}')
            print(f"处理类别 {class_name} (ID: {class_id})。最佳关键帧(最大面积): {kf}")
            
            if class_gt[kf].sum() == 0:
                print(f"警告: 类别 {class_name} 在关键帧 {kf} 没有标注，跳过此类别的预测。")
                continue

            # 修改：使用确定性的紧密边界框
            bbox = mask2D_to_bbox(class_gt[kf]) 
            if bbox is None:
                print(f"警告: 类别 {class_name} 无法生成边界框，跳过。")
                continue

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = predictor.init_state(img_resized, video_height=H, video_width=W)
                
                # 正向传播
                predictor.add_new_points_or_box(st, frame_idx=kf, obj_id=1, box=bbox)
                for idx, _, logits_tensor in predictor.propagate_in_video(st):
                    logits_np = logits_tensor[0, 0].cpu().numpy()
                    pred_logits_4D[i, idx] = np.maximum(pred_logits_4D[i, idx], logits_np)
                predictor.reset_state(st)
                
                # 反向传播
                predictor.add_new_points_or_box(st, frame_idx=kf, obj_id=1, box=bbox)
                for idx, _, logits_tensor in predictor.propagate_in_video(st, reverse=True):
                    logits_np = logits_tensor[0, 0].cpu().numpy()
                    pred_logits_4D[i, idx] = np.maximum(pred_logits_4D[i, idx], logits_np)
                predictor.reset_state(st)
            
            # 实时dice
            pred_bin = (np.argmax(pred_logits_4D, axis=0) == i) & (np.max(pred_logits_4D, axis=0) > 0)
            pred_bin = pred_bin.astype(np.uint8)
            gt_bin = (gt_3D_multiclass == class_id).astype(np.uint8)
            intersection = (gt_bin * pred_bin).sum()
            dice_cls = (2.0 * intersection + 1.0) / (gt_bin.sum() + pred_bin.sum() + 1.0)
            print(f"  >> 类别 {class_name} (ID: {class_id}) 当前 Dice: {dice_cls:.4f}")
        
        # --- 修正：从 Logits 生成最终预测掩码 ---
        pred_indices = np.argmax(pred_logits_4D, axis=0)
        max_logits = np.max(pred_logits_4D, axis=0)
        
        pred_3D_multiclass = np.zeros_like(gt_3D_multiclass)
        idx_to_class_id = {i: cid for cid, i in class_id_to_idx.items()}

        for i in range(num_classes_processed):
            class_id = idx_to_class_id[i]
            # 如果一个像素的类别索引是i，并且其最大logit大于0，则分配该类别ID
            pred_3D_multiclass[(pred_indices == i) & (max_logits > 0.0)] = class_id

        # --- 可选的后处理 ---
        if args.use_largest_cc:
            print("应用“最大连通分量”后处理...")
            final_processed_mask = np.zeros_like(pred_3D_multiclass)
            for class_id in existing_classes:
                class_mask = (pred_3D_multiclass == class_id)
                if class_mask.max() > 0:
                    largest_cc_mask = getLargestCC(class_mask)
                    final_processed_mask[largest_cc_mask > 0] = class_id
            pred_3D_multiclass = final_processed_mask
        
        dice = dice_multi_class(pred_3D_multiclass, gt_3D_multiclass)
        results_summary['sample_id'].append(sample_id)
        results_summary['dice_score'].append(dice)
        results_summary['num_classes'].append(len(existing_classes))
        pbar.set_postfix_str(f"平均 Dice: {dice:.4f}, 类别数: {len(existing_classes)}")

        save_dir = os.path.join(args.pred_save_dir, sample_id)
        os.makedirs(save_dir, exist_ok=True)
        for i in range(num_slices):
            pm = pred_3D_multiclass[i]
            cm = np.zeros((H, W, 3), dtype=np.uint8)
            unique_preds = np.unique(pm)
            for cid in unique_preds[unique_preds != 0]:
                color = (np.array(plt.cm.tab10(int(cid) % 10)[:3]) * 255).astype(np.uint8)
                cm[pm == cid] = color
            pred_path = os.path.join(save_dir, slices_info[i]['slice_file'])
            Image.fromarray(cm).save(pred_path)
        
        if args.visualize:
            pbar.set_description(f"可视化: {sample_id}")
            all_involved_classes = np.union1d(np.unique(gt_3D_multiclass), np.unique(pred_3D_multiclass))
            all_involved_classes = all_involved_classes[all_involved_classes != 0]

            for cid in all_involved_classes:
                gt_class_vol = (gt_3D_multiclass == cid)
                pred_class_vol = (pred_3D_multiclass == cid)
                relevant_slices = np.where(np.any(gt_class_vol, axis=(1,2)) | np.any(pred_class_vol, axis=(1,2)))[0]

                for i in relevant_slices:
                    cname = id2name.get(cid, f"class_{cid}")
                    base, ext = os.path.splitext(slices_info[i]['slice_file'])
                    cname_safe = re.sub(r'[^\w]', '_', cname)
                    viz_filename = f"{base}_{cname_safe}{ext}"
                    viz_path = os.path.join(save_dir, viz_filename)
                    visualize_results(img_3D_ori[i], gt_class_vol[i], pred_class_vol[i], viz_path)

    df = pd.DataFrame(results_summary)
    results_csv = os.path.join(args.pred_save_dir, 'results_summary.csv')
    df.to_csv(results_csv, index=False)
    print(f"\n所有样本测试完成，结果已保存至 {args.pred_save_dir}")
    if not df.empty:
        print(f"整体平均 Dice: {df['dice_score'].mean():.4f}, 总处理类别实例数: {df['num_classes'].sum()}")
    print(f"详细结果保存至: {results_csv}")

if __name__ == "__main__":
    main()