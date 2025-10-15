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

# 设置 PyTorch 和 NumPy 的随机种子和精度
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

# --- 辅助函数 --- 

def getLargestCC(segmentation):
    """保留分割结果中最大的连通分量。"""
    labels = measure.label(segmentation)
    if labels.max() == 0:
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


def mask2D_to_bbox(gt2D, max_shift=5): # You can adjust max_shift
    """从 2D 掩码生成一个略微扩大的边界框。"""
    y_indices, x_indices = np.where(gt2D > 0)
    if len(y_indices) == 0:
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape

    # Generate a single, positive shift value to expand the box
    shift = np.random.randint(0, max_shift + 1)
    
    x_min = max(0, x_min - shift)
    y_min = max(0, y_min - shift)
    x_max = min(W - 1, x_max + shift)
    y_max = min(H - 1, y_max + shift)
    
    # Ensure box has a valid area
    if x_min >= x_max: x_max = x_min + 1
    if y_min >= y_max: y_max = y_min + 1
        
    return np.array([x_min, y_min, x_max, y_max])


def find_key_frame_for_class(class_gt_3D):
    """为特定类别找到最合适的关键帧（中间有标注的帧）"""
    num_slices = class_gt_3D.shape[0]
    mid_idx = num_slices // 2
    if np.any(class_gt_3D[mid_idx] > 0):
        return mid_idx
    for offset in range(1, num_slices // 2 + 1):
        upper_idx = mid_idx + offset
        lower_idx = mid_idx - offset
        if upper_idx < num_slices and np.any(class_gt_3D[upper_idx] > 0):
            return upper_idx
        if lower_idx >= 0 and np.any(class_gt_3D[lower_idx] > 0):
            return lower_idx
    return mid_idx


def visualize_results(slice_img, gt_mask, pred_mask, save_path):
    """将真实标签和预测结果分别叠加展示在原图像的左右两个子图中，并保存同一张 PNG。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 检查掩码是否为空
    gt_empty = gt_mask.max() == 0
    pred_empty = pred_mask.max() == 0
    
    # Ground Truth overlay
    axes[0].imshow(slice_img, cmap='gray')
    if not gt_empty:
        axes[0].imshow(gt_mask, alpha=0.5, cmap='jet')
    axes[0].set_title(f"GT Overlay {'(Empty)' if gt_empty else ''}")
    axes[0].axis('off')
    
    # Prediction overlay
    axes[1].imshow(slice_img, cmap='gray')
    if not pred_empty:
        axes[1].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[1].set_title(f"Pred Overlay {'(Empty)' if pred_empty else ''}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    # 添加调试信息
    print(f"保存可视化: {save_path}, GT非空: {not gt_empty}, Pred非空: {not pred_empty}")


def main():
    parser = argparse.ArgumentParser("MedSAM2 Pre-trained Model Test Script for Multi-Class PNG Dataset")
    parser.add_argument('--data_dir', type=str,
                        default="/staff/wangtiantong/SAM2_new/dataset/amos22/MRI",
                        help='包含 reorganized_test 和 reorganized_test_mask 的数据集根目录。')
    parser.add_argument('--pred_save_dir', type=str, default="./MedSAM2_results/AMOS_MRI",
                        help='保存分割结果的路径。')
    parser.add_argument('--checkpoint', type=str,
                        default="/staff/wangtiantong/MedSAM2/checkpoints/MedSAM2_latest.pt",
                        help='MedSAM2 预训练模型检查点路径。')
    parser.add_argument('--cfg', type=str, default="/configs/sam2.1_hiera_t512.yaml",
                        help='模型配置文件路径。')
    parser.add_argument('--propagate_with_box', action='store_true', default=True,
                        help='使用边界框作为提示进行传播。')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='保存可视化结果（会显著增加处理时间）。')
    args = parser.parse_args()
    os.makedirs(args.pred_save_dir, exist_ok=True)

    print("正在加载 MedSAM2 预训练模型...")
    from sam2.build_sam import build_sam2_video_predictor_npz
    predictor = build_sam2_video_predictor_npz(args.cfg, args.checkpoint)
    print("模型加载完毕。")

    test_img_base = os.path.join(args.data_dir, "reorganized_test")
    test_mask_base = os.path.join(args.data_dir, "reorganized_test_mask")
    sample_ids = [d for d in os.listdir(test_img_base) if os.path.isdir(os.path.join(test_img_base, d))]

    results_summary = OrderedDict(sample_id=[], dice_score=[], num_classes=[] )
    smooth = 1.0

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

        # 创建名称到ID的反向映射（小写处理）
        name2id = {name.lower().strip(): cid for name, cid in class_mapping.items()}
        id2name = {cid: name for name, cid in class_mapping.items()}

        print("name2id:", name2id)
        print("id2name:", id2name)
        
        slices_info = sorted(prompt_data['slices'], key=lambda x: x['slice_number'])
        if not slices_info:
            print(f"跳过样本 {sample_id}: 没有切片信息")
            continue
            
        first_img = np.array(Image.open(os.path.join(sample_img_dir, slices_info[0]['slice_file'])).convert('L'))
        H, W = first_img.shape
        num_slices = len(slices_info)

        img_3D_ori = np.zeros((num_slices, H, W), dtype=np.uint8)
        for i, si in enumerate(slices_info):
            slice_file = si['slice_file']
            img_path = os.path.join(sample_img_dir, slice_file)
            if not os.path.exists(img_path):
                print(f"警告: 图像文件不存在 - {img_path}")
                continue
            img_3D_ori[i] = np.array(Image.open(img_path).convert('L'))


        # 预处理
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512) / 255.0
        img_resized = torch.from_numpy(img_resized).cuda()
        img_mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None].cuda()
        img_std = torch.tensor([0.229, 0.224, 0.225])[:,None,None].cuda()
        img_resized = (img_resized - img_mean) / img_std


        # 构建GT掩码 - 通过直接扫描掩码目录来确保加载所有存在的GT
        print(f"为样本 {sample_id} 构建GT掩码，通过直接扫描文件...")
        gt_3D_multiclass = np.zeros((num_slices, H, W), dtype=np.uint8)
        # 预先读取该样本的所有掩码文件名，提高效率
        all_mask_files_in_sample = os.listdir(sample_mask_dir)
        print(f"样本 {sample_id} 掩码目录中的文件数: {len(all_mask_files_in_sample)}")

        for i, si in enumerate(slices_info):
            slice_gt = np.zeros((H, W), dtype=np.uint8)
            # 获取当前切片图像文件的基本名，例如 "amos_ct_0001_0095"
            slice_base_name = os.path.splitext(si['slice_file'])[0]

            # 从所有掩码文件中筛选出属于当前切片的文件
            # 例如，找出所有以 "amos_ct_0001_0095" 开头的文件
            corresponding_mask_files = [f for f in all_mask_files_in_sample if f.startswith(slice_base_name)]
            # print(f"切片 {i} ({slice_base_name}) 对应的掩码文件: {len(corresponding_mask_files)} 个")

            for mask_file in corresponding_mask_files:
                mask_path = os.path.join(sample_mask_dir, mask_file)
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path).convert('L'))
                    
                    # --- 这部分逻辑与您原来的一致，是正确的 ---
                    # 使用下划线分割文件名，取最后一部分
                    category_part = mask_file.split('_')[-1]
                    # 去除文件扩展名
                    category_name = os.path.splitext(category_part)[0]
                    # 将 '+' 替换为空格并转为小写
                    category_name = category_name.replace('+', ' ').lower().strip()
                    
                    # print(f"处理掩码文件: {mask_file}, 类别: {category_name}")

                    # 从之前创建的映射中获取类别ID
                    cid = name2id.get(category_name, 0)
                    
                    if cid > 0:
                        # 将掩码区域赋值为对应的类别ID
                        mask_sum = (mask > 0).sum()
                        slice_gt[mask > 0] = cid
                        # print(f"  -> 类别ID: {cid}, 掩码像素数: {mask_sum}")
                    else:
                        print(f"  -> 警告: 类别 '{category_name}' 未找到对应的ID")
                # (这里可以省略不存在的警告，因为我们就是从存在的文件列表里读取的)
                
            gt_3D_multiclass[i] = slice_gt

        # 调试输出：验证类别匹配
        unique_classes = np.unique(gt_3D_multiclass)
        unique_classes = [c for c in unique_classes if c != 0]
        if unique_classes:
            print(f"样本 {sample_id} - GT类别: {[id2name.get(c, f'未知{c}') for c in unique_classes]}")
        else:
            print(f"警告: 样本 {sample_id} 没有有效的GT类别")

        pred_3D_multiclass = np.zeros_like(gt_3D_multiclass)
        existing_classes = [cid for cid in class_mapping.values() if np.any(gt_3D_multiclass == cid)]

        print(f"样本 {sample_id} 存在的类别: {[id2name.get(c, f'未知{c}') for c in existing_classes]}")
        num_classes_processed = 0

        for class_id in existing_classes:
            class_gt = (gt_3D_multiclass == class_id).astype(np.uint8)
            kf = find_key_frame_for_class(class_gt)
            print(f"处理类别 {id2name[class_id]} (ID: {class_id}) 的关键帧: {kf}")
            if class_gt[kf].sum() == 0: 
                print(f"警告: 类别 {id2name[class_id]} 在关键帧 {kf} 没有标注")
                continue
                
            bbox = mask2D_to_bbox(class_gt[kf])
            if bbox is None: 
                print(f"警告: 类别 {id2name[class_id]} 无法生成边界框")
                continue

            class_pred = np.zeros_like(class_gt)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = predictor.init_state(img_resized, video_height=H, video_width=W)
                predictor.add_new_points_or_box(st, frame_idx=kf, obj_id=class_id, box=bbox)
                
                # Forward pass
                for idx, _, logits in predictor.propagate_in_video(st):
                    mask = (logits[0] > 0.0).cpu().numpy()[0]
                    class_pred[idx] |= mask
                
                predictor.reset_state(st)

                predictor.add_new_points_or_box(st, frame_idx=kf, obj_id=class_id, box=bbox)
                    
                # Backward pass - uses the state from the end of the forward pass
                # DO NOT reset or re-add the prompt here.
                for idx, _, logits in predictor.propagate_in_video(st, reverse=True):
                    mask = (logits[0] > 0.0).cpu().numpy()[0]
                    class_pred[idx] |= mask
    
                predictor.reset_state(st)

            # After the loop, apply post-processing
            if class_pred.max() > 0:
                class_pred = getLargestCC(class_pred).astype(np.uint8)

            intersection = (class_pred * class_gt).sum()
            dice_cls = (2.0 * intersection + smooth) / (class_pred.sum() + class_gt.sum() + smooth)
            cname = id2name[class_id]
            print(f"样本 {sample_id}, 类别 '{cname}', Dice: {dice_cls:.4f}")

            # 重要修改：只有在预测结果存在时才添加到最终结果中
            if class_pred.max() > 0:
                pred_3D_multiclass[class_pred > 0] = class_id
            num_classes_processed += 1

        if num_classes_processed == 0: 
            print(f"警告: 样本 {sample_id} 没有处理任何类别")
            continue
            
        dice = dice_multi_class(pred_3D_multiclass, gt_3D_multiclass)
        results_summary['sample_id'].append(sample_id)
        results_summary['dice_score'].append(dice)
        results_summary['num_classes'].append(num_classes_processed)
        pbar.set_postfix_str(f"平均 Dice: {dice:.4f}, 类别数: {num_classes_processed}")

        save_dir = os.path.join(args.pred_save_dir, sample_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存多类别彩色掩码
        for i in range(num_slices):
            pm = pred_3D_multiclass[i]
            cm = np.zeros((H, W, 3), dtype=np.uint8)
            for cid in existing_classes:
                mask_idx = pm == cid
                color = (np.array(plt.cm.tab10(cid % 10)[:3]) * 255).astype(np.uint8)
                cm[mask_idx] = color
            pred_path = os.path.join(save_dir, slices_info[i]['slice_file'])
            Image.fromarray(cm).save(pred_path)

        # 可视化结果
        if args.visualize:
            pbar.set_description(f"可视化: {sample_id}")
            for i in range(num_slices):
                slice_img = img_3D_ori[i]
                original_slice_filename = slices_info[i]['slice_file']
                
                # 获取当前切片实际存在的类别
                slice_classes = np.unique(gt_3D_multiclass[i])
                slice_classes = [c for c in slice_classes if c != 0]  # 排除背景
                
                # 添加预测中存在的类别
                pred_classes = np.unique(pred_3D_multiclass[i])
                slice_classes = list(set(slice_classes) | set([c for c in pred_classes if c != 0]))
                
                for class_id in slice_classes:
                    cname = id2name.get(class_id, f"class_{class_id}")
                    # 使用原始JSON中的类别名称，不进行额外处理
                    gt_mask_cls = (gt_3D_multiclass[i] == class_id).astype(np.uint8)
                    pred_mask_cls = (pred_3D_multiclass[i] == class_id).astype(np.uint8)
                    
                    # 调试输出：检查当前切片的GT和预测情况
                    gt_has_mask = gt_mask_cls.max() > 0
                    pred_has_mask = pred_mask_cls.max() > 0
                    
                    print(f"切片 {i}, 类别 '{cname}': GT存在={gt_has_mask}, Pred存在={pred_has_mask}")
                    
                    # 仅当GT或Pred存在时才可视化
                    if gt_has_mask or pred_has_mask:
                        base, ext = os.path.splitext(original_slice_filename)
                        # 处理类别名称中的特殊字符
                        cname_safe = re.sub(r'[^\w]', '_', cname)  # 替换非字母数字字符为下划线
                        viz_filename = f"{base}_{cname_safe}{ext}"
                        viz_path = os.path.join(save_dir, viz_filename)
                        visualize_results(slice_img, gt_mask_cls, pred_mask_cls, viz_path)

    df = pd.DataFrame(results_summary)
    results_csv = os.path.join(args.pred_save_dir, 'results_summary.csv')
    df.to_csv(results_csv, index=False)
    print(f"\n所有样本测试完成，结果已保存至 {args.pred_save_dir}")
    print(f"整体平均 Dice: {df['dice_score'].mean():.4f}, 总类别数: {df['num_classes'].sum()}")
    print(f"详细结果保存至: {results_csv}")


if __name__ == "__main__":
    main()