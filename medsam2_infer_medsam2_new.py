import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from skimage import measure
import pandas as pd
import matplotlib.pyplot as plt
import re
import cv2 # 为纠错点击功能添加的依赖

# --- 设置与初始化 ---
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

# --- 辅助函数 ---

def getLargestCC(segmentation):
    """在分割结果中保留最大的连通分量。"""
    labels = measure.label(segmentation)
    if labels.max() == 0:  # 没有前景对象
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC.astype(np.uint8)

def dice_binary(pred, target, smooth=1.0):
    """计算单个二元类别的Dice分数。"""
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """将3D灰度NumPy数组转换为RGB并调整其大小。"""
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
    """从2D掩码生成边界框。"""
    y_indices, x_indices = np.where(gt2D > 0)
    if len(y_indices) == 0:
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(W - 1, x_max + padding)
    y_max = min(H - 1, y_max + padding)
    if x_min >= x_max: x_max = x_min + 1
    if y_min >= y_max: y_max = y_min + 1
    return np.array([x_min, y_min, x_max, y_max])

def find_best_key_frame(class_gt_3D):
    """为给定类别找到掩码面积最大的帧。"""
    if class_gt_3D.max() == 0:
        return class_gt_3D.shape[0] // 2
    areas = np.sum(class_gt_3D, axis=(1, 2))
    return np.argmax(areas)

def visualize_results(slice_img, gt_mask, pred_mask, save_path):
    """显示GT和预测的叠加图并保存。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(slice_img, cmap='gray')
    if gt_mask.max() > 0:
        axes[0].imshow(gt_mask, alpha=0.5, cmap='jet')
    axes[0].set_title('GT Overlay')
    axes[0].axis('off')

    axes[1].imshow(slice_img, cmap='gray')
    if pred_mask.max() > 0:
        axes[1].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[1].set_title('Pred Overlay')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def get_single_correction_click(pred_mask, gt_mask):
    """
    根据预测/GT差异生成单个纠错点击。
    返回(point, label)，其中point是[x, y]，label是1(正向)或0(负向)。
    如果未找到错误，则返回(None, None)。
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    fn_map = np.logical_and(gt_mask, np.logical_not(pred_mask))
    fp_map = np.logical_and(pred_mask, np.logical_not(gt_mask))

    fn_dist_max, fp_dist_max = 0, 0
    fn_point, fp_point = None, None

    if np.any(fn_map):
        dist_transform = cv2.distanceTransform(fn_map.astype(np.uint8), cv2.DIST_L2, 5)
        fn_dist_max = np.max(dist_transform)
        y, x = np.unravel_index(np.argmax(dist_transform), fn_map.shape)
        fn_point = [x, y]

    if np.any(fp_map):
        dist_transform = cv2.distanceTransform(fp_map.astype(np.uint8), cv2.DIST_L2, 5)
        fp_dist_max = np.max(dist_transform)
        y, x = np.unravel_index(np.argmax(dist_transform), fp_map.shape)
        fp_point = [x, y]
    
    if fn_dist_max >= fp_dist_max and fn_point is not None:
        return fn_point, 1
    elif fp_dist_max > fn_dist_max and fp_point is not None:
        return fp_point, 0
    else:
        return None, None

# --- 主程序 ---

def main():
    parser = argparse.ArgumentParser("MedSAM2二元分割评估脚本（带迭代纠错功能）")
    parser.add_argument('--data_dir', type=str, default="/staff/wangtiantong/SAM2_new/dataset/amos22/CT_new")
    parser.add_argument('--pred_save_dir', type=str, default="./MedSAM2_binary_results/AMOS_CT_10_10")
    parser.add_argument('--checkpoint', type=str, default="/staff/wangtiantong/MedSAM2/checkpoints/MedSAM2_latest.pt")
    parser.add_argument('--cfg', type=str, default="/configs/sam2.1_hiera_t512.yaml")
    parser.add_argument('--use_largest_cc', action='store_true', default=False, help="对预测结果应用最大连通分量后处理。")
    parser.add_argument('--three_prompt', action='store_true', default=False, help="在第一帧、最后一帧和关键帧上使用Bbox提示。")
    parser.add_argument('--if_visualize', action='store_true', default=True)
    
    parser.add_argument('--correction_clicks', type=int, default=3, help="要应用的迭代纠错点击次数。0表示禁用。")
    
    args = parser.parse_args()
    os.makedirs(args.pred_save_dir, exist_ok=True)

    print("加载 MedSAM2 模型...")
    from sam2.build_sam import build_sam2_video_predictor_npz
    predictor = build_sam2_video_predictor_npz(args.cfg, args.checkpoint)
    print("模型加载成功。")

    test_img_base = os.path.join(args.data_dir, "reorganized_test")
    test_mask_base = os.path.join(args.data_dir, "reorganized_test_mask")
    sample_ids = [d for d in os.listdir(test_img_base) if os.path.isdir(os.path.join(test_img_base, d))]

    results = []

    for sample_id in tqdm(sample_ids, desc="测试3D样本"):
        # ... (数据加载部分与之前相同，此处省略以保持简洁) ...
        sample_img_dir = os.path.join(test_img_base, sample_id)
        sample_mask_dir = os.path.join(test_mask_base, sample_id)
        prompt_path = os.path.join(sample_img_dir, "prompt.json")
        if not os.path.exists(prompt_path): continue

        prompt_data = json.load(open(prompt_path))
        class_mapping = prompt_data.get('all_obj', {})
        name2id = {n.lower().strip(): i for n, i in class_mapping.items()}
        id2name = {i: n for n, i in class_mapping.items()}

        slices_info = sorted(prompt_data['slices'], key=lambda x: x['slice_number'])
        if not slices_info: continue

        num_slices = len(slices_info)
        first_img = np.array(Image.open(os.path.join(sample_img_dir, slices_info[0]['slice_file'])).convert('L'))
        H, W = first_img.shape
        
        all_class_ids = sorted(class_mapping.values())
        num_classes = len(all_class_ids)
        gt_4D = np.zeros((num_classes, num_slices, H, W), dtype=np.uint8)
        
        img_3D = np.zeros((num_slices, H, W), dtype=np.uint8)
        for i, si in enumerate(slices_info):
            img_3D[i] = np.array(Image.open(os.path.join(sample_img_dir, si['slice_file'])).convert('L'))
        
        mask_files = os.listdir(sample_mask_dir)
        for i, si in enumerate(slices_info):
            base = os.path.splitext(si['slice_file'])[0]
            for mf in [f for f in mask_files if f.startswith(base)]:
                cname = mf.split('_')[-1].replace('+', ' ').rsplit('.', 1)[0].lower().strip()
                cid = name2id.get(cname, 0)
                if cid == 0: continue
                
                class_idx = all_class_ids.index(cid)
                mask_path = os.path.join(sample_mask_dir, mf)
                mask_img = np.array(Image.open(mask_path).convert('L'))
                gt_4D[class_idx, i] = np.logical_or(gt_4D[class_idx, i], (mask_img > 0)).astype(np.uint8)
        
        existing_indices = []
        for class_idx, cid in enumerate(all_class_ids):
            if np.any(gt_4D[class_idx] > 0):
                existing_indices.append((class_idx, cid))
        
        if not existing_indices: continue
        
        print(f"样本 {sample_id} - 类别: {[id2name[cid] for _, cid in existing_indices]}")
        
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D, 512) / 255.0
        img_tensor = torch.from_numpy(img_resized).cuda()
        
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].cuda()
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].cuda()
        img_tensor = (img_tensor - mean) / std
        
        for class_idx, cid in existing_indices:
            cname = id2name.get(cid, f"class_{cid}")
            class_gt = gt_4D[class_idx]
            
            # --- [阶段 1] 使用Bbox提示进行初始预测 ---
            st = predictor.init_state(img_tensor, video_height=H, video_width=W)

            kf = find_best_key_frame(class_gt)
            slice_indices = [kf]
            if args.three_prompt:
                nonzero_slices = np.where(np.any(class_gt > 0, axis=(1, 2)))[0]
                if len(nonzero_slices) > 0:
                    slice_indices = sorted(list(set([kf, nonzero_slices[0], nonzero_slices[-1]])))
            
            print(f"  类别 '{cname}' 在帧 {slice_indices} 上进行提示")

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for idx in slice_indices:
                    if class_gt[idx].sum() == 0: continue
                    bbox = mask2D_to_bbox(class_gt[idx])
                    if bbox is not None:
                        predictor.add_new_points_or_box(st, frame_idx=idx, obj_id=1, box=bbox)
                
                current_pred_logits = np.full((num_slices, H, W), -100, dtype=np.float32)
                for idx, _, logit in predictor.propagate_in_video(st):
                    current_pred_logits[idx] = np.maximum(current_pred_logits[idx], logit[0, 0].cpu().numpy())
                for idx, _, logit in predictor.propagate_in_video(st, reverse=True):
                    current_pred_logits[idx] = np.maximum(current_pred_logits[idx], logit[0, 0].cpu().numpy())

                # --- [阶段 2] 迭代纠错 ---
                if args.correction_clicks > 0:
                    correction_frame_idx = -1
                    
                    # --- 新增逻辑：第一步 - 识别表现最差的帧 ---
                    initial_pred_bin = (current_pred_logits > 0).astype(np.uint8)
                    dice_per_slice = []
                    for frame_idx in range(num_slices):
                        # 只在有真实标签的帧上计算Dice以进行比较
                        if np.any(class_gt[frame_idx]):
                            slice_dice = dice_binary(initial_pred_bin[frame_idx], class_gt[frame_idx])
                            dice_per_slice.append((slice_dice, frame_idx))
                    
                    if not dice_per_slice:
                        print("    未找到带有真实标签的帧，无法进行纠错。")
                    else:
                        # 按Dice分数升序排序，找到分数最低的帧
                        dice_per_slice.sort(key=lambda x: x[0])
                        correction_frame_idx = dice_per_slice[0][1]
                        initial_worst_dice = dice_per_slice[0][0]
                        print(f"    已确定纠错目标帧: {correction_frame_idx} (初始Dice: {initial_worst_dice:.4f})")
                        print(f"    将对该帧进行 {args.correction_clicks} 次迭代纠错...")

                        # --- 修改后逻辑：第二步 - 在已确定的“最差帧”上集中进行纠错 ---
                        for i in range(args.correction_clicks):
                            # 只在目标帧上寻找错误点
                            pred_slice_bin = (current_pred_logits[correction_frame_idx] > 0).astype(np.uint8)
                            gt_slice = class_gt[correction_frame_idx]
                            
                            point, label = get_single_correction_click(pred_slice_bin, gt_slice)

                            if point is None:
                                print(f"      在目标帧 {correction_frame_idx} 上未找到更多可纠正的错误。提前停止。")
                                break

                            # 添加单个最佳点击并重新传播
                            click_type = "正向 (漏检)" if label == 1 else "负向 (误检)"
                            print(f"      第 {i+1}/{args.correction_clicks} 次纠错: 在帧 {correction_frame_idx} @ {point} 添加 {click_type} 点击")
                            
                            points_input = torch.tensor([[point]], dtype=torch.float, device=predictor.device)
                            labels_input = torch.tensor([[label]], dtype=torch.int32, device=predictor.device)
                            
                            predictor.add_new_points_or_box(st, frame_idx=correction_frame_idx, obj_id=1, points=points_input, labels=labels_input)
                            
                            # 用新的点击重新进行预测
                            temp_logits = np.full((num_slices, H, W), -100, dtype=np.float32)
                            for idx, _, logit in predictor.propagate_in_video(st):
                                temp_logits[idx] = np.maximum(temp_logits[idx], logit[0, 0].cpu().numpy())
                            for idx, _, logit in predictor.propagate_in_video(st, reverse=True):
                                temp_logits[idx] = np.maximum(temp_logits[idx], logit[0, 0].cpu().numpy())
                            current_pred_logits = temp_logits

            predictor.reset_state(st)

            # --- [阶段 3] 最终化与评估 ---
            pred_bin_3D = (current_pred_logits > 0).astype(np.uint8)
            if args.use_largest_cc:
                for i in range(num_slices):
                    if pred_bin_3D[i].max() > 0:
                        pred_bin_3D[i] = getLargestCC(pred_bin_3D[i])
            
            dice_score = dice_binary(pred_bin_3D, class_gt)
            results.append({'sample_id': sample_id, 'class': cname, 'dice': dice_score})
            print(f"      -> 最终类别 '{cname}' Dice: {dice_score:.4f}")

            if args.if_visualize or (dice_score < 0.5 and dice_score > 0):
                save_dir = os.path.join(args.pred_save_dir, sample_id, re.sub(r'[^\w]', '_', cname))
                os.makedirs(save_dir, exist_ok=True)
                for i in range(num_slices):
                    if np.any(class_gt[i]) or np.any(pred_bin_3D[i]):
                        base = os.path.splitext(slices_info[i]['slice_file'])[0]
                        viz_path = os.path.join(save_dir, f"{base}.png")
                        visualize_results(img_3D[i], class_gt[i], pred_bin_3D[i], viz_path)

    # --- 保存结果并计算总体平均Dice ---
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.pred_save_dir, 'binary_correction_results_focused.csv')
        df.to_csv(csv_path, index=False)
        
        overall_avg_dice = df['dice'].mean()
        
        print("\n" + "="*50)
        print(f"评估完成。结果保存在 {csv_path}")
        print(f"总体平均Dice分数: {overall_avg_dice:.4f}")
        print("="*50)
    else:
        print("\n没有成功测试的样本，未生成任何结果。")

if __name__ == "__main__":
    main()