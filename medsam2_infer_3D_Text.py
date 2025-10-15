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

# --- Setup and Initialization ---
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

# --- Helper Functions (Aligned with the first example) ---

def getLargestCC(segmentation):
    """Retains the largest connected component in the segmentation result."""
    labels = measure.label(segmentation)
    if labels.max() == 0:  # No foreground objects
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def dice_binary(pred, target, smooth=1.0):
    """Calculates the Dice score for a single binary class."""
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """Converts a 3D grayscale NumPy array to RGB and resizes it."""
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size), Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1)
        resized_array[i] = img_array
    return resized_array

def find_best_key_frame(class_gt_3D):
    """Finds the frame with the largest mask area for a given class."""
    if class_gt_3D.max() == 0:
        return class_gt_3D.shape[0] // 2
    areas = np.sum(class_gt_3D, axis=(1, 2))
    return np.argmax(areas)

def visualize_results(slice_img, gt_mask, pred_mask, save_path):
    """Displays GT and prediction overlays on the original image and saves to a single PNG."""
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

# --- Main Program ---

def main():
    parser = argparse.ArgumentParser("MedSAM2 Text-Prompted Binary Segmentation Evaluation Script")
    parser.add_argument('--data_dir', type=str,default='/staff/wangtiantong/SAM2_new/dataset/amos22/MRI', help="Root directory containing reorganized_test and reorganized_test_mask.")
    parser.add_argument('--pred_save_dir', type=str, default="./MedSAM2_text_results/AMOS-MRI-one_prompt", help="Directory to save segmentation results and visualizations.")
    parser.add_argument('--checkpoint', type=str, default="/staff/wangtiantong/MedSAM2/exp_log/all_dataset_20250703-1735/checkpoints/checkpoint_200.pt", help="Path to your model checkpoint.")
    parser.add_argument('--cfg', type=str, default="/configs/sam2.1_hiera_t512_text_predict.yaml", help="Path to the model config file.")
    parser.add_argument('--use_largest_cc', action='store_true', default=False, help="Apply largest connected component to the prediction for each class.")
    parser.add_argument('--three_prompt', action='store_true', default=False, help="Use text prompts on the first, last, and key frames for each class.")
    args = parser.parse_args()
    os.makedirs(args.pred_save_dir, exist_ok=True)

    print("Loading MedSAM2 model for text prompts...")
    from sam2.build_sam import build_sam2_video_predictor_text
    predictor = build_sam2_video_predictor_text(args.cfg, args.checkpoint)
    print("Model loaded successfully.")

    test_img_base = os.path.join(args.data_dir, "reorganized_test")
    test_mask_base = os.path.join(args.data_dir, "reorganized_test_mask")
    sample_ids = [d for d in os.listdir(test_img_base) if os.path.isdir(os.path.join(test_img_base, d))]

    results = []

    for sample_id in tqdm(sample_ids, desc="Testing 3D Samples"):
        sample_img_dir = os.path.join(test_img_base, sample_id)
        sample_mask_dir = os.path.join(test_mask_base, sample_id)
        prompt_path = os.path.join(sample_img_dir, "prompt.json")
        if not os.path.exists(prompt_path): continue

        with open(prompt_path) as f:
            prompt_data = json.load(f)
        class_mapping = prompt_data.get('all_obj', {})
        # Create mappings, ensuring names are lowercase and stripped for consistency
        name2id = {n.lower().strip(): i for n, i in class_mapping.items()}
        id2name = {i: n for n, i in class_mapping.items()}
        
        slices_info = sorted(prompt_data['slices'], key=lambda x: x['slice_number'])
        if not slices_info: continue

        num_slices = len(slices_info)
        first_img_path = os.path.join(sample_img_dir, slices_info[0]['slice_file'])
        first_img = np.array(Image.open(first_img_path).convert('L'))
        H, W = first_img.shape
        
        # --- GT and Image Loading ---
        # Create 4D array to store GT for all classes: [num_classes, num_slices, H, W]
        all_class_ids = sorted([cid for cid in id2name.keys() if cid > 0])
        num_classes = len(all_class_ids)
        gt_4D = np.zeros((num_classes, num_slices, H, W), dtype=np.uint8)
        
        img_3D = np.zeros((num_slices, H, W), dtype=np.uint8)
        for i, si in enumerate(slices_info):
            img_3D[i] = np.array(Image.open(os.path.join(sample_img_dir, si['slice_file'])).convert('L'))
        
        # Load and process masks for each class
        mask_files = os.listdir(sample_mask_dir)
        for i, si in enumerate(slices_info):
            base = os.path.splitext(si['slice_file'])[0]
            for mf in [f for f in mask_files if f.startswith(base)]:
                # Parse class name from filename
                cname = mf.split('_')[-1].replace('+', ' ').rsplit('.', 1)[0].lower().strip()
                cid = name2id.get(cname, 0)
                if cid == 0: continue
                
                # Find the index for this class in our sorted list
                try:
                    class_idx = all_class_ids.index(cid)
                    mask_path = os.path.join(sample_mask_dir, mf)
                    mask_img = np.array(Image.open(mask_path).convert('L'))
                    # Use logical_or to combine masks if a class appears in multiple files for one slice
                    gt_4D[class_idx, i] = np.logical_or(gt_4D[class_idx, i], (mask_img > 0)).astype(np.uint8)
                except ValueError:
                    # This class_id is not in our target list (e.g. background)
                    pass
        
        # Identify classes that actually exist in this sample's ground truth
        existing_indices = []
        for class_idx, cid in enumerate(all_class_ids):
            if np.any(gt_4D[class_idx] > 0):
                existing_indices.append((class_idx, cid))
        
        if not existing_indices: continue
        
        print(f"Sample {sample_id} - Classes: {[id2name[cid] for _, cid in existing_indices]}")
        
        # --- Preprocessing ---
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D, 512) / 255.0
        img_tensor = torch.from_numpy(img_resized).cuda()
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].cuda()
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].cuda()
        img_tensor = (img_tensor - mean) / std
        
        # --- Process each existing class ---
        for class_idx, cid in existing_indices:
            cname = id2name.get(cid, f"class_{cid}")
            class_gt = gt_4D[class_idx]  # Get the 3D GT for the current class
            pred_logits = np.full((num_slices, H, W), -100, dtype=np.float32)

            # Determine prompt frames
            kf = find_best_key_frame(class_gt)
            if args.three_prompt:
                nonzero_slices = np.where(np.any(class_gt > 0, axis=(1, 2)))[0]
                if len(nonzero_slices) > 0:
                    slice_indices = sorted(list(set([kf, nonzero_slices[0], nonzero_slices[-1]])))
                    print(f"  Class '{cname}' prompted at frames: {slice_indices}")
                else:
                    slice_indices = [kf]
                    print(f"  Class '{cname}' prompted at frame: {kf} (no non-zero slices found, using key frame)")
            else:
                slice_indices = [kf]
                print(f"  Class '{cname}' prompted at frame: {kf}")
            
            # --- Model Inference ---
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                # Forward Pass
                st = predictor.init_state(img_tensor, video_height=H, video_width=W)
                for idx in slice_indices:
                    # Provide text prompt for the class at the specified frames
                    predictor.add_new_text(st, frame_idx=idx, obj_id=1, text_prompt=cname)
                for idx, _, logit in predictor.propagate_in_video(st):
                    pred_logits[idx] = np.maximum(pred_logits[idx], logit[0, 0].cpu().numpy())
                predictor.reset_state(st)

                # Backward Pass
                st = predictor.init_state(img_tensor, video_height=H, video_width=W)
                for idx in slice_indices:
                    predictor.add_new_text(st, frame_idx=idx, obj_id=1, text_prompt=cname)
                for idx, _, logit in predictor.propagate_in_video(st, reverse=True):
                    pred_logits[idx] = np.maximum(pred_logits[idx], logit[0, 0].cpu().numpy())
                predictor.reset_state(st)
                
            # --- Post-processing and Evaluation ---
            pred_bin_3D = (pred_logits > 0).astype(np.uint8)
            if args.use_largest_cc:
                # Apply LCC slice by slice
                for i in range(num_slices):
                    if pred_bin_3D[i].max() > 0:
                        pred_bin_3D[i] = getLargestCC(pred_bin_3D[i])
            
            dice_score = dice_binary(pred_bin_3D, class_gt)
            results.append({'sample_id': sample_id, 'class': cname, 'dice': dice_score})
            print(f"    -> Class '{cname}' Dice: {dice_score:.4f}")

            # --- Save Visualization ---
            save_dir = os.path.join(args.pred_save_dir, sample_id, re.sub(r'[^\w]', '_', cname))
            os.makedirs(save_dir, exist_ok=True)
            for i in range(num_slices):
                if np.any(class_gt[i]) or np.any(pred_bin_3D[i]):
                    base = os.path.splitext(slices_info[i]['slice_file'])[0]
                    viz_path = os.path.join(save_dir, f"{base}.png")
                    visualize_results(img_3D[i], class_gt[i], pred_bin_3D[i], viz_path)

    # --- Finalize and Save Results ---
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(args.pred_save_dir, 'text_prompt_results.csv')
        df.to_csv(csv_path, index=False)
        
        # Calculate and print overall average Dice
        overall_avg_dice = df['dice'].mean()
        print("\n" + "="*50)
        print(f"Evaluation complete. Results saved to {csv_path}")
        print(f"Overall Average Dice Score: {overall_avg_dice:.4f}")
        print("="*50)
    else:
        print("No samples were successfully tested. No results generated.")

if __name__ == "__main__":
    main()