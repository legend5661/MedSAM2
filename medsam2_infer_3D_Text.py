from glob import glob
from tqdm import tqdm
import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
import SimpleITK as sitk
import torch
from sam2.build_sam import build_sam2_video_predictor_npz
from skimage import measure

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint path')
parser.add_argument('--cfg', type=str, required=True, help='model config')
parser.add_argument('-i', '--imgs_path', type=str, required=True, help='reorganized_train/val目录路径')
parser.add_argument('-o', '--pred_save_dir', type=str, default="./BioMed3D_results", help='预测结果保存路径')
parser.add_argument('--propagate_with_box', action='store_true', help='使用框传播模式')
parser.add_argument('--multi_mask', action='store_true', help='处理多类别mask')
parser.add_argument('--prompt_slice', type=int, default=0, help='使用第几个切片作为提示切片')

args = parser.parse_args()

def load_volume(folder_path):
    """从PNG切片加载3D体积数据"""
    slice_files = sorted(glob(os.path.join(folder_path, "*.png")),
                        key=lambda x: int(Path(x).stem.split('_')[-1]))
    volume = []
    for f in slice_files:
        img = np.array(Image.open(f).convert('L'))  # 转为灰度
        volume.append(img)
    return np.stack(volume, axis=0)

def get_prompts(video_folder):
    """从prompt.json加载提示信息"""
    prompt_path = os.path.join(video_folder, "prompt.json")
    with open(prompt_path) as f:
        prompts = json.load(f)
    
    # 解析所有对象的bbox信息
    bboxes = {}
    for obj in prompts['slices'][args.prompt_slice]['annotations']:
        obj_id = obj['obj_id']
        bbox = [obj['bbox']['xmin'], 
                obj['bbox']['ymin'],
                obj['bbox']['xmax'],
                obj['bbox']['ymax']]
        bboxes[obj_id] = np.array(bbox)
    return bboxes

def process_case(case_path, predictor):
    """处理单个3D样本"""
    # 加载数据
    img_volume = load_volume(case_path)  # (D, H, W)
    D, H, W = img_volume.shape
    
    # 预处理
    img_volume = (img_volume - img_volume.min()) / (img_volume.max() - img_volume.min()) * 255.0
    img_volume = img_volume.astype(np.uint8)
    
    # 转换为RGB并resize
    img_resized = np.zeros((D, 3, 512, 512), dtype=np.float32)
    for i in range(D):
        img_pil = Image.fromarray(img_volume[i]).resize((512, 512)).convert("RGB")
        img_resized[i] = np.array(img_pil).transpose(2, 0, 1) / 255.0
    
    # 标准化
    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].cuda()
    img_tensor = torch.from_numpy(img_resized).cuda()
    img_tensor = (img_tensor - img_mean) / img_std
    
    # 加载提示信息
    bboxes = get_prompts(case_path)
    
    # 初始化分割结果
    seg_volume = np.zeros_like(img_volume, dtype=np.uint8)
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # 处理每个对象
        for obj_id, bbox in bboxes.items():
            inference_state = predictor.init_state(img_tensor, H, W)
            
            # 添加初始框
            _, out_obj_ids, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=args.prompt_slice,
                obj_id=obj_id,
                box=bbox
            )
            
            # 双向传播
            for out_frame_idx, _, mask_logits in predictor.propagate_in_video(inference_state):
                mask = (mask_logits[0] > 0.0).cpu().numpy()[0]
                seg_volume[out_frame_idx][mask] = obj_id
                
            for out_frame_idx, _, mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                mask = (mask_logits[0] > 0.0).cpu().numpy()[0]
                seg_volume[out_frame_idx][mask] = obj_id
                
            predictor.reset_state(inference_state)
    
    return seg_volume

def main():
    # 初始化模型
    predictor = build_sam2_video_predictor_npz(args.cfg, args.checkpoint)
    
    # 创建输出目录
    os.makedirs(args.pred_save_dir, exist_ok=True)
    
    # 获取所有样本
    case_folders = sorted([d for d in os.scandir(args.imgs_path) if d.is_dir()])
    
    for case_folder in tqdm(case_folders, desc="Processing cases"):
        case_path = case_folder.path
        case_name = case_folder.name
        
        # 运行推理
        seg_result = process_case(case_path, predictor)
        
        # 后处理：保留最大连通域
        if not args.multi_mask:
            seg_result = (seg_result > 0).astype(np.uint8)
            seg_result = measure.label(seg_result)
            largest_cc = seg_result == np.argmax(np.bincount(seg_result.flat)[1:]) + 1
            seg_result = largest_cc.astype(np.uint8)
        
        # 保存结果
        sitk_img = sitk.GetImageFromArray(seg_result)
        output_path = os.path.join(args.pred_save_dir, f"{case_name}_pred.nii.gz")
        sitk.WriteImage(sitk_img, output_path)

if __name__ == "__main__":
    main()