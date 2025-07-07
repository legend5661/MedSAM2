

export PATH=/usr/local/cuda/bin:$PATH
timestamp=$(date +"%Y%m%d-%H%M")

python3 /staff/wangtiantong/MedSAM2/medsam2_infer_3D_Text.py \
        --checkpoint /staff/wangtiantong/MedSAM2/exp_log/text_prompt_single_gpu_4/checkpoint.pth \
        --cfg /staff/wangtiantong/MedSAM2/sam2/configs/sam2.1_text_infer.yaml \
        --imgs_path  /staff/wangtiantong/SAM2_new/dataset/ACDC \
        --pred_save_dir /staff/wangtiantong/MedSAM2/exp_log/text_prompt_single_gpu_4/inference \
        --propagate_with_box True \
        --multi_mask True \
        --prompt_slice 0 \