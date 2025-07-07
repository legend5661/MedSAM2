# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F


from training.trainer import CORE_LOSS_KEY


from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


# 您的新对比损失函数
def contrastive_loss(text_latents: torch.Tensor,
                     image_latents: torch.Tensor,
                     temperature: float) -> torch.Tensor:
    """
    计算文本和图像潜变量之间的对称对比损失。
    text_latents: [N, D], image_latents: [N, D]
    temperature: 标量浮点数 (固定的超参数)
    """
    text_latents = F.normalize(text_latents, dim=-1)
    image_latents = F.normalize(image_latents, dim=-1)

    # 确保输入不为空，这可能导致 arange 或 cross_entropy 出错
    if text_latents.shape[0] == 0 or image_latents.shape[0] == 0:
        # 如果任一批次为空，则返回零损失。
        # 确保它在正确的设备上并具有正确的 dtype。
        # 假设 text_latents 或 image_latents (如果不为空) 可以提供 device/dtype。
        # 如果两者都为空，可能需要默认设备 (例如, 'cpu') 或由调用者处理。
        device = text_latents.device if text_latents.shape[0] > 0 else image_latents.device if image_latents.shape[0] > 0 else "cpu"
        dtype_ref = text_latents.dtype if text_latents.nelement() > 0 else image_latents.dtype if image_latents.nelement() > 0 else torch.float32
        return torch.tensor(0.0, device=device, dtype=dtype_ref)

    sim = torch.matmul(text_latents, image_latents.t())
    sim = sim * temperature # 根据您的定义。注意：CLIP 通常是除以 temperature。

    batch = sim.shape[0]
    labels = torch.arange(batch, device=sim.device)

    loss_t2i = F.cross_entropy(sim, labels)
    loss_i2t = F.cross_entropy(sim.t(), labels)
    return 0.5 * (loss_t2i + loss_i2t)


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    计算 DICE 损失, 类似于掩码的广义 IOU
    Args:
        inputs: 任意形状的浮点张量。
                每个样本的预测值。
        targets: 与 inputs 形状相同的浮点张量。存储 inputs 中每个元素的
                 二元分类标签 (负类为0，正类为1)。
        num_objects: 批次中的对象数量
        loss_on_multimask: 如果启用了多掩码预测，则为 True
    Returns:
        Dice 损失张量
    """
    inputs = inputs.sigmoid()
    # print("inputs:",inputs.shape) # 保留原始打印语句
    # print("targets:",targets.shape) # 保留原始打印语句
    # print("num_objects:",num_objects) # 保留原始打印语句
    if loss_on_multimask:
        # inputs 和 targets 是 [N, M, H, W]，其中 M 对应于多个预测掩码
        assert inputs.dim() == 4 and targets.dim() == 4
        # 展平空间维度，同时保留多掩码通道维度
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        targets = targets.flatten(1) # 假设 targets 在此也需要展平
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # print("loss_shape",loss.shape) # 保留原始打印语句
    # print("loss:",loss) # 保留原始打印语句
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    RetinaNet 中用于密集检测的损失函数: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: 任意形状的浮点张量。
                每个样本的预测值。
        targets: 与 inputs 形状相同的浮点张量。存储 inputs 中每个元素的
                 二元分类标签 (负类为0，正类为1)。
        num_objects: 批次中的对象数量
        alpha: (可选) 范围 (0,1) 内的加权因子，用于平衡
               正负样本。默认为 -1 (不加权)。
        gamma: 调制因子 (1 - p_t) 的指数，用于平衡
               难易样本。
        loss_on_multimask: 如果启用了多掩码预测，则为 True
    Returns:
        focal loss 张量
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss 是 [N, M, H, W]，其中 M 对应于多个预测掩码
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # 在空间维度上取平均
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: 任意形状的浮点张量。
                每个样本的预测值。
        targets: 与 inputs 形状相同的浮点张量。存储 inputs 中每个元素的
                 二元分类标签 (负类为0，正类为1)。
        pred_ious: 包含每个掩码预测的 IoU 分数的浮点张量
        num_objects: 批次中的对象数量
        loss_on_multimask: 如果启用了多掩码预测，则为 True
        use_l1_loss: 是否使用 L1 损失代替 MSE 损失
    Returns:
        IoU 损失张量
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    # 确保 inputs 是用于掩码计算的概率值，或者如果它们是 logits 则调整逻辑
    # 假设 inputs 是掩码的 logits, 在阈值化之前应用 sigmoid
    pred_mask = (inputs.sigmoid()).flatten(2) > 0.5 # sigmoid 后通常使用 0.5 作为阈值
    gt_mask = targets.flatten(2) > 0 # 假设 targets 已经是二元掩码 (0 或 1)

    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        contrastive_temperature: float = 1.0, # 为对比损失添加的温度系数
    ):
        """
        该类计算多步多掩码和 IoU 损失。
        Args:
            weight_dict: 包含 focal, dice, iou, contrastive 损失权重的字典
            focal_alpha: sigmoid focal loss 的 alpha 值
            focal_gamma: sigmoid focal loss 的 gamma 值
            supervise_all_iou: 如果为 True, 则对所有预测掩码反向传播 iou 损失
            iou_use_l1_loss: iou 使用 L1 损失代替 MSE 损失
            pred_obj_scores: 如果为 True, 计算对象分数的损失
            focal_gamma_obj_score: 对象分数的 sigmoid focal loss 的 gamma 值
            focal_alpha_obj_score: 对象分数的 sigmoid focal loss 的 alpha 值
            contrastive_temperature: 对比损失的温度系数
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict

        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        # 如果未指定，为对比损失添加默认权重
        if "loss_contrastive" not in self.weight_dict:
            self.weight_dict["loss_contrastive"] = 0.0 # 如果不激活，默认为 0

        self.contrastive_temperature = contrastive_temperature # 存储温度系数

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        """
        这里的 outs_batch 是一个列表, 列表长度为 num_frames (帧数)
        """
        assert len(outs_batch) == len(targets_batch)
        num_frames = len(outs_batch) # 获取批次中的帧数
        num_objects_tensor = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        ) 	# 批次内的对象数量是固定的
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects_tensor)
        num_objects = torch.clamp(num_objects_tensor / get_world_size(), min=1).item()

        losses = defaultdict(float) # 初始化为浮点型以累积浮点损失值
        
        # --- 新增逻辑: 为对比损失单独计数 ---
        num_contrastive_frames = 0

        for outs, targets in zip(outs_batch, targets_batch): # 遍历每一帧
            # --- 新增逻辑: 检查当前帧是否计算了对比损失 ---
            # 只有当 text 和 image 的 latents 都存在时，我们才认为这一帧对 contrastive loss 有贡献
            if outs.get("text_cls") is not None and outs.get("img_cls") is not None:
                num_contrastive_frames += 1

            cur_losses_dict = self._forward(outs, targets, num_objects) # 计算当前帧的损失
            for k, v in cur_losses_dict.items(): # 累加各损失分量
                if isinstance(v, torch.Tensor):
                    losses[k] += v
                elif k != CORE_LOSS_KEY :
                        losses[k] += v
        
        # --- 修改后的逻辑: 分别对不同类型的损失进行平均 ---
        for k, v in losses.items():
            if k == "loss_contrastive":
                # 对比损失只除以实际计算它的帧数
                # 添加一个安全检查以防 num_contrastive_frames 为 0
                if num_contrastive_frames > 0:
                    losses[k] = v / num_contrastive_frames
                else:
                    # 如果没有帧计算对比损失，则其损失应为0
                    losses[k] = v # v 初始为0.0，所以这没问题
            else:
                # 其他所有损失都除以总帧数
                if num_frames > 0:
                    losses[k] = v / num_frames

        # 如果 CORE_LOSS_KEY 是逐帧累加的，则需要基于总的累加和重新计算
        # 或者，更简单地说，在所有分量都累加完毕后，计算一次最终的 reduce后 的损失。
        if CORE_LOSS_KEY in losses: # 在全局 reduce 之前移除逐帧的 reduce后 的损失
            del losses[CORE_LOSS_KEY]
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses) # 计算最终的 reduce后 的损失

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        为单帧/单个输出计算与掩码相关的损失。
        `outputs` 包含用于对比损失的 "text_cls" 和 "img_cls"。
        """
        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4 	# [N, 1, H, W] N是该帧中的对象/样本数

        src_masks_list = outputs.get("multistep_pred_multimasks_high_res", [])
        ious_list = outputs.get("multistep_pred_ious", [])
        object_score_logits_list = outputs.get("multistep_object_score_logits", [])

        # 确保列表长度相同（如果不为空），否则跳过多步损失计算
        if not (len(src_masks_list) == len(ious_list) == len(object_score_logits_list)):
                # 如果对当前调用只有对比损失是重要的，处理这些列表可能为空的情况
                if not src_masks_list and not ious_list and not object_score_logits_list:
                    pass # 没有多步分量需要处理
                else:
                    raise ValueError("多步预测列表的长度不匹配。")

        # 初始化当前帧/输出的损失
        current_frame_losses = {
            "loss_mask": torch.tensor(0.0, device=target_masks.device, dtype=target_masks.dtype),
            "loss_dice": torch.tensor(0.0, device=target_masks.device, dtype=target_masks.dtype),
            "loss_iou": torch.tensor(0.0, device=target_masks.device, dtype=target_masks.dtype),
            "loss_class": torch.tensor(0.0, device=target_masks.device, dtype=target_masks.dtype)
        }

        # 如果对比损失的权重大于零，则计算对比损失
        if self.weight_dict.get("loss_contrastive", 0) > 0:
            text_latents = outputs.get("text_cls")
            image_latents = outputs.get("img_cls")

            if text_latents is not None and image_latents is not None:
                # 确保潜变量与其他张量在同一设备上（如果尚未在）
                text_latents = text_latents.to(target_masks.device)
                image_latents = image_latents.to(target_masks.device)

                if text_latents.shape[0] > 0 and image_latents.shape[0] > 0 :
                    loss_c = contrastive_loss(
                        text_latents, image_latents, self.contrastive_temperature
                    )
                    current_frame_losses["loss_contrastive"] = loss_c
                else:
                    # 如果潜变量为空，则赋零损失
                    current_frame_losses["loss_contrastive"] = torch.tensor(0.0, device=target_masks.device, dtype=target_masks.dtype)
            else:
                # 如果潜变量缺失，则赋零损失 (或者如果它们是强制性的，则作为错误处理)
                current_frame_losses["loss_contrastive"] = torch.tensor(0.0, device=target_masks.device, dtype=target_masks.dtype)
                if text_latents is None and image_latents is None:
                    pass # 两者都缺失，如果不使用则是预期行为
                else: # 其中一个缺失，可能是个问题
                    pass
                    # print(f"警告: 对于对比损失, text_latents 为 {'存在' if text_latents is not None else 'None'} "
                    #       f"且 image_latents 为 {'存在' if image_latents is not None else 'None'}。两者都是必需的。")
        else: # 如果权重为0，为了保持一致性（如果 reduce_loss 期望 weight_dict 中的所有键），仍然添加该键
            if "loss_contrastive" in self.weight_dict:
                current_frame_losses["loss_contrastive"] = torch.tensor(0.0, device=target_masks.device, dtype=target_masks.dtype)


        # 在预测步骤上累积掩码/dice/iou/类别损失
        # 仅当有步骤需要处理时
        if src_masks_list: # 检查列表是否不为空
            for src_masks, ious, object_score_logits in zip(
                src_masks_list, ious_list, object_score_logits_list
            ):
                self._update_losses( # _update_losses 将使用 `+=`
                    current_frame_losses, src_masks, target_masks, ious, num_objects, object_score_logits
                )

        # 计算当前帧损失的加权和 (此处可选, 也可以全局进行)
        # current_frame_losses[CORE_LOSS_KEY] = self.reduce_loss(current_frame_losses)
        # 当前结构是全局累加各个分量，然后 reduce。因此这行在这里不是严格必需的。

        return current_frame_losses # 返回当前帧的各个损失分量字典

    def _update_losses(
        self, losses_dict, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        # 此函数通过添加当前步骤的损失来原地修改 losses_dict

        # 确保 target_masks 被扩展以匹配 src_masks 以适应多掩码场景
        # src_masks 形状: [N_obj_in_frame, M_multimask, H, W]
        # target_masks 形状: [N_obj_in_frame, 1, H, W]
        expanded_target_masks = target_masks.expand_as(src_masks)

        loss_multimask = sigmoid_focal_loss(
            src_masks,
            expanded_target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, expanded_target_masks, num_objects, loss_on_multimask=True
        )

        if not self.pred_obj_scores:
            loss_class_step = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            # 对于 target_obj, 确保 N (loss_multimask 的批次维度) 是正确的。
            # loss_multimask 是 [N_obj_in_frame, M_multimask]
            target_obj = torch.ones(
                loss_multimask.shape[0], # N_obj_in_frame
                1, # 目标是每个对象，而不是多掩码中的每个掩码
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            # 目标对象存在性：如果GT掩码（第一个通道）中的任何像素 > 0，则为True
            # target_masks 是 [N_obj_in_frame, 1, H, W]
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None # 形状变为 [N_obj_in_frame, 1]
            ].float()
            # object_score_logits 形状应为 [N_obj_in_frame, 1] 或 [N_obj_in_frame, M_multimask]?
            # 此处假设为 [N_obj_in_frame, 1] 用于对象分数
            loss_class_step = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
                loss_on_multimask=False, # 假设 object_score_logits 在这里不是多掩码的
            )

        loss_multiiou = iou_loss(
            src_masks,
            expanded_target_masks,
            ious, # ious 形状应为 [N_obj_in_frame, M_multimask]
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )

        assert loss_multimask.dim() == 2 # [N_obj_in_frame, M_multimask]
        assert loss_multidice.dim() == 2 # [N_obj_in_frame, M_multimask]
        assert loss_multiiou.dim() == 2  # [N_obj_in_frame, M_multimask]

        if loss_multimask.size(1) > 1: # 如果是多掩码输出 (M_multimask > 1)
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)

            loss_mask_step = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice_step = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)

            if self.supervise_all_iou:
                loss_iou_step = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou_step = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else: # 单掩码输出 (M_multimask == 1)
            loss_mask_step = loss_multimask
            loss_dice_step = loss_multidice
            loss_iou_step = loss_multiiou

        # 仅当对象存在时才反向传播 focal, dice, 和 iou 损失 (target_obj 是 [N_obj_in_frame, 1])
        loss_mask_step = loss_mask_step * target_obj
        loss_dice_step = loss_dice_step * target_obj
        loss_iou_step = loss_iou_step * target_obj
        # loss_class_step 通过其定义已隐式地以 target_obj 为条件，或者应该如此

        # 对此步骤的批次维度求和 (损失已从其各自的函数中除以 num_objects)
        losses_dict["loss_mask"] += loss_mask_step.sum()
        losses_dict["loss_dice"] += loss_dice_step.sum()
        losses_dict["loss_iou"] += loss_iou_step.sum()
        losses_dict["loss_class"] += loss_class_step # loss_class_step 通常已经是该批次的标量或总和

    def reduce_loss(self, losses_dict):
        # 从损失张量获取设备；如果 losses_dict 为空，则默认为 "cpu"
        device_to_use = next(iter(losses_dict.values())).device if losses_dict and any(isinstance(v, torch.Tensor) for v in losses_dict.values()) else "cpu"
        reduced_loss = torch.tensor(0.0, device=device_to_use)

        for loss_key, weight in self.weight_dict.items():
            if loss_key in losses_dict and weight != 0:
                loss_value = losses_dict[loss_key]
                # 确保损失分量是一个张量
                if not isinstance(loss_value, torch.Tensor):
                    # 如果损失未计算并且仍然是浮点数（例如0.0），则可能发生这种情况
                    # 如果是简单的浮点数/整数，则转换为张量。
                    # 然而，此时所有损失都应该是张量。
                    # print(f"警告: 损失分量 {loss_key} 不是张量，正在转换。")
                    loss_value = torch.tensor(float(loss_value), device=reduced_loss.device)


                if isinstance(loss_value, torch.Tensor): # 再次检查以防万一
                     reduced_loss += loss_value * weight
                # 如果一个损失缺失但有权重，这意味着上游存在问题。
            elif loss_key not in losses_dict and weight != 0 :
                 print(f"警告: 损失键 {loss_key} 在损失字典中未找到，但其权重为 {weight}。")
                 # raise ValueError(f"{type(self)} 期望计算 {loss_key} 因为它具有非零权重。")

        return reduced_loss