# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_

from sam2.modeling.sam.language.build import register_model

from sam2.modeling.sam.language.LangEncoder import build_tokenizer, build_lang_encoder

from transformers import AutoTokenizer, AutoModel

import random

class LanguageEncoder(nn.Module):

    def __init__(
        self,
        tokenizer,
        tokenizer_type,
        lang_encoder,
        lang_projection,
        max_token_num,
        queue_operator,
    ):
        super().__init__()
        # seg
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.lang_encoder = lang_encoder
        self.lang_proj = lang_projection
        self.max_token_num = max_token_num
        self.logit_scale = nn.Parameter(torch.ones([]))
        
        # captioning & retrieval
        for key, value in queue_operator.items():
            self.register_buffer(key, value)
            
        self.biomed_encoder = AutoModel.from_pretrained("/staff/wangtiantong/MedSAM2/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

    @classmethod
    def from_config(cls, cfg):
        # build up text encoder for seg
        tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
        tokenizer_type = cfg['MODEL']['TEXT']['TOKENIZER']
        lang_encoder = build_lang_encoder(cfg['MODEL']['TEXT'], tokenizer, cfg['VERBOSE'])
        max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        
        dim_lang = cfg['MODEL']['TEXT']['WIDTH']
        dim_projection = cfg['MODEL']['DIM_PROJ']
        lang_projection = nn.Parameter(torch.empty(dim_lang, dim_projection))
        trunc_normal_(lang_projection, std=.02)

        # tested not working better      
        queue_operator = {}

        return {
            "tokenizer": tokenizer,
            "tokenizer_type": tokenizer_type,
            "lang_encoder": lang_encoder,
            "lang_projection": lang_projection,
            "max_token_num": max_token_num,
            "queue_operator": queue_operator,
        }

    def get_text_embeddings(self, class_names, name='default', is_eval=False, add_bgd=False, prompt=True, norm=True, store_buffer=None):
        if not is_eval:
            if prompt:
                # randomly sample one template
                arbitary_concepts = [
                    prompt_engineering(class_names[label].replace('-other','').replace('-merged','').replace('-stuff',''), topk=10000, suffix='.') \
                    for label in range(len(class_names))
                ]
                if add_bgd:
                    arbitary_concepts.append("A background in coco.")
            else:
                arbitary_concepts = class_names
            
            input_ids = []
            attention_masks = []
            for txt in arbitary_concepts:
                tokens = self.tokenizer(
                    txt, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
                tokens['input_ids'].squeeze_()
                tokens['attention_mask'].squeeze_()

                input_ids.append(tokens['input_ids'])
                attention_masks.append(tokens['attention_mask'])

            arbitary_tokens = torch.stack(input_ids)
            arbitary_attention_masks = torch.stack(attention_masks)

            text_emb = self.forward_language((arbitary_tokens.cuda(), arbitary_attention_masks.cuda()), norm=norm)
            setattr(self, '{}_text_embeddings'.format(name), text_emb)
        else:
            with torch.no_grad():
                def extract_mean_emb(txts):
                    tokens = self.tokenizer(
                        txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                    )
                    clss_embedding = self.forward_language((tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()), norm=norm)
                    clss_embedding = clss_embedding.mean(dim=0)
                    clss_embedding /= clss_embedding.norm()
                    return clss_embedding

                templates = get_prompt_templates()
                clss_embeddings = []
                if prompt:
                    for clss in class_names:
                        txts = [template.format(clss.replace('-other','').replace('-merged','').replace('-stuff','')) for template in templates]
                        clss_embeddings.append(extract_mean_emb(txts))
                else:
                    for clss in class_names:
                        clss_embeddings.append(extract_mean_emb([clss]))

                if add_bgd:
                    txts = ["A background in coco."]
                    clss_embeddings.append(extract_mean_emb(txts))

                text_emb = torch.stack(clss_embeddings, dim=0)
                setattr(self, '{}_text_embeddings'.format(name), text_emb)

    def reset_text_embeddings(self, name='default'):
        pass

    def get_text_token_embeddings(self, txts, name='default', token=False, norm=False):
        if not token:
            tokens = self.tokenizer(
                txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
            )
            tokens = {key: value.cuda() for key, value in tokens.items()}
        else:
            tokens = txts
        token_emb, class_emb = self.forward_language_token((tokens['input_ids'], tokens['attention_mask']), norm=norm)
        ret = {"tokens": tokens,
                "token_emb": token_emb,
                "class_emb": class_emb,}
        setattr(self, '{}_token_embeddings'.format(name), ret)
        return ret

    def forward_language(self, texts, norm=True):
        if self.tokenizer_type == 'biomed-clip':
            with torch.no_grad():  # Disable gradient calculation
                outputs = self.biomed_encoder(*texts)
            # Extract the last hidden state
            x = outputs['last_hidden_state']
            x = x[:, 0]  # Get the [CLS] token's embeddings for all examples
        else:
            x = self.lang_encoder(*texts)
            x = x['last_hidden_state']

            if self.tokenizer_type == 'clip':
                x = x[torch.arange(x.size(0)), texts[0].argmax(dim=-1)]
            else:
                x = x[:, 0]

        x = x @ self.lang_proj
        if norm:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-7)
        return x
    
    def forward_language_token(self, texts, norm=False):
        if self.tokenizer_type == 'biomed-clip':
            with torch.no_grad():  # Disable gradient calculation
                outputs = self.biomed_encoder(*texts)
            # Extract the last hidden state
            token_x = outputs['last_hidden_state']
            class_x = token_x[:, 0]  # Get the [CLS] token's embeddings for all examples
        else:
            x = self.lang_encoder(*texts)
            token_x = x['last_hidden_state']

            if self.tokenizer_type == 'clip':
                class_x = token_x[torch.arange(token_x.size(0)), texts[0].argmax(dim=-1)]
            else:
                class_x = token_x[:, 0]

        class_x = class_x @ self.lang_proj
        token_x = token_x @ self.lang_proj

        if norm:
            class_x = class_x / (class_x.norm(dim=-1, keepdim=True) + 1e-7)
            token_x = token_x / (token_x.norm(dim=-1, keepdim=True) + 1e-7)

        return token_x, class_x
    
    def compute_similarity(self, v_emb, name='default', fake=False):
        if fake:
            return None
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        t_emb = getattr(self, '{}_text_embeddings'.format(name))
        output = self.logit_scale.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2)
        return output


@register_model
def get_language_model(cfg, **kwargs):
    return LanguageEncoder(cfg)


import numpy as np


def get_prompt_templates():
    prompt_templates = [
        '{}.',
        'a photo of a {}.',
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]
    return prompt_templates

def prompt_engineering(classnames, topk=1, suffix='.'):
    prompt_templates = get_prompt_templates()
    temp_idx = np.random.randint(min(len(prompt_templates), topk))

    if isinstance(classnames, list):
        classname = random.choice(classnames)
    else:
        classname = classnames

    return prompt_templates[temp_idx].replace('.', suffix).format(classname.replace(',', '').replace('+', ' '))

def build_lang_encoder_sam2():
    """
    实例化LanguageEncoder，自定义配置参数
    仿照from_config实现，但直接返回实例化对象
    """
    # 创建配置字典
    cfg = {
        'MODEL': {
            'TEXT': {
                'ARCH': 'vlpencoder',
                'NAME': 'transformer',
                'TOKENIZER': 'clip',
                'CONTEXT_LENGTH': 77,
                'WIDTH': 512,
                'HEADS': 8,
                'LAYERS': 12,
                'AUTOGRESSIVE': True
            },
            'DIM_PROJ': 256
        },
        'VERBOSE': True
    }
    
    # 构建tokenizer
    tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
    tokenizer_type = cfg['MODEL']['TEXT']['TOKENIZER']
    
    # 构建语言编码器
    lang_encoder = build_lang_encoder(cfg['MODEL']['TEXT'], tokenizer, cfg['VERBOSE'])
    max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
    
    # 创建投影层
    dim_lang = cfg['MODEL']['TEXT']['WIDTH']
    dim_projection = cfg['MODEL']['DIM_PROJ']
    lang_projection = nn.Parameter(torch.empty(dim_lang, dim_projection))
    trunc_normal_(lang_projection, std=.02)
    
    # 可以根据需要自定义其他参数
    queue_operator = {}
    
    # 实例化并返回LanguageEncoder
    return LanguageEncoder(
        tokenizer=tokenizer,
        tokenizer_type=tokenizer_type,
        lang_encoder=lang_encoder,
        lang_projection=lang_projection,
        max_token_num=max_token_num,
        queue_operator=queue_operator
    )

if __name__ == '__main__':
    vlpencoder = build_lang_encoder_sam2()
    print(vlpencoder)