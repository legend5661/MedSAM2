import torch
from torch import einsum, nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.distributed as dist

from einops import rearrange, repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out


class ImageAttentionPooler(nn.Module):
    def __init__(
        self,
        image_feature_dim: int,       # 输入图像特征的通道数 C
        output_embed_dim: int,        # 输出图像token的维度
        num_queries: int = 256,       # 除CLS Token外的查询数量
        dim_head: int = 64,
        heads: int = 8,
        # norm_context: bool = True # CoCa img_attn_pool 有这个参数
    ):
        super().__init__()
        self.num_queries = num_queries
        self.output_embed_dim = output_embed_dim
        self.image_feature_dim = image_feature_dim

        # 可学习的查询，第一个将作为CLS Token
        self.img_queries = nn.Parameter(torch.randn(num_queries + 1, output_embed_dim))

        # 交叉注意力层，查询关注图像特征
        # CoCa的CrossAttention定义: CrossAttention(dim, context_dim, dim_head, heads, norm_context)
        # 这里的 dim 是查询的维度 (output_embed_dim)
        # context_dim 是图像特征的维度 (image_feature_dim)
        self.attn_pool = CrossAttention(
            dim=output_embed_dim,
            context_dim=image_feature_dim,
            dim_head=dim_head,
            heads=heads,
            norm_context=True # 通常会对context进行归一化
        )
        self.norm = LayerNorm(output_embed_dim) # 对池化后的输出进行归一化

    def forward(self, image_features_bcwh: torch.Tensor):
        """
        Args:
            image_features_bcwh (torch.Tensor): 形状为 [B, C, H, W] 的图像特征

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - image_cls_embedding (torch.Tensor): 图像CLS Token嵌入, [B, output_embed_dim]
                - image_token_sequence (torch.Tensor): 图像Token序列 (不含CLS), [B, num_queries, output_embed_dim]
        """
        B, C, H, W = image_features_bcwh.shape
        assert C == self.image_feature_dim, \
            f"Input image feature dim {C} does not match expected {self.image_feature_dim}"

        # 1. 将图像特征展平并调整维度顺序以匹配CrossAttention的context输入
        # context 期望形状 [B, NumTokens, ContextDim]
        image_features_flattened = image_features_bcwh.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # 2. 复制 img_queries 到 batch size
        # queries 期望形状 [B, NumQueries, QueryDim]
        queries = self.img_queries.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries+1, output_embed_dim]

        # 3. 通过Attention Pooling
        pooled_output = self.attn_pool(queries, image_features_flattened) # [B, num_queries+1, output_embed_dim]
        pooled_output = self.norm(pooled_output)

        # 4. 分离CLS Token和其他Token
        image_cls_embedding = pooled_output[:, 0]                  # [B, output_embed_dim]
        image_token_sequence = pooled_output[:, 1:]                # [B, num_queries, output_embed_dim]

        return image_cls_embedding, image_token_sequence
    

class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)