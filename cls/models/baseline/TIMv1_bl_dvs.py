
from braincog.model_zoo.base_module import BaseModule
from einops import rearrange
from networkx.classes.filters import hide_nodes
from timm.models import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg

from ..utils.node import LIFNode
from braincog.base.strategy.surrogate import *

import torch.nn as nn

class SPS(BaseModule):

    def __init__(self, step=4, img_h=32, img_w=32, patch_size=4, in_channels=3, embed_dims=384, **kwargs):
        super().__init__(step=step, encode_type='direct',layer_by_layer=True)

        self.step = step
        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        # 64x64 input no need for first 2 pooling
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape

        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x) # TB C H W
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.proj_lif1(x)
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x)
        x = self.proj_lif2(x)
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.proj_lif3(x)
        x = self.maxpool3(x)

        x_feat = x # TB, -1, H // 4, W // 4

        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.rpe_lif(x) # TB, -1, H // 4, W // 4

        x = x + x_feat

        x = x.flatten(-2)

        return x # TB, C, N


class TIMv1(BaseModule):
    def __init__(self, step = 4, in_channels=16, TIM_alpha=0.5):
        super().__init__(step=step, encode_type='direct')

        self.T = step
        #  channels may depends on the shape of input
        self.interactor = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1,
                                    padding=2, bias=True)

        self.in_lif = LIFNode(tau=2.0, v_threshold=0.5, layer_by_layer=False, step=1, mem_detach=False)  # spike-driven
        self.out_lif = LIFNode(tau=2.0, v_threshold=0.5, layer_by_layer=False, step=1, mem_detach=False)  # spike-driven

        self.tim_alpha = TIM_alpha

    # input [TB, H, N, C/H]
    def forward(self, x):
        self.reset()

        TB, H, N, CoH = x.shape
        B = TB // self.T
        x = rearrange(x, '(t b) h n c -> t b h n c', t=self.T)  # T B H N CoH

        output = []
        x_tim = torch.empty_like(x[0])


        for i in range(self.T):
            # 1st step
            if i == 0:
                x_tim = x[i]
                output.append(x_tim)
            # other steps
            else:
                x_tim = self.interactor(x_tim.flatten(0, 1)).reshape(B, H, N, CoH).contiguous()
                x_tim = self.in_lif(x_tim) * self.tim_alpha + x[i] * (1 - self.tim_alpha)
                x_tim = self.out_lif(x_tim)

                output.append(x_tim)

        output = torch.stack(output).flatten(0, 1)  # TB H N CoH

        return output  # TB H N CoH

class SSA(BaseModule):
    def __init__(self,embed_dim, step=4, num_heads=12,attn_scale=0.125):
        super().__init__(step=step, encode_type='direct', layer_by_layer=True)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = attn_scale
        self.step = step

        self.q_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.TIMv1 = TIMv1(step=self.step, in_channels=16)

        self.k_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.v_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        # low v_threshold of attn_lif
        self.attn_lif = LIFNode(step=self.step, tau=2.0,threshold=0.5, mem_detach=False)

        self.proj_linear = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1,bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

    def forward(self, x):
        self.reset()

        TB, C, N = x.shape
        x_for_qkv = x
        # qkv
        q_linear_out = self.q_linear(x_for_qkv)
        q_linear_out = self.q_bn(q_linear_out)
        # print(x.shape)
        q_linear_out = self.q_lif(q_linear_out).transpose(-2, -1) #TB N C
        q = q_linear_out.reshape(-1, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        q = self.TIMv1(q) # TIMv1

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out)
        k_linear_out = self.k_lif(k_linear_out).transpose(-2, -1) #TB N C
        k = k_linear_out.reshape(-1, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out)
        v_linear_out = self.v_lif(v_linear_out).transpose(-2, -1) #TB N C
        v = v_linear_out.reshape(-1, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(-2, -1).reshape(TB, C, N).contiguous()
        x = self.attn_lif(x) # TB, C, N
        x = self.proj_lif(self.proj_bn(self.proj_linear(x))) # TB C N

        return x # TB C N

class MLP(BaseModule):
    def __init__(self, in_features, step=4, mlp_ratio = 4.0, out_features=None,mlp_drop=0.):
        super().__init__(step=step, encode_type='direct', layer_by_layer=True)

        self.out_features = out_features or in_features
        self.hidden_features = int(in_features * mlp_ratio)
        self.mlp_drop = mlp_drop
        self.step = step

        self.id = nn.Identity()

        self.fc1_linear = nn.Conv1d(in_features, self.hidden_features, kernel_size=1, stride=1,bias=False)
        self.fc1_bn = nn.BatchNorm1d(self.hidden_features)
        self.fc1_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.fc2_linear = nn.Conv1d(self.hidden_features, out_features, kernel_size=1, stride=1,bias=False)
        self.fc2_bn = nn.BatchNorm1d(self.out_features)
        self.fc2_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)


    def forward(self, x):
        self.reset()
        # TB C N
        x = self.fc1_linear(x)
        x = self.fc1_bn(x)
        x = self.fc1_lif(x)

        x = self.fc2_linear(x)
        x = self.fc2_bn(x)
        x = self.fc2_lif(x)
        return x # TB C N


# Spikformer block
class Block(nn.Module):
    def __init__(self, embed_dim=384, num_heads=12, step=4,mlp_ratio=4. ,attn_scale=0.125,mlp_drop=0.):
        super().__init__()

        self.attn = SSA(embed_dim, step=step, num_heads=num_heads, attn_scale=attn_scale)

        self.mlp = MLP(step=step,in_features=embed_dim,mlp_ratio=mlp_ratio,out_features=embed_dim,mlp_drop=mlp_drop,)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Spikformer(BaseModule):
    def __init__(self,
                 step=4, img_size=32, patch_size=4, in_channels=3, num_classes=10,attn_scale=0.125,
                 embed_dim=384, num_heads=12, mlp_ratio=4, attn_drop=0.,depths=4,
                 **kwargs):
        super().__init__(step=step, encode_type='direct',layer_by_layer=True)
        self.T = step  # time step
        self.num_classes = num_classes
        self.depths = depths


        # for meta_transformer


        patch_embed = SPS(
                step=step,
                embed_dims=embed_dim,
                img_h=img_size,
                img_w=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
            )
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule


        block = nn.ModuleList([Block(
            step=step,
             embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_scale=attn_scale)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)
        for blk in block:
            x = blk(x) # TB N C
        # dim adjustment
        _, C, N = x.shape
        x = x.reshape(self.step, -1, C, N).contiguous()
        return x.mean(-1)

    def forward(self, x):
        # B T C H W
        x = x.transpose(0, 1).contiguous()  # T B C H W
        x = x.flatten(0, 1)   # TB C H W

        # print(x.shape)
        # print(self.depths)


        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


#### models for static datasets
@register_model
def TIMv2_bl_TIMv1_dvs(pretrained=False,**kwargs):
    model = Spikformer(
        step=kwargs.get('step', 4),
        img_size=kwargs.get('img_size', 32),
        patch_size=kwargs.get('patch_size', 4),
        in_channels=kwargs.get('in_channels', 3),
        num_classes=kwargs.get('num_classes', 10),
        embed_dim=kwargs.get('embed_dim', 384),
        num_heads=kwargs.get('num_heads', 12),
        mlp_ratio=kwargs.get('mlp_ratio', 4),
        attn_scale=kwargs.get('attn_scale', 0.125),
        attn_drop=kwargs.get('attn_drop', 0.0),
        depths=kwargs.get('depths', 4),
    )
    model.default_cfg = _cfg()
    return model

