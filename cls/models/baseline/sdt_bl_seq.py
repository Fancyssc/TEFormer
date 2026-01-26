from timm.models import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg
from ..utils.node import *
import torch.nn as nn

class sequential_embed(BaseModule):
    def __init__(self, step=4, encode_type='direct', sequence_length=1024, in_channels=3,
                 embed_dims=384, layer_by_layer=True, **kwargs):

        super(sequential_embed, self).__init__(step=step, encode_type=encode_type, layer_by_layer=layer_by_layer)

        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.embed_dims = embed_dims
        self.out_length = 64

        if sequence_length >= 1024:
            downsample_rates = [2, 2, 2, 2]  # 这些是步长，不是指数
        else:
            downsample_rates = [1, 2, 2, 2]
        # 按照要求设置每层的通道数: embed_dim//8, embed_dim//4, embed_dim//2, embed_dim
        channels = [
            in_channels,
            embed_dims // 8,
            embed_dims // 4,
            embed_dims // 2,
            embed_dims
        ]

        self.layers = nn.ModuleList()

        current_length = sequence_length

        for i in range(4):
            stride = downsample_rates[i]
            kernel_size = 5
            target_output_size = current_length // stride
            padding = self._calculate_padding(current_length, kernel_size, stride, target_output_size)

            if i != 3:
                layer = nn.Sequential(
                    nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size,
                              stride=stride, padding=padding),
                    nn.BatchNorm1d(channels[i + 1]),
                    LIFNode(step=self.step, tau=2.0, threshold=1.0, mem_detach=False)
                )

            else:
                layer = nn.Sequential(
                    nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel_size,
                              stride=stride, padding=padding),
                    nn.BatchNorm1d(channels[i + 1]),
                )

            self.layers.append(layer)
            current_length = target_output_size

        self.fine_tuning_layer = nn.AdaptiveAvgPool1d(self.out_length)

        self.proj_lif3 = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)
        # 添加可学习的相对位置编码
        self.pe_conv = nn.Conv1d(
            embed_dims,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=True
        )
        self.pe_bn = nn.BatchNorm1d(embed_dims)

    def _calculate_padding(self, input_size, kernel_size, stride, target_size):
        padding = ((target_size - 1) * stride + kernel_size - input_size) // 2
        return max(0, int(padding))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.fine_tuning_layer(x)

        x_feat = x
        x = self.proj_lif3(x)

        x = self.pe_conv(x)
        x = self.pe_bn(x)

        x = x + x_feat # T B C N

        return x.reshape(*x.shape[:-1], int(self.out_length ** 0.5), int(self.out_length ** 0.5)).contiguous() # T B C H W

# SDSA
class SDSA(BaseModule):
    def __init__(self,embed_dim, step=4,num_heads=12,):
        super().__init__(encode_type='direct', step=step,layer_by_layer=True)
        self.num_heads = num_heads

        self.q_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(embed_dim)
        self.q_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.k_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(embed_dim)
        self.k_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.v_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(embed_dim)
        self.v_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)
        #special v_thres
        self.attn_lif = LIFNode(step=self.step, tau=2.0,threshold=0.5, mem_detach=False)

        self.talking_heads = nn.Conv1d(num_heads, num_heads, kernel_size=1, stride=1, bias=False)
        self.talking_heads_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.proj_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(embed_dim)

        self.shortcut_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape #TB dim H//4 W//4
        N = H * W

        identity = x

        #shortcut
        x = self.shortcut_lif(x).reshape(TB, C, H, W)

        x_for_qkv = x
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out).flatten(-2, -1) #TB C N

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out).flatten(-2, -1)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out)
        v_conv_out = self.v_lif(v_conv_out).flatten(-2, -1)

        q = (
            q_conv_out
            .transpose(-1, -2)
            .reshape(TB, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous())
        k = (k_conv_out
            .transpose(-1, -2)
            .reshape(TB, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous())
        v = (v_conv_out
            .transpose(-1, -2)
            .reshape(TB, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()) #TB H N C//H

        # attn
        kv = k.mul(v)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv) #TB H N C//H
        x = q.mul(kv)

        x = x.transpose(2, 3).reshape(TB, C, H, W).contiguous()
        x = self.proj_bn(self.proj_conv(x))

        x = x + identity
        return x

class MLP(BaseModule):
    def __init__(self, in_features, step=4,  mlp_ratio = 4.0, out_features=None,mlp_drop=0.,):
        super().__init__(step=step, encode_type='direct',layer_by_layer=True)

        self.in_features = in_features
        self.mlp_ratio = mlp_ratio
        self.out_features = out_features or in_features
        self.hidden_features = int(self.in_features * self.mlp_ratio)

        self.fc_conv1 = nn.Conv2d(in_features, self.hidden_features, kernel_size=1, stride=1)
        self.fc_bn1 = nn.BatchNorm2d(self.hidden_features)
        self.fc_lif1 = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.fc_conv2 = nn.Conv2d(self.hidden_features, self.out_features, kernel_size=1, stride=1)
        self.fc_bn2 = nn.BatchNorm2d(self.out_features)
        self.fc_lif2 = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

    def forward(self, x):
        self.reset()

        identity = x

        x = self.fc_lif1(x)
        x = self.fc_conv1(x)
        x = self.fc_bn1(x)

        x = self.fc_lif2(x)
        x = self.fc_conv2(x)
        x = self.fc_bn2(x)

        return x+identity

# Spikformer block
class SDT_Block_s(nn.Module):
    def __init__(self, embed_dim=384, num_heads=12, step=4, mlp_ratio=4.,mlp_drop=0., ):
        super().__init__()

        self.attn = SDSA(
                embed_dim, step=step, num_heads=num_heads, )
        # self.layernorm1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(step=step,in_features=embed_dim,mlp_ratio=mlp_ratio,out_features=embed_dim,mlp_drop=mlp_drop)
        # self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x



class SDTV1(BaseModule):
    def __init__(self, step=4,img_size=32, patch_size=4, in_channels=3, num_classes=10,embed_dim=384,
                 num_heads=12, mlp_ratio=4,mlp_drop=0., depths=4):
        super().__init__(step=step, encode_type='direct',layer_by_layer=True)
        self.step = step  # time step
        self.num_classes = num_classes
        self.depths = depths
        self.head_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)


        patch_embed = sequential_embed(img_h=img_size,
                          img_w=img_size,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dim)

        block = nn.ModuleList([SDT_Block_s(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio,
                                           mlp_drop=mlp_drop, )


                               for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)
        # classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x = patch_embed(x)
        for blk in block:
            x = blk(x) #TB C H W
        # dim adjustment
        _, C, H, W = x.shape
        x = x.flatten(-2,-1) # TB C N

        return x.mean(2).reshape(self.step, -1, C).contiguous() # T B C

    def forward(self, x):
        self.reset()

        if len(x.shape) == 4:
            x = self.encoder(x) # TB C H W

        # sequence datasets
        else:
            x = (x.unsqueeze(0)).repeat(self.step, 1, 1, 1).flatten(0, 1)
        x = self.forward_features(x) #T B C
        T, B, C = x.shape
        x = self.head_lif(x.flatten(0, 1)).reshape(T, B, C)
        x = self.head(x).mean(0) # TET = False
        return x

#### models for static datasets
@register_model
def TIMv2_bl_sdt_seq(pretrained=False,**kwargs):
    model = SDTV1(
        step=kwargs.get('step', 4),
        img_size=kwargs.get('img_size', 32),
        patch_size=kwargs.get('patch_size', 4),
        in_channels=kwargs.get('in_channels', 3),
        num_classes=kwargs.get('num_classes', 10),
        embed_dim=kwargs.get('embed_dim', 384),
        num_heads=kwargs.get('num_heads', 12),
        mlp_ratio=kwargs.get('mlp_ratio', 4),
        mlp_drop=kwargs.get('mlp_drop', 0.0),
        depths=kwargs.get('depths', 2),
    )
    model.default_cfg = _cfg()
    return model