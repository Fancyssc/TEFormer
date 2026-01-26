# from visualizer import get_local
from timm.models.vision_transformer import _cfg
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models import register_model
from ..utils.node import *


class MLP(BaseModule):
    def __init__(self,  in_features, step=4,hidden_features=None, out_features=None, drop=0.):
        super().__init__(step=step,encode_type='direct',layer_by_layer=True)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features
    def forward(self, x):
        self.reset()


        TB,C,W,H = x.shape
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).reshape(TB,self.c_hidden,W,H).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(TB,C,W,H).contiguous()
        x = self.fc2_lif(x)
        return x

class Token_QK_Attention(BaseModule):
    def __init__(self, dim, step=4, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__(step=step,encode_type='direct',layer_by_layer=True)
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.attn_lif = LIFNode(step=self.step, tau=2.0,threshold=0.5, mem_detach=False)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)


    def forward(self, x):
        self.reset()
        TB, C, H, W = x.shape
        x = x.flatten(-2, -1)
        TB, C, N = x.shape
        x_for_qkv = x

        q_conv_out = self.q_conv( x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out) #TB C N
        q = q_conv_out.reshape(TB, self.num_heads, C // self.num_heads, N) #TB H C' N

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(TB, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.reshape(TB, self.num_heads, C // self.num_heads, N)

        q = torch.sum(q, dim = 2, keepdim = True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)

        x = x.flatten(1, 2)
        x = self.proj_bn(self.proj_conv(x)).reshape(TB, C, H, W)
        x = self.proj_lif(x)

        return x


class Spiking_Self_Attention(BaseModule):
    def __init__(self, dim, step=4, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__(step=step,encode_type='direct',layer_by_layer=True)
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)
        self.attn_lif = LIFNode(step=self.step, tau=2.0,threshold=0.5, mem_detach=False)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape

        x = x.flatten(-2, -1)
        TB, C, N = x.shape
        x_for_qkv = x
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(TB,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(TB, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(TB,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(TB, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(TB,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(TB, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(2, 3).reshape(TB, C, N).contiguous()
        x = self.attn_lif(x)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(TB,C,W,H)

        return x

class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.attn = Spiking_Self_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class PatchEmbedInit(BaseModule):
    def __init__(self, step=4, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__(step=step,encode_type='direct',layer_by_layer=True)
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        # Downsampling + Res 0
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims)
        self.proj1_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.proj2_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims)
        self.proj2_lif =LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)


    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape
        # Downsampling + Res
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_maxpool(x).reshape(TB, -1, H//2, W//2).contiguous()
        x = self.proj_lif(x)

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.proj1_maxpool(x).reshape(TB, -1, H // 4, W // 4).contiguous()
        x = self.proj1_lif(x)

        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(TB, -1, H//4, W//4).contiguous()
        x = self.proj2_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(TB, -1, H//4, W//4).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x

class PatchEmbeddingStage(BaseModule):
    def __init__(self,step=4, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__(step=step,encode_type='direct',layer_by_layer=True)
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = LIFNode(step=self.step, tau=2.0,threshold=1.0, mem_detach=False)

    def forward(self, x):
        self.reset()

        TB, C, H, W = x.shape
        # Downsampling + Res
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.proj3_maxpool(x).reshape(TB, -1, H//2, W//2).contiguous()
        x = self.proj3_lif(x)

        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(TB, -1, H//2, W//2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(TB, -1, H//2, W//2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x

class hierarchical_spiking_transformer(BaseModule):
    def __init__(self,
                 step=4,
                 img_size_h=224, img_size_w=224, patch_size=16, in_channels=2, num_classes=1000,
                 embed_dims=512, num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=8, #sr_ratios=[8, 4, 2]
                 ):
        super().__init__(step=step,encode_type='direct',layer_by_layer=True)
        self.num_classes = num_classes
        self.depths = depths
        self.T = step
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims // 4)

        stage1 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 4, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer,)
            for j in range(1)])

        patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 2)


        stage2 = nn.ModuleList([TokenSpikingTransformer(
            dim=embed_dims // 2, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer,)
            for j in range(2)])


        patch_embed3 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims)

        stage3 = nn.ModuleList([SpikingTransformer(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer,)
            for j in range(depths - 3)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        # classification head 这里不需要脉冲，因为输入的是在T时长平均发射值
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed3, H, W):
        if H * W == self.patch_embed3.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed3.H, patch_embed3.W, -1).permute(0, 3, 1, 2),
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
        # TB C H W
        stage1 = getattr(self, f"stage1")
        stage2 = getattr(self, f"stage2")
        stage3 = getattr(self, f"stage3")
        patch_embed1 = getattr(self, f"patch_embed1")
        patch_embed2 = getattr(self, f"patch_embed2")
        patch_embed3 = getattr(self, f"patch_embed3")


        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)


        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        x = patch_embed3(x)
        for blk in stage3:
            x = blk(x)

        # TB C H W
        x =  x.flatten(-2, -1).mean(-1) # TB C
        _, C = x.shape
        return x.reshape(self.step, -1, C).contiguous()

    def forward(self, x):
        T = self.T
        x = self.encoder(x) # TB C H W
        x = self.forward_features(x)

        x = self.head(x.mean(0))
        return x



@register_model
def TIMv2_bl_qk_img(pretrained=False,**kwargs):
    model = hierarchical_spiking_transformer(
        step=kwargs.get('step', 4),
        img_size_h=kwargs.get('img_size', 224),
        img_size_w=kwargs.get('img_size', 224),
        patch_size=kwargs.get('patch_size', 16),
        in_channels=kwargs.get('in_channels', 3),
        num_classes=kwargs.get('num_classes', 1000),
        embed_dims=kwargs.get('embed_dim', 512),
        num_heads=kwargs.get('num_heads', 8),
        mlp_ratios=kwargs.get('mlp_ratio', 4),
        depths=kwargs.get('depths', 8),
        drop_path_rate=kwargs.get('drop_path', 0.2)
    )
    model.default_cfg = _cfg()
    return model

if __name__ == "__main__":
    import torch

    model = TIMv2_bl_qk_img()
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    print(f" Test success! Output shape: {output.shape}")

