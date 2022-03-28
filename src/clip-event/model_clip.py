from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



from model_ot import optimal_transport_dist
from utils_image import patch_from_norm_bbox

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, use_grid=False):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # print('attn_mask', self.attn_mask )
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # print('_x', x.size(), x) # 50, 2, 768
        # print('self.ln_1', self.ln_1.weight.size(), self.ln_1.weight)
        # print('self.ln_1(x)', self.ln_1(x))
        # print('self.attention(self.ln_1(x)', self.attention(self.ln_1(x)))
        # print()
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        # print('__x', x.size(), x)
        # print('========================')
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.patch_num = input_resolution // patch_size
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x, use_grid=False):
        # print('input_x', x.size(), x)   #[batch, 3, 224, 224]
        # print('self.conv1', self.conv1.weight)
        x = self.conv1(x)  # shape = [*, width, grid, grid]   #[batch, 768, 7, 7]
        # print('conv1', x.size(), x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]  #[batch, 768, 49]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]   #[batch, 49, 768]
        # print('permute,reshape', x.size(), x)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # print('cat', x.size(), x)  #[batch, 50, 768]
        x = x + self.positional_embedding.to(x.dtype)  #[batch, 50, 768]
        # print('positional_embedding', x.size(), x)
        x = self.ln_pre(x)  #[batch, 50, 768]
        # print('ln_pre', x.size(), x)

        x = x.permute(1, 0, 2)  # NLD -> LND  # [50, batch, 768]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD  # [batch, 50, 768]

        # how to use CLIP to get objects: https://github.com/openai/CLIP/issues/82
        # https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb#scrollTo=fWKGyu2YAeSV
        if use_grid:
            x = self.ln_post(x[:, :, :])  # [batch, 50, 768]
        else:
            x = self.ln_post(x[:, 0, :])  # [batch, 1, 768]

        # print(self.proj)
        if self.proj is not None:
            x = x @ self.proj
        # print('x', x)

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # constrastive
                 constrastive_overbatch=True,
                 # alignment
                 alignment=False,
                 # multi-level attention
                 multiattention=False
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width
        # print('embed_dim', embed_dim)
        # print('vision_width', vision_width)
        # print('vision_layers', vision_layers)
        # print('vision_patch_size', vision_patch_size)
        # print('image_resolution', image_resolution)

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        # contrastive
        self.constrastive_overbatch = constrastive_overbatch

        # OT alignment
        self.alignment = alignment

        # multi-level attention
        self.multiattention = multiattention
    
    def set_hyps(self, constrastive_overbatch=True, alignment=False, multiattention=False):
        self.constrastive_overbatch = constrastive_overbatch
        self.alignment = alignment
        self.multiattention = multiattention

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # text attention
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, use_grid=False):
        return self.visual(image.type(self.dtype), use_grid=use_grid)
        # if use_grid:
        #     return self.visual(image.type(self.dtype))
        # else:
        #     x = self.visual(image.type(self.dtype), use_grid=use_grid)
        #     return image_cls, grid_feature

    def encode_text(self, text):

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # print('xx', x)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print('xxx', x)
        x = self.transformer(x)
        # print('xxxx', x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # print('xxxxx', x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, train_arg=None, bboxs=None, bbox_desc_vec=None, bbox_label_vec=None):
        # print('text before', text.size(), text)
        # print('image before', image.size())
        # the bbox of SR are training bbox, so they can be aligned
        image_features = self.encode_image(image, use_grid=train_arg is not None)
        if train_arg is not None:
            grid_features = image_features[:, 1:, :].reshape(image_features.size(0), self.visual.patch_num, self.visual.patch_num, -1) # [batch, 49, d]
            image_features = image_features[:, 0, :] # image CLS
            # # using batch bbox
            # bbox_feature = grid_features[:, bboxs[:, 0]:bboxs[:, 2], bboxs[:, 1]:bboxs[:, 3], :]
            # for each image, no padding
            loss_per_bbox = 0 #list()
            loss_per_arg = 0 # list()
            for image_idx, bbox_image in enumerate(bboxs):
                bbox_feaures = None
                bbox_desc_features = None
                bbox_label_features = None
                for bbox_id, bbox in enumerate(bbox_image):
                    if bbox is not None:
                        bbox = patch_from_norm_bbox(bbox, patch_size=self.visual.patch_num)
                        bbox_feature = grid_features[image_idx, bbox[0]:bbox[2], bbox[1]:bbox[3], :]
                        # print('bbox_feaure', bbox_feature.size())
                        bbox_feaure = bbox_feature.reshape(-1, grid_features.size(-1)) #grid_features.size(0), -1, grid_features.size(-1))
                        bbox_feaure = torch.mean(bbox_feaure, dim=0) # or 1
                        bbox_feaures = torch.cat((bbox_feaures, bbox_feaure.unsqueeze(0)), dim=0) if bbox_feaures is not None else bbox_feaure.unsqueeze(0)
                        # print('bbox_feaure', bbox_feaure.size(), 'image_features', image_features.size())
                        # print('bbox_label ', bbox_desc_vec[image_idx][bbox_id].size())
                        bbox_desc_features = torch.cat((bbox_desc_features, bbox_desc_vec[image_idx][bbox_id].unsqueeze(0)), dim=0) if bbox_desc_features is not None else bbox_desc_vec[image_idx][bbox_id].unsqueeze(0)
                        if train_arg.startswith('desc_type'): #train_arg == 'desc_type' or train_arg == 'desc_type_text':
                            bbox_label_features = torch.cat((bbox_label_features, bbox_label_vec[image_idx][bbox_id].unsqueeze(0)), dim=0) if bbox_label_features is not None else bbox_label_vec[image_idx][bbox_id].unsqueeze(0)
            
                if bbox_desc_features is None:
                    # print('bbox_desc_features is None')
                    continue
                if bbox is None:
                    # print('bbox is None')
                    continue
                bbox_feaures = bbox_feaures / bbox_feaures.norm(dim=-1, keepdim=True)
                # desc
                bbox_desc_features = self.encode_text(bbox_desc_features)
                bbox_desc_features = bbox_desc_features / bbox_desc_features.norm(dim=-1, keepdim=True)
                # cosine similarity as logits
                logit_scale = self.logit_scale.exp()
                logits_per_bbox = logit_scale * bbox_feaures @ bbox_desc_features.t()
                logits_per_arg = logit_scale * bbox_desc_features @ bbox_feaures.t()
                # constrastive loss
                labels = torch.arange(logits_per_bbox.shape[0], device=logits_per_bbox.device)
                loss_i = self.loss_func(logits_per_bbox, labels)
                loss_t = self.loss_func(logits_per_arg, labels)
                loss_per_bbox += loss_i
                loss_per_arg += loss_t
                if train_arg.startswith('desc_type'):
                    # type
                    bbox_label_features = self.encode_text(bbox_label_features)
                    bbox_label_features = bbox_label_features / bbox_label_features.norm(dim=-1, keepdim=True)
                    # cosine similarity as logits
                    logit_scale = self.logit_scale.exp()
                    logits_per_bbox_ = logit_scale * bbox_feaures @ bbox_label_features.t()
                    logits_per_arg_ = logit_scale * bbox_label_features @ bbox_feaures.t()
                    # constrastive loss
                    labels_ = torch.arange(logits_per_bbox_.shape[0], device=logits_per_bbox_.device)
                    loss_i_ = self.loss_func(logits_per_bbox_, labels_)
                    loss_t_ = self.loss_func(logits_per_arg_, labels_)
                    loss_per_bbox += loss_i_
                    loss_per_arg += loss_t_
                if train_arg.startswith('desc_type_text'):
                    logits_per_role = logit_scale * bbox_desc_features @ bbox_label_features.t()
                    labels__ = torch.arange(logits_per_role.shape[0], device=logits_per_role.device)
                    loss_i__ = self.loss_func(logits_per_role, labels__)
                    loss_per_arg += loss_i__


        text_features = self.encode_text(text)
        # print('image_features encoded', image_features.size(), image_features)
        # print('text_features encoded', text_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # print('image_features normalized', image_features)
        # print('text_features normalized', text_features)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # logits text is always over batch, not over instance
        logits_per_text = logit_scale * text_features @ image_features.t()
        # logtis image can be over batch, or over instance
        if self.constrastive_overbatch: 
            # use `mm()`
            logits_per_image = logit_scale * image_features @ text_features.t()
        else:
            # resize image_features to use `bmm()`
            image_features = image_features.unsqueeze(1)
            batch_size = image_features.size(0)
            embed_dim = text_features.size(-1)
            text_features = text_features.view(batch_size, -1, embed_dim) # [batch, step, textfeature]
            # print('image_features_resize', image_features.size()) # torch.Size([4, 1, 512])
            # print('text_features_resize', text_features.size()) # torch.Size([4, 3, 512])
            # use `bmm()`
            logits_per_image = logit_scale * torch.bmm(image_features, torch.transpose(text_features, -2, -1))
            # logits_per_text = logit_scale * torch.bmm(text_features, torch.transpose(image_features, -2, -1))
            logits_per_image = logits_per_image.squeeze(1)
            # print('logits_per_image', logits_per_image.size())

        # shape = [global_batch_size, global_batch_size]
	
        if train_arg is not None:
            return logits_per_image, logits_per_text, loss_per_bbox, loss_per_arg
        else:
            return logits_per_image, logits_per_text


    def sim_entity(self, img_obj, txt_ent):
        BATCH_SIZE = img_obj.size(0)
        NUM_IMG = img_obj.size(1)
        NUM_TXT = txt_ent.size(1)

        # print('img_obj', img_obj.size())  # torch.Size([32, 7, 3, 224, 224])
        # print('txt_ent', txt_ent.size(), txt_ent)  # torch.Size([32, 11, 77])

        # for batch_idx in range(BATCH_SIZE):
        image_features = self.encode_image(img_obj.view(BATCH_SIZE * NUM_IMG, img_obj.size(2), img_obj.size(3), img_obj.size(4)))  # torch.Size([224, 512])
        image_features = image_features.view(BATCH_SIZE, NUM_IMG, -1)  # torch.Size([32, 7, 512])
        text_features = self.encode_text(txt_ent.view(BATCH_SIZE * NUM_TXT, txt_ent.size(2)))  # torch.Size([352, 512])
        text_features = text_features.view(BATCH_SIZE, NUM_TXT, -1)  # torch.Size([32, 11, 512])
        
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # # sim needs softmax
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * torch.bmm(image_features, torch.transpose(text_features, -2, -1))  # torch.Size([32, 7, 11])
        # sim = F.softmax(logits, dim=0)
        return image_features, text_features

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # NOTE: this function is used to covert to fp16
    # convert_weights(model)  

    model.load_state_dict(state_dict)
    return model #model.eval()


class CriterionContrastive(nn.Module):
    def __init__(self, constrastive_loss):
        super(CriterionContrastive, self).__init__()
        self.loss_func_text = nn.CrossEntropyLoss()
        if constrastive_loss == 'ce':
            self.loss_func_image = nn.CrossEntropyLoss()
        elif constrastive_loss == 'bce':
            self.loss_func_image = nn.BCEWithLogitsLoss()
        elif constrastive_loss == 'kl':
            self.loss_func_image = nn.KLDivLoss()
        else:
            raise RuntimeError("Invalid constrastive_loss '{}'. ".format(constrastive_loss))

    def forward(self, logits_per_image, logits_per_text, labels_per_image=None,labels_per_text=None,
                index_pos=None, constrastive_overbatch=True):
        if labels_per_image is None:
            # use constrastive learning labels
            labels_per_image = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
        if labels_per_text is None:
            # use constrastive learning labels
            labels_per_text = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
        # print('labels_per_image', labels_per_image)  # tensor([0, 0, 0, 0], device='cuda:0') 
        # print('logits_per_image', logits_per_image) # torch.Size([4, 3])
        # print('labels_per_text', labels_per_text) # tensor([0, 1, 2, 3], device='cuda:0')
        # print('logits_per_text', logits_per_text.size()) # torch.Size([12, 4])

        if constrastive_overbatch:
            # constrastive loss over batch
            loss_i = self.loss_func_image(logits_per_image, labels_per_image)
        else:
            # constrastive loss over instance
            loss_i = self.loss_func_image(logits_per_image, labels_per_image)
        # print('logits_per_text', logits_per_text)
        # print('labels_per_text', labels_per_text)
        # logits_per_text = logits_per_text[::step]
        logits_per_text = logits_per_text.index_select(index=index_pos, dim=0)
        labels_per_text = labels_per_text.index_select(index=index_pos, dim=0)
        # print('logits_per_text_pos', logits_per_text)
        # print('labels_per_text_pos', labels_per_text)
        loss_t = self.loss_func_text(logits_per_text, labels_per_text)

        # return (loss_i + loss_t) / 2.0
        return {'loss_i': loss_i, 'loss_t': loss_t}

class CriterionAlignment(nn.Module):
    def __init__(self):
        super(CriterionAlignment, self).__init__()
    
    # def len2pad(x_len):
    #     max_len = max(x_len)
    #     x_pad = torch.zeros(x_len.size(0), max_len).to(x_len.device)
    #     x_pad[]

    def mask2pad(self, x_mask):
        # switch 0 and 1
        x_pad = (x_mask == 0)#.long()
        return x_pad


    def forward(self, entitytxt_vec, object_vec, entitytxt_num, object_num):
        # cost,       txt_nodes_emb, img_nodes_emb, txt_pad, img_pad
        # [B, M, N],  [B, M, D],     [B, N, D],     [B, M],  [B, N] 


        # only for entity level
        txt_nodes_emb = entitytxt_vec
        img_nodes_emb = object_vec[:, 1:]
        # print('object_vec', object_vec.size(), 'img_nodes_emb', img_nodes_emb.size())
        txt_pad = self.mask2pad(entitytxt_num)
        # print('entitytxt_num', entitytxt_num, 'txt_pad', txt_pad)
        img_pad = self.mask2pad(object_num[:, 1:])
        # print('object_num', object_num, 'img_pad', img_pad)

        # cost = 1 - sim

        b = txt_nodes_emb.size(0)
        tl = txt_nodes_emb.size(1)
        il = img_nodes_emb.size(1)
        # NOTE: run in fp32 for stability
        ot_dist = optimal_transport_dist(#cost=cost, 
                                        txt_emb=txt_nodes_emb.float(), img_emb=img_nodes_emb.float(),
                                        txt_pad=txt_pad, img_pad=img_pad
                                        ).to(txt_nodes_emb)  # torch.Size([32])
        # print('ot_dist', ot_dist.size())

        # loss 1: within each instance: minimize the absolute OT distance between the postive <image-caption> pair
        # FIXIT: the ratio
        ot_loss = sum(ot_dist) * 0.01

        # loss 2: overbatch: enlarge the distance between negative and postive <image-caption> pairs
        # ot_pos_dist = ot_dist.masked_select(targets == 1)
        # ot_neg_dist = ot_dist.masked_select(targets == 0)
        # ot_loss = (ot_pos.sum() - ot_neg.sum()
        #                    ) / (ot_pos.size(0) + ot_neg.size(0))

        return {'loss_ot': ot_loss}