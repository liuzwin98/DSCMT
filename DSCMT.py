import resnet
import torch.nn as nn
from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal_ as normal
from torch.nn.init import constant_ as constant

from typing import Optional
from torch import Tensor
import random


class BEF(nn.Module):
    def __init__(self, channel, reduction=8):
        super(BEF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputt):
        x = inputt.permute(0, 2, 1).contiguous()
        b, c, f = x.size()
        gap = self.avg_pool(x).view(b, c)
        y = self.fc(gap).view(b, c, 1)
        out = x * y.expand_as(x)

        return out.permute(0, 2, 1).contiguous()


class SA(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(SA, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # MLP, used for FFN
        # self.activation = nn.ReLU(inplace=True)
        # self.linear_in = nn.Linear(d_model, dim_feedforward)
        # self.dropout_mlp = nn.Dropout(dropout)
        # self.linear_out = nn.Linear(dim_feedforward, d_model)
        # self.drop2 = nn.Dropout(dropout)
        # self.norm2 = nn.LayerNorm(d_model)

        self.se = BEF(channel=d_model, reduction=8)

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                val=None):

        src_self = self.self_attention(src, src, value=val if val is not None else src,
                                       attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = src + self.drop1(src_self)
        src = self.norm1(src)
        # tmp = self.linear_out(self.dropout_mlp(self.activation(self.linear_in(src))))  # FFN

        tmp = self.se(src)
        src = self.norm2(src + self.drop2(tmp))

        return src


class CA(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CA, self).__init__()
        self.crs_attention1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.crs_attention2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP, used for FF
        # self.activation = nn.ReLU(inplace=True)
        # self.linear_in_1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout_mlp_1 = nn.Dropout(dropout)
        # self.linear_out_1 = nn.Linear(dim_feedforward, d_model)
        # self.drop_1 = nn.Dropout(dropout)
        # self.norm_1 = nn.LayerNorm(d_model)

        # self.linear_in_2 = nn.Linear(d_model, dim_feedforward)
        # self.dropout_mlp_2 = nn.Dropout(dropout)
        # self.linear_out_2 = nn.Linear(dim_feedforward, d_model)
        # self.drop_2 = nn.Dropout(dropout)
        # self.norm_2 = nn.LayerNorm(d_model)

        self.se1 = BEF(channel=d_model, reduction=8)    # 替换mlp可行
        self.se2 = BEF(channel=d_model, reduction=8)

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                ):

        src1_cross = self.crs_attention1(query=src1,
                                         key=src2,
                                         value=src2, attn_mask=src2_mask,
                                         key_padding_mask=src2_key_padding_mask)[0]

        src2_cross = self.crs_attention2(query=src2,
                                         key=src1,
                                         value=src1, attn_mask=src1_mask,
                                         key_padding_mask=src1_key_padding_mask)[0]

        src1 = src1 + self.drop1(src1_cross)
        src1 = self.norm1(src1)
        # tmp = self.linear_out_1(self.dropout_mlp_1(self.activation(self.linear_in_1(src1))))  # FFN

        tmp = self.se1(src1)
        src1 = self.norm_1(src1 + self.drop_1(tmp))

        src2 = src2 + self.drop2(src2_cross)
        src2 = self.norm2(src2)
        # tmp = self.linear_out_2(self.dropout_mlp_2(self.activation(self.linear_in_2(src2))))  # FFN

        tmp = self.se2(src2)
        src2 = self.norm_2(src2 + self.drop_2(tmp))

        return src1, src2


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        # x = tensor_list.tensors
        nt, f, c = tensor_list.size()
        tensor_list = tensor_list.permute(0, 2, 1).contiguous().view(nt, c, int(f ** 0.5), int(f ** 0.5))
        mask = (tensor_list[:, 0, :, :] != 0).int()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 沿y方向累加，(1，1，1)--(1，2，3)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 沿x方向累加，(1，1，1).T--(1，2，3).T
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # 第三个维度是num_pos_feats的2倍
        # return pos
        return pos.flatten(2).permute(0, 2, 1).contiguous()


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        nt, f, c = tensor_list.size()
        x = tensor_list.permute(0, 2, 1).contiguous().view(nt, c, int(f ** 0.5), int(f ** 0.5))
        h, w = x.shape[-2:]
        # x = tensor_list
        # h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos.flatten(2).permute(0, 2, 1).contiguous()
        # return pos


# -------------------------
# ------ DSCMT Model -------
# -------------------------
class FusionNet(nn.Module):
    def __init__(self, backbone_dim=2048, c_dim=512, num_c=60):
        super(FusionNet, self).__init__()

        self.c_dim = c_dim  # 降维后的通道数
        self.backbone_dim = backbone_dim
        self.droprate = 0.3  # transformer的droprate
        self.nheads = 8
        self.dim_feedforward = 2048  # transformer中MLP的隐层节点数
        self.layers = 4
        self.pos_rgb = PositionEmbeddingSine(c_dim // 2)
        self.pos_depth = PositionEmbeddingSine(c_dim // 2)

        self.reduce_channel1 = nn.Conv2d(self.backbone_dim, c_dim, kernel_size=1, bias=False)
        self.reduce_channel2 = nn.Conv2d(self.backbone_dim, c_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_dim)
        self.bn2 = nn.BatchNorm2d(c_dim)

        self.sa1 = SA(c_dim, self.nheads, self.dim_feedforward, self.droprate)
        self.sa2 = SA(c_dim, self.nheads, self.dim_feedforward, self.droprate)
        self.ca_list = nn.ModuleList([CA(c_dim, self.nheads, self.dim_feedforward, self.droprate)
                                      for _ in range(self.layers)])

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.1)

        self.fc_out1 = nn.Linear(c_dim, num_c)
        self.fc_out2 = nn.Linear(c_dim, num_c)
        self.fc_out3 = nn.Linear(c_dim, num_c)

        std = 0.001
        normal(self.fc_out1.weight, 0, std)
        constant(self.fc_out1.bias, 0)
        normal(self.fc_out2.weight, 0, std)
        constant(self.fc_out2.bias, 0)
        normal(self.fc_out3.weight, 0, std)
        constant(self.fc_out3.bias, 0)

    def forward(self, img_feat1, img_feat2):
        # 对channel做attention
        img_feat1 = self.reduce_channel1(img_feat1)  # con1x1 减少通道数
        img_feat2 = self.reduce_channel2(img_feat2)
        img_feat1 = self.bn1(img_feat1)
        img_feat2 = self.bn2(img_feat2)

        # (L, N, E),where L is the target sequence length, N is the batch size, E is the embedding dimension.
        img_feat1 = img_feat1.flatten(2).permute(0, 2, 1).contiguous()  # b f c
        img_feat2 = img_feat2.flatten(2).permute(0, 2, 1).contiguous()

        feat1 = self.pos_rgb(img_feat1)
        feat2 = self.pos_depth(img_feat2)

        for ca in self.ca_list:
            feat1 = self.sa1(feat1)
            feat2 = self.sa2(feat2)
            feat1, feat2 = ca(feat1, feat2)

        feat_fus = feat1 + feat2

        feat_fus = feat_fus.permute(0, 2, 1).contiguous()
        img_feat1 = img_feat1.permute(0, 2, 1).contiguous()  # b c f
        img_feat2 = img_feat2.permute(0, 2, 1).contiguous()

        feat_fus = self.avgpool(feat_fus).squeeze(2)
        img_feat1 = self.avgpool(img_feat1).squeeze(2)  # 比conv1D好
        img_feat2 = self.avgpool(img_feat2).squeeze(2)

        img_feat1 = self.drop1(img_feat1)
        img_feat2 = self.drop2(img_feat2)
        # feat_fus = self.drop3(feat_fus)

        img_feat1 = self.fc_out1(img_feat1)
        img_feat2 = self.fc_out2(img_feat2)
        feat_fus = self.fc_out3(feat_fus)

        return img_feat1, img_feat2, feat_fus


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.arch = base_model
        if not before_softmax and consensus_type != 'avg':  # consensus function的限制,只有avg才能与True配合使用
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 5 if modality in ['Flow', 'RGBDiff'] else 1
        else:
            self.new_length = new_length

        print(("""
        Initializing TSN with base model: {}.
        TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

        self.fusmodel = FusionNet(backbone_dim=2048, c_dim=512, num_c=num_class)
        # self.init_fusenet()

    def init_fusenet(self):
        # for p in self.fusmodel.modules():
        #     if isinstance(p, torch.nn.Linear):
        #         normal(p.weight, 0, 0.001)
        #         constant(p.bias, 0)
        for p in self.fusmodel.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _prepare_base_model(self, base_model):

        if base_model == 'vgg11':
            import vgg
            self.base_model = vgg.vgg11(pretrained=True)
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
            self.input_std = [0.229, 0.224, 0.225, 0.226, 0.226, 0.226]
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'resnet101':
            self.base_model = resnet.resnet101(pretrained=True)
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
            self.input_std = [0.229, 0.224, 0.225, 0.226, 0.226, 0.226]
        elif base_model == 'resnet50':
            self.base_model = resnet.resnet50(pretrained=True)
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5]
            self.input_std = [0.229, 0.224, 0.225, 0.226, 0.226, 0.226]
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()  # default:false
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and self.arch != 'vgg19':
            print("Freezing BatchNorm2D except the first one.")
            for (name, m) in self.base_model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1) and name != 'bn02':
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for (name, m) in self.base_model.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                # print('#'*50, '\n', m, '#'*50)
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1 or name == 'conv02':
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1 or name == 'bn02' or self.arch == 'vgg19':
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        # 新增fusion model的参数
        fusion_weight = []  # 一个多头attentio包括1个权重，1个bias，LayerNorm包括一个w一个b
        fusion_bais = []
        cls_weight = []
        cls_bais = []
        for (name, m) in self.fusmodel.named_modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())
                fusion_weight.append(ps[0])
                if len(ps) == 2:
                    fusion_bais.append(ps[1])

            elif isinstance(m, nn.MultiheadAttention):
                ps = list(m.parameters())
                fusion_weight.append(ps[0])  # 1，3是各自的mask
                fusion_bais.append(ps[2])

            elif isinstance(m, nn.LayerNorm):
                ps = list(m.parameters())
                fusion_weight.append(ps[0])
                fusion_bais.append(ps[1])

            elif isinstance(m, nn.Embedding):
                ps = list(m.parameters())
                fusion_weight.append(ps[0])

            elif isinstance(m, nn.Linear) and not (name.split('.')[-1] == 'out_proj'):
                ps = list(m.parameters())
                if "fc_out" in name:
                    cls_weight.append(ps[0])
                    if len(ps) == 2:
                        cls_bais.append(ps[1])
                else:
                    fusion_weight.append(ps[0])
                    if len(ps) == 2:
                        fusion_bais.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn.extend(list(m.parameters()))

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))

            elif isinstance(m, nn.Conv1d):
                ps = list(m.parameters())
                fusion_weight.append(ps[0])
                if len(ps) == 2:
                    fusion_bais.append(ps[1])

        # TODO 可修改fusionmodel学习率的倍率
        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': fusion_weight, 'lr_mult': 1, 'decay_mult': 1,  # 0.5
             'name': "fusion_weight"},
            {'params': fusion_bais, 'lr_mult': 2, 'decay_mult': 0,  # 1
             'name': "fusion_bais"},
            {'params': cls_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "cls_weight"},
            {'params': cls_bais, 'lr_mult': 10, 'decay_mult': 0,
             'name': "cls_bias"},
        ]

    def forward(self, input):
        sample_len = 6 * self.new_length  # 两张图片连接为6通道

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out_1, base_out_2 = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        base_out1, base_out2, base_out3 = self.fusmodel(base_out_1, base_out_2)  # 第一个参数是query

        if not self.before_softmax:
            base_out1 = self.softmax(base_out1)
            base_out2 = self.softmax(base_out2)
        if self.reshape:
            base_out1 = base_out1.view((-1, self.num_segments) + base_out1.size()[1:])  # b * 3 * class
            base_out2 = base_out2.view((-1, self.num_segments) + base_out2.size()[1:])
            base_out3 = base_out3.view((-1, self.num_segments) + base_out3.size()[1:])
        output1 = self.consensus(base_out1)  # b * 1 * 60
        output2 = self.consensus(base_out2)
        output3 = self.consensus(base_out3)

        return output1.squeeze(1), output2.squeeze(1), output3.squeeze(1)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'Appearance' or 'Motion':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
