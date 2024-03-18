import torch
import torch.nn as nn
import torch.nn.functional as F

from sige.nn import Gather, Scatter, SIGEConv2d, SIGEModel, SIGEModule

from net.layers.features import FeatureNet, ContextNet
from net.layers.deform_layers import DeformConv2dAdp, DeformConv2d
from net.layers.transformer import EncoderLayer as TransformerEncoderLayer

from math import pi

eps = 1e-12

autocast = torch.cuda.amp.autocast

def ray_pos_enc(feat, proj_inv, mu=60, num=7):
    '''
    Args:
        depth: Bx1xHxW
        R: Bx4x4
        K: Bx3x3
    Return:
        ray = (KR)^-1 (u v 1)^T'''
    b, _, h, w = feat.shape

    y, x = torch.meshgrid([torch.arange(0, h, dtype=torch.float32, device=feat.device),
                            torch.arange(0, w, dtype=torch.float32, device=feat.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(h * w), x.view(h * w)
    xyz = torch.stack((x, y, torch.ones_like(x)))   # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(b, 1, 1)   # [B, 3, H*W]
    ray = torch.matmul(proj_inv, xyz)

    coee = torch.arange(1, mu/2+1, step=(mu/2-1)/(num-1), device=feat.device) * pi
    coee = coee.unsqueeze(0).unsqueeze(-1).repeat(1,2,1)   # 1x20x1
    pos_x = ray[:,:1].repeat(1,num*2,1) * coee
    pos_y = ray[:,1:2].repeat(1,num*2,1) * coee
    pos_z = ray[:,2:3].repeat(1,num*2,1) * coee
    pos_enc = torch.cat((ray,
                         ray,
                         pos_x[:, :num, :].sin(),
                         pos_x[:, num:, :].cos(),
                         pos_y[:, :num, :].sin(),
                         pos_y[:, num:, :].cos(),
                         pos_z[:, :num, :].sin(),
                         pos_z[:, num:, :].cos()), dim=1)
    return pos_enc.view(b,num*6+6,h,w)


def cal_cam_proj(ref_intrinsics, src_intrinsics, ref_extrinsics, src_extrinsics):
    # cam proj inv at 1/8 resolution 
    level = 3
    ref_in_l3 = ref_intrinsics.clone() # Bx3x3
    ref_in_l3[:, :2, :3] /= (2**level)
    src_ins_l3 = src_intrinsics.clone() # BxNx3x3
    src_ins_l3[:, :, :2, :3] /= (2**level)
    ref_cam_l3 = torch.matmul(ref_in_l3, ref_extrinsics[:, :3, :3])
    src_cams_l3 = torch.matmul(src_ins_l3, src_extrinsics[:, :, :3, :3])
    cam_invs_l3 = torch.cat([ref_cam_l3.unsqueeze(1), src_cams_l3], dim=1).inverse()


    # cam to cam proj at 1/8 resolution
    ref_proj_l3 = ref_extrinsics.clone() # Bx4x4
    ref_proj_l3[:, :3, :4] = torch.matmul(ref_in_l3, ref_extrinsics[:, :3, :4])
    src_proj_l3 = src_extrinsics.clone() # BxNx4x4
    src_proj_l3[:, :, :3, :4] = torch.matmul(src_ins_l3, src_extrinsics[:, :, :3, :4])
    proj_l3 = torch.matmul(src_proj_l3, ref_proj_l3.inverse())

    # cam to cam proj at 1/4 resolution
    level = 2
    ref_in_l2 = ref_intrinsics.clone() # Bx3x3
    ref_in_l2[:, :2, :3] /= (2**level)
    src_in_l2 = src_intrinsics.clone() # BxNx3x3
    src_in_l2[:, :, :2, :3] /= (2**level)
    ref_proj_l2 = ref_extrinsics.clone() # Bx4x4
    ref_proj_l2[:, :3, :4] = torch.matmul(ref_in_l2, ref_extrinsics[:, :3, :4])
    src_proj_l2 = src_extrinsics.clone()
    src_proj_l2[:, :, :3, :4] = torch.matmul(src_in_l2, src_extrinsics[:, :, :3, :4])
    proj_l2 = torch.matmul(src_proj_l2, ref_proj_l2.inverse())

    return proj_l2, proj_l3, cam_invs_l3, ref_cam_l3.inverse()


def homo_warping(src_fea, height, width, proj, num_depth, depth_samps_flat):
    '''
    Args:
        src_fea: [B, C, H1, W1]
        proj: [B, 4, 4]
        depth_samps_flat: [B, n_depths, H2, W2]
    Return:
        warped_src_fea: [B, C, Ndepth, H2, W2]
    '''
    with torch.no_grad():
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        rot_xyz = torch.matmul(rot, xyz.view(1,3,-1))  # [B, 3, H*W]

        batch = depth_samps_flat.shape[0]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samps_flat.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(-1, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 1e-12)  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        homo_samp_coords = proj_xy.view(batch, num_depth * height, width, 2)

    warped_src_fea = F.grid_sample(src_fea,
                                   homo_samp_coords,
                                   mode='bilinear',
                                   padding_mode='zeros',
                                   align_corners=True)
    return warped_src_fea


class PixelViewWeight(nn.Module):
    def __init__(self, G):
        super(PixelViewWeight, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(G, 8, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(G, 1, 1, stride=1, padding=0),
        )

    def forward(self, x, training=False, need_view_weights=True):
        # x: [B, G, N, H, W]
        batch, dim, num_depth, height, width = x.shape
        x = x.permute(0,2,1,3,4).reshape(batch*num_depth, dim, height, width) # [B*N,G,H,W]
        corr = self.conv(x).view(batch, num_depth, height, width)

        if need_view_weights:
            x = torch.softmax(corr, dim=1)
            x = torch.amax(x, dim=1, keepdim=True)
        else:
            x = None
        return corr, x


class CorrBlock1D:
    def __init__(self, pixel_wise_net, pixel_wise_net_sp, ref_feats, multi_src_feats, depth_range, num_depth,
                 proj_l3, proj_l2,
                 num_levels=3, radius=4, G=8, ref_depth=None, training=True, identity=False,
                 depth_samps_flat=None, depth_samps=None, res_div=True):
        # homo_samp_coords: BxN
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.training = training
        self.G = G
        self.pixel_wise_net = pixel_wise_net
        self.num_depth = num_depth

        # Initialize 1/8 dense correlation volume with level 2 feature

        device = ref_feats['level2'].device
        ref_feat_l2 = ref_feats['level2']  # 1/4 res
        ref_feat_l2 = F.interpolate(ref_feat_l2, scale_factor=0.5, mode='bilinear') # 1/8 res
        multi_src_feat_l2 = multi_src_feats['level2']

        _, _, h8, w8 = ref_feat_l2.shape
        depth_samps_l3 = depth_samps_flat.repeat(1,1,h8,w8)
        corr_init, pair_view_weights = self.corr(pixel_wise_net, ref_feat_l2, multi_src_feat_l2, proj_l3, depth_samps_l3)
        self.corr_init = corr_init

        # 2 Build sparse correlation at 1/4 resolution

        device = ref_feats['level1'].device
        ref_feat_l1 = ref_feats['level1']  # 1/2 res
        ref_feat_l1 = F.interpolate(ref_feat_l1, scale_factor=0.5, mode='bilinear') # 1/4 res
        multi_src_feat_l1 = multi_src_feats['level1']
        batch, _, h4, w4 = ref_feat_l1.shape

        topk = 16
        _, corr_ind_l3 = torch.topk(corr_init, k=topk, dim=1) 
        corr_ind_l2 = F.interpolate(corr_ind_l3.float(), scale_factor=2, mode='nearest').long()
        depth_samps_topk = torch.gather(depth_samps_flat.repeat(1,1,h4,w4), 1, corr_ind_l2)

        corr_sp, pair_view_weights = self.corr(pixel_wise_net_sp, ref_feat_l1, multi_src_feat_l1, proj_l2, depth_samps_topk, 
                                               pair_view_weights=None)
        corr = torch.zeros([batch, num_depth, h4, w4], device=device)
        corr = corr.scatter(dim=1, index=corr_ind_l2, src=corr_sp)

        self.corr_full = corr
        self.corr_sp = corr_sp
        self.depth_samps_sp = depth_samps_topk
        self.corr_ind = corr_ind_l2
        self.batch_ind = torch.arange(batch, device=device).view(batch, 1, 1, 1).repeat(1, topk, h4, w4)

        corr = corr.permute(0, 2, 3, 1).reshape(batch*h4*w4, 1, 1, num_depth) # batchxh1xw1, dim, 1, num_depth
        self.corr_pyramid.append(corr)

        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

        # fast inference
        dx = torch.linspace(-radius, radius, 2*radius+1, device=device).view(1, 1, 2*radius+1, 1)
        self.dx = {'level{}'.format(level): dx / int(num_depth / 2**(level-1)) for level in (1,2,3)}
        self.grid_y = torch.zeros(batch*h4*w4, 1, 2*radius+1, 1, device=device)

    def corr(self, pixel_wise_net, ref_feat, multi_src_feat, proj, depth_samps, pair_view_weights=None):
        need_view_weights = pair_view_weights is None
        if self.training:
            pair_corr_sum, view_weight_sum = None, None
            if need_view_weights:
                pair_view_weights = []
            for idx in range(len(multi_src_feat)):
                pair_group_corr = self.pair_corr_coarse(ref_feat, multi_src_feat[idx], proj[:,idx], depth_samps)
                pair_corr, pair_view_weight = pixel_wise_net(pair_group_corr, self.training, need_view_weights)
                if need_view_weights:
                    pair_view_weights.append(pair_view_weight)
                else:
                    pair_view_weight = pair_view_weights[idx]
                if idx == 0:
                    pair_corr_sum = pair_corr * pair_view_weight
                    view_weight_sum = pair_view_weight
                else:
                    pair_corr_sum = pair_corr_sum + pair_corr * pair_view_weight
                    view_weight_sum = view_weight_sum + pair_view_weight
            corr = pair_corr_sum / view_weight_sum
            pair_view_weights = [F.interpolate(x, scale_factor=2, mode='bilinear') for x in pair_view_weights]
        else:
            n_srcs = len(multi_src_feat)
            group_corr = self.pair_corr_coarse(ref_feat.repeat(n_srcs, 1, 1, 1), torch.cat(multi_src_feat, dim=0).contiguous(),
                                          proj[0], depth_samps.repeat(n_srcs, 1, 1, 1))
            pair_corrs, pair_view_weights = pixel_wise_net(group_corr, self.training, need_view_weights)
            corr = (pair_corrs * pair_view_weights).sum(dim=0, keepdim=True) / pair_view_weights.sum(dim=0, keepdim=True)
        return corr, None

    def pair_corr_coarse(self, ref_feats, src_feat, proj_l3, depth_samps):
        '''先对特征降通道，然后再采样'''
        batch, dim, ht, wd = ref_feats.shape
        num_depth = depth_samps.shape[1]

        warped_volume = homo_warping(src_feat, ht, wd, proj_l3, num_depth, depth_samps)  # 30ms

        # group-wise correlation
        ref_feats = ref_feats.view(batch, self.G, dim // self.G, 1, ht, wd)                 # BxGxCx1xHxW
        warped_volume = warped_volume.view(batch, self.G, dim // self.G, num_depth, ht, wd) # BxGxCxDxHxW
        if self.training:
            group_corr = (warped_volume * ref_feats).mean(2)    # BxGxDxHxW
        else:
            warped_volume *= ref_feats
            group_corr = warped_volume.mean(2)
            del warped_volume, ref_feats, src_feat

        return group_corr

    def lookup(self, sigmoid_depth):
        '''Args:
        samples: NxDxHxW
        ds: downsample by scale 2'''
        r = self.radius
        batch, d, h1, w1 = sigmoid_depth.shape
        coords = (1 - sigmoid_depth).reshape(batch*h1*w1, 1, 1, 1)
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            n_samples = corr.shape[-1]
            x0 = self.dx['level{}'.format(i+1)] + coords
            grid = torch.cat([2 * x0 - 1, self.grid_y], dim=-1)
            corr = F.grid_sample(corr, grid, align_corners=True)     
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1) # (BxHxW)x1xD
        out = out.view(batch, h1, w1, -1).permute(0, 3, 1, 2)  # BxDxHxW
        return out.contiguous().float()


class ChannelSqueeze(nn.Module):
    def __init__(self, input_dim=96, output_dim=1):
        super(ChannelSqueeze, self).__init__()
        hidden_dim = input_dim // 2
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=2, dilation=2, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SDropout(nn.Module):
    def __init__(self, in_channels, thresh=3):
        '''Args:
        thresh: 3 (signal-to-noise 1/e^3=0.05), 4.605 (stn=0.01), 3.91 (stn=0.02)'''
        super(SDropout, self).__init__()
        self.thresh = thresh
        self.channel_squeeze = ChannelSqueeze(in_channels)
        self.test_sample_mode = 'no_greedy'

    def contextual_dropout(self, input):
        log_sigma2 = self.channel_squeeze(input)
        return log_sigma2

    def gen_log_alpha(self, batch_size, input_):
        batch, _, width, height = input_.shape
        if self.training:
            input_ = input_.detach()
        self.log_sigma2 = self.contextual_dropout(input_)
        log_alpha = self.log_sigma2 - 2 * torch.log(torch.abs(input_).mean(dim=1, keepdim=True) + eps)
        self.log_alpha = torch.clamp(log_alpha, -10, 10)

    def forward(self, input):
        log_alpha = self.gen_log_alpha(input.size(0), input)
        z = None
        if self.training:
            mu_post = torch.ones((batch_size, 1, width, height), device=input_.device)
            self.u = torch.empty((batch_size, 1, width, height), device=input_.device).normal_(0, 1)
            alpha_sqrt = torch.sqrt(torch.exp(self.log_alpha))

            # Reparameterization trick
            z = mu_post + alpha_sqrt * self.u * 0.1
            return z.view(batch_size, 1, width, height)
        return z

    @property
    def clip_mask(self):
        return torch.lt(self.log_alpha, self.thresh)

    def get_reg(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        log_alpha = self.log_alpha
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - \
            0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        return -torch.sum(mdkl)


class ConvGRU(SIGEModel): # nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3, mdconv=False, pac=False, deformable_groups=2,
                 sparse_dropout=False, sparse_infer=False):
        super(ConvGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.sparse_dropout = sparse_dropout

        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size-1, dilation=2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size-1, dilation=2)
        if self.sparse_dropout:
            self.sdrop = SDropout(in_channels=hidden_dim)
        self.sparse_infer = sparse_infer
        self.mdconv = mdconv
        if mdconv:
            self.convq = DeformConv2dAdp(hidden_dim+input_dim, hidden_dim, kernel_size, 
                                         dilation=2, deformable_groups=deformable_groups)
        else:
            self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size-1, dilation=2)

        if self.sparse_infer:
            BS = (8, 10)
            self.sige_convz = SIGEConv2d(in_channels=hidden_dim+input_dim, out_channels=hidden_dim, 
                                    kernel_size=kernel_size, padding=kernel_size-1, 
                                    dilation=2, bias=True)  
            self.sige_convz.weight = self.convz.weight
            self.sige_convz.bias = self.convz.bias
            self.gather_convz = Gather(self.sige_convz, block_size=BS, verbose=False)
            self.scatter_convz = Scatter(self.gather_convz)

            self.sige_convr = SIGEConv2d(in_channels=hidden_dim+input_dim, out_channels=hidden_dim, 
                                    kernel_size=kernel_size, padding=kernel_size-1, 
                                    dilation=2,  bias=True)
            self.sige_convr.weight = self.convr.weight
            self.sige_convr.bias = self.convr.bias
            self.gather_convr = Gather(self.sige_convr, block_size=BS)
            self.scatter_convr = Scatter(self.gather_convr)

            if not self.mdconv:
                self.sige_convq = SIGEConv2d(in_channels=hidden_dim+input_dim, out_channels=hidden_dim, 
                                        kernel_size=kernel_size, padding=kernel_size-1, 
                                        dilation=2,  bias=True)
                self.sige_convq.weight = self.convq.weight
                self.sige_convq.bias = self.convq.bias
                self.gather_convq = Gather(self.sige_convq, block_size=BS)
                self.scatter_convq = Scatter(self.gather_convq)

            self.prev_z = None
            self.prev_r = None
            self.prev_q = None

    def forward(self, h, motion_features, inp, offset_mask=None, prev_clip_mask=None, fast_infer=False):
        hx = torch.cat([h, motion_features, inp], dim=1)
        q_dense = None
        sparse_flag = fast_infer and prev_clip_mask is not None and (prev_clip_mask.sum() / prev_clip_mask.numel() < 0.7)
        
        if sparse_flag:
            if self.mode == 'full':
                self.set_mode('sparse')
           
            self.gather_convz.input_res = prev_clip_mask.shape[2:]
            self.scatter_convz.original_output = self.prev_z.float()
            self.gather_convr.input_res = prev_clip_mask.shape[2:]
            self.scatter_convr.original_output = self.prev_r.float()
            if not self.mdconv:
                self.gather_convq.input_res = prev_clip_mask.shape[2:]
                self.scatter_convq.original_output = self.prev_q

            self.set_masks({tuple(h.shape[-2:]): prev_clip_mask[0,0]})

            hx_gat = self.gather_convz(hx)
            z_gat = torch.sigmoid(self.sige_convz(hx_gat))
            z = self.scatter_convz(z_gat)

            hx_gat = self.gather_convr(hx)
            r_gat = torch.sigmoid(self.sige_convr(hx_gat))
            r = self.scatter_convr(r_gat)

            if self.mdconv:
                q = torch.tanh(self.convq(torch.cat([r*h, motion_features, inp], dim=1), 
                                            offset_mask=offset_mask))
            else:
                hx_r_gat = self.gather_convq(torch.cat([r*h, motion_features, inp], dim=1))
                q_gat = torch.tanh(self.sige_convq(hx_r_gat))
                q = self.scatter_convq(q_gat)

        else:
            z = torch.sigmoid(self.convz(hx))
            r = torch.sigmoid(self.convr(hx))

            if self.mdconv:
                with autocast(enabled=False):
                    q = torch.tanh(self.convq(torch.cat([r*h, motion_features, inp], dim=1).to(r.dtype), offset_mask=offset_mask.to(r.dtype)))
            else:
                q = torch.tanh(self.convq(torch.cat([r*h, motion_features, inp], dim=1)))

        if self.sparse_dropout:
            inject_noise = self.sdrop(q)

            if self.training:
                h = (1-z) * h + z * q * inject_noise
            else:
                h = (1-z) * h + z * q
        else:
            h = (1-z) * h + z * q

        if fast_infer:
            self.prev_z = z
            self.prev_r = r
            self.prev_q = q

        return h

    def get_reg(self):
        return self.sdrop.get_reg()


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=2, dilation=2, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UncertaintyRefineNet(nn.Module):
    def __init__(self, net_dim=128, corr_dim=192, hidden_dim=48, output_dim=1):
        super(UncertaintyRefineNet, self).__init__()

        self.net_enc = nn.Sequential(
            nn.Conv2d(net_dim+1, net_dim//2, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(net_dim//2, 32, 1, padding=0, bias=True))
        self.corr_enc = nn.Sequential(
            nn.Conv2d(corr_dim+1, corr_dim//2, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(corr_dim//2, 64, 1, padding=0, bias=True))
        self.conv1 = nn.Conv2d(32+64, hidden_dim, 1, padding=0)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, net, corr, uncertainty, depth):
        corr_enc = self.corr_enc(torch.cat([uncertainty, corr], dim=1))
        net_enc = self.net_enc(torch.cat([depth, net], dim=1))
        x = torch.cat([corr_enc, net_enc], dim=1)
        out = self.conv2(self.relu(self.conv1(x)))
        out = torch.sigmoid(out)
        return out


class CAttnHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(CAttnHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class FeatScaleNet(nn.Module):
    def __init__(self, input_dim=16, output_dim=32):
        super(FeatScaleNet, self).__init__()
        # scale extractor
        self.scale_conv1 = nn.Conv2d(input_dim + 1, input_dim, kernel_size=3, padding=1)
        self.scale_bn1 = nn.BatchNorm2d(input_dim)
        self.scale_conv2 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1)
        self.scale_bn2 = nn.BatchNorm2d(input_dim)
        self.scale_conv3 = nn.Conv2d(input_dim,  1, kernel_size=1, padding=0)
        self.scale_bn3 = nn.BatchNorm2d(1)

        # embedding
        # self.embed_conv1 = nn.Conv2d(input_dim*3, input_dim, kernel_size=3, padding=1)
        # self.embed_bn1 = nn.BatchNorm2d(input_dim)
        self.embed_conv2 = nn.Conv2d(input_dim, output_dim, kernel_size=6, padding=0, stride=6)
        self.embed_bn2 = nn.BatchNorm2d(output_dim)

    def forward(self, feat, depth):
        # scale inference networks
        x = F.relu(self.scale_bn1(self.scale_conv1(torch.cat([feat, depth], dim=1))))
        x = F.relu(self.scale_bn2(self.scale_conv2(x)))
        scale = 2*torch.sigmoid(self.scale_bn3(self.scale_conv3(x)))

        b, c, h, w = x.size()
        device = x.device

        grid_h, grid_w = torch.meshgrid(torch.linspace(-1, 1, h, dtype=torch.float, requires_grad=False, device=device),
                                        torch.linspace(-1, 1, w, dtype=torch.float, requires_grad=False, device=device))
        grid_h = grid_h.repeat(b,1,1,1)
        grid_w = grid_w.repeat(b,1,1,1)
        grid = torch.cat((grid_w,grid_h),1).permute(0,2,3,1)

        scale_t = scale.transpose(1,2).transpose(2,3)

        grid_enlarge = torch.zeros([b,3*h,3*w,2], device=device)

        step_x = 2/(h-1)
        step_y = 2/(w-1)

        grid_enlarge[:,0::3,0::3,:] = grid+torch.cat(((-1)*step_y*scale_t,(-1)*step_x*scale_t),3)
        grid_enlarge[:,0::3,1::3,:] = grid+torch.cat((( 0)*step_y*scale_t,(-1)*step_x*scale_t),3)
        grid_enlarge[:,0::3,2::3,:] = grid+torch.cat(((+1)*step_y*scale_t,(-1)*step_x*scale_t),3)
        grid_enlarge[:,1::3,0::3,:] = grid+torch.cat(((-1)*step_y*scale_t,( 0)*step_x*scale_t),3)
        grid_enlarge[:,1::3,1::3,:] = grid+torch.cat((( 0)*step_y*scale_t,( 0)*step_x*scale_t),3)
        grid_enlarge[:,1::3,2::3,:] = grid+torch.cat(((+1)*step_y*scale_t,( 0)*step_x*scale_t),3)
        grid_enlarge[:,2::3,0::3,:] = grid+torch.cat(((-1)*step_y*scale_t,(+1)*step_x*scale_t),3)
        grid_enlarge[:,2::3,1::3,:] = grid+torch.cat((( 0)*step_y*scale_t,(+1)*step_x*scale_t),3)
        grid_enlarge[:,2::3,2::3,:] = grid+torch.cat(((+1)*step_y*scale_t,(+1)*step_x*scale_t),3)

        feat_enlarge = F.grid_sample(feat,grid_enlarge,align_corners=True)

        feat2 = F.relu(self.embed_bn2(self.embed_conv2(feat_enlarge)))
        del feat_enlarge, grid_enlarge, grid
        return feat2


class DeformWeightNet(nn.Module):
    def __init__(self, in_channels, k=3, kernel_size=3, stride=2, dilation=2, deformable_groups=1):
        '''
        k: 3 if modulation else 2
        '''
        super(DeformWeightNet, self).__init__()
        offset_out_channels = deformable_groups * k * kernel_size * kernel_size
        self.convdep = nn.Conv2d(1, 8, 3, padding=1)
        self.convcon = nn.Conv2d(1, 8, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.offset_mask_conv = nn.Conv2d(in_channels+16, offset_out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=dilation, dilation=dilation,
                                     groups=deformable_groups, bias=True)
        # Initialize the weight for offset_conv as 0 to act like regular conv
        nn.init.constant_(self.offset_mask_conv.weight, 0.)
        nn.init.constant_(self.offset_mask_conv.bias, 0.)

    def forward(self, x, depth, conf):
        dep_fea = self.relu1(self.convdep(depth))
        con_fea = self.relu1(self.convcon(conf))
        offset_mask = self.offset_mask_conv(torch.cat([x, dep_fea, con_fea], dim=1))
        return offset_mask


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius, out_channel=64):
        super(BasicMotionEncoder, self).__init__()

        cor_planes = corr_levels * (2*corr_radius + 1)

        mid_dim = out_channel // 2
        self.convc1 = nn.Conv2d(cor_planes, mid_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(mid_dim, mid_dim, 3, padding=1)
        self.convf1 = nn.Conv2d(1, mid_dim, 7, padding=3)
        self.convf2 = nn.Conv2d(mid_dim, mid_dim, 3, padding=1)
        self.conv = nn.Conv2d(mid_dim+mid_dim, out_channel-1, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))

        return torch.cat([out, flow], dim=1)


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, n_gru_layers, corr_levels, corr_radius, n_downsample, hidden_dims, motion_dim=64, mdconv=False, pac=False, need_conf=True,
                 deformable_groups=1, sparse_dropout=False, sparse_infer=False):
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius, out_channel=motion_dim)
        encoder_output_dim = motion_dim
        self.n_gru_layers = n_gru_layers
        self.sparse_dropout = sparse_dropout
        self.sparse_infer = sparse_infer

        self.gru04 = ConvGRU(hidden_dims[0], encoder_output_dim+hidden_dims[0],
                             mdconv=mdconv, pac=pac, deformable_groups=deformable_groups,
                             sparse_dropout=self.sparse_dropout, sparse_infer=self.sparse_infer)
        self.flow_head = FlowHead(hidden_dims[0], hidden_dim=128, output_dim=1)
        factor = 2**n_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[0], 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (factor**2)*9, 1, padding=0))

    def forward(self, net, inp, corr=None, flow=None, need_mask=True,
                offset_mask=None, prev_clip_mask=None, fast_infer=False):
        motion_features = self.encoder(flow, corr)
        
        net[0] = self.gru04(net[0], motion_features, inp, offset_mask=offset_mask, prev_clip_mask=prev_clip_mask,
                            fast_infer=fast_infer)

        delta_flow = self.flow_head(net[0])
        if need_mask:
            mask = .25 * self.mask(net[0])
        else:
            mask = None

        return net, mask, delta_flow

    def get_reg(self):
        return self.gru04.get_reg()


class ASMVSNet(nn.Module):
    def __init__(self, nsrc=4, n_depths=192, n_iters=12, n_iters_val=6, mixed_precision=True, 
                base_chs=[48, 32, 16], motion_dim=64, hidden_dims=[128, 64], n_gru_layers=3, 
                corr_levels=3, corr_radius=4, n_downsample=1, need_conf=True, 
                feat_mdconv=True, gru_mdconv=True, sparse_dropout=True, sparse_infer=True,
                use_attn_loss=True, lambda_ce=10, loss_annealing=False, lambda_kl=1e-4):
        super(ASMVSNet, self).__init__()

        self.n_depths = n_depths
        self.base_chs = base_chs
        self.ds_ratio = {"stage1": 4.0,
                         "stage2": 2.0,
                         "stage3": 1.0
                         }
        self.mixed_precision = mixed_precision
        self.n_gru_layers = n_gru_layers
        self.n_downsample = n_downsample
        self.need_conf = need_conf
        self.nsrc = nsrc
        self.n_iters = n_iters
        self.n_iters_val = n_iters_val
        self.corr_radius = corr_radius
        self.use_attn_loss = use_attn_loss
        self.lambda_ce = lambda_ce
        self.lambda_kl = lambda_kl
        self.loss_annealing = loss_annealing
        self.sparse_dropout = sparse_dropout
        self.sparse_infer = sparse_infer

        context_dims=hidden_dims 
        self.hidden_dims = hidden_dims
        self.deformable_groups = 1

        self.feature_extraction = FeatureNet(mdconv=feat_mdconv, use_dynamic_feature=True)
        self.context_extraction_v2 = ContextNet(mdconv=feat_mdconv, use_dynamic_feature=True) # out_channel=16
        d_model=48
        self.self_trans_layers = nn.ModuleList([TransformerEncoderLayer(
            d_model=d_model,
            n_heads=8,
            d_ff=256,
            dropout=0.) for _ in range(4)])
        self.cross_trans_layers = nn.ModuleList([TransformerEncoderLayer(
            d_model=d_model,
            n_heads=8,
            d_ff=256,
            dropout=0.) for _ in range(4)])
        self.context_trans_layers = nn.ModuleList([TransformerEncoderLayer(
            d_model=d_model,
            n_heads=8,
            d_ff=256,
            dropout=0.) for _ in range(4)])
        self.update_block_v2 = BasicMultiUpdateBlock(n_gru_layers, corr_levels, corr_radius, n_downsample, hidden_dims=hidden_dims,
                                                  motion_dim=motion_dim, mdconv=gru_mdconv, pac=False, need_conf=self.need_conf,
                                                  deformable_groups=self.deformable_groups,
                                                  sparse_dropout=self.sparse_dropout, sparse_infer=self.sparse_infer)

        self.feat_scale_net_v2 = FeatScaleNet(input_dim=base_chs[2], output_dim=hidden_dims[0])

        self.offset_mask_conv = DeformWeightNet(in_channels=base_chs[2], deformable_groups = self.deformable_groups)
        self.hidden_init_head04_v2 = nn.Sequential(
            nn.Conv2d(n_depths+1, 128, 1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dims[0], 1, stride=1, padding=0, dilation=1),
        )
        self.G = 8
        self.pixel_wise_net = PixelViewWeight(self.G)
        self.pixel_wise_net_sp = PixelViewWeight(self.G)
        if self.need_conf:
            self.uncertainty_refine_net_v2 = UncertaintyRefineNet(net_dim=hidden_dims[0], corr_dim=n_depths, output_dim=1)
        self.cattn_head_v2 = CAttnHead(hidden_dims[0]+1, hidden_dim=hidden_dims[0]//2, output_dim=1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.self_trans_layers.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.cross_trans_layers.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def regress_depth_local(self, prob_volume_sp, corr_ind, depth_samps_sp):
        n_depths = self.n_depths
        with torch.no_grad():
            corr_radius = self.corr_radius
            index_sp = torch.argmax(prob_volume_sp, dim=1, keepdim=True)
            index = corr_ind.gather(1, index_sp)
            index_low = index - corr_radius
            index_high = index + corr_radius
            mask = (corr_ind >= index_low) & (corr_ind <= index_high)

        local_prob = prob_volume_sp * mask
        local_samps = depth_samps_sp * mask
        regress_depth = (local_prob * local_samps).sum(dim=1, keepdim=True) / (local_prob.sum(dim=1, keepdim=True) + eps)
        return regress_depth #, index

    def init_depth(self, corr_sp, corr_ind, depth_samps_sp):
        prob_volume_sp = F.softmax(corr_sp, dim=1)
        conf = prob_volume_sp.max(dim=1, keepdim=True)[0]
        depth = self.regress_depth_local(prob_volume_sp, corr_ind, depth_samps_sp)
        return depth, conf

    def upsample_depth(self, depth, mask):
        N, D, H, W = depth.shape
        factor = 2 ** self.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        depth_pad = F.pad(depth, (1,1,1,1), 'replicate')
        up_depth = F.unfold(depth_pad, [3,3], padding=0)
        up_depth = up_depth.view(N, D, 9, 1, 1, H, W)

        up_depth = torch.sum(mask * up_depth, dim=2)
        up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)
        return up_depth.reshape(N, D, factor*H, factor*W)

    def compute_depth(self, ref_feat, src_feats, ref_context, depth_range, inverse_depth_base,
                      inverse_depth_multiplier, proj_l3, proj_l2, is_training=False,
                      homo_samp_coords=None, depth_samps_flat=None):
        n_iters = self.n_iters if self.training else self.n_iters_val

        with autocast(enabled=self.mixed_precision):

            # initialization, 1/8 resolution dense corr, 192 hypos; 1/4 sparse corr, 16 hypos
            corr_fn = CorrBlock1D(self.pixel_wise_net, self.pixel_wise_net_sp, ref_feat, src_feats, depth_range, 
                                self.n_depths, proj_l3, proj_l2, num_levels=3, G=self.G,
                                training=self.training, depth_samps_flat=depth_samps_flat,
                                res_div=True) 
            corr_init = corr_fn.corr_init

            cur_depth, conf = self.init_depth(corr_fn.corr_sp, corr_fn.corr_ind, corr_fn.depth_samps_sp)
            cur_depth_inv = 1 / (cur_depth + eps)
            sigmoid_depth = (cur_depth_inv - inverse_depth_base) / inverse_depth_multiplier

            init_ds=2
            if n_iters == 0:
                pred_depths = [F.interpolate(cur_depth, scale_factor=2**init_ds, mode="bilinear", align_corners=False)]
            else:
                pred_depths = []
            pred_confs = []

        if n_iters > 0:
            with autocast(enabled=self.mixed_precision):
                net_feats04 = self.hidden_init_head04_v2(torch.cat([corr_fn.corr_full, conf.detach()], dim=1))
                inp_feats04 = self.feat_scale_net_v2(ref_context,
                                                     F.interpolate(sigmoid_depth, scale_factor=2, mode="bilinear", align_corners=False).detach())
                net_feats = [torch.tanh(net_feats04)]
                inp_feats = [torch.relu(inp_feats04)]

                cattn_guid = self.cattn_head_v2(torch.cat([net_feats[0], conf], dim=1))
                cattn_inp = torch.sigmoid(cattn_guid)

                if self.training:
                    inp_feats[0] = inp_feats[0] * cattn_inp
                else:
                    inp_feats[0] *= cattn_inp
                inp_fea = inp_feats[0]

            offset_mask = self.offset_mask_conv(ref_context, 
                F.interpolate(conf, scale_factor=2, mode="bilinear", align_corners=False).detach(),
                F.interpolate(sigmoid_depth, scale_factor=2, mode="bilinear", align_corners=False).detach())

        reg_loss_list = []
        clip_mask = None
        fast_infer = (not self.training) and self.sparse_dropout and self.sparse_infer

        with autocast(enabled=False):
            for itr in range(n_iters):
                if self.training:
                    cur_depth_inv = cur_depth_inv.detach()
                sigmoid_depth = (cur_depth_inv - inverse_depth_base) / inverse_depth_multiplier
                corr = corr_fn.lookup(sigmoid_depth)
                
                net_feats, up_mask, depth_guid = self.update_block_v2(net_feats, inp_fea, corr, sigmoid_depth,
                                                                    need_mask=True,
                                                                    offset_mask=offset_mask,
                                                                    prev_clip_mask=clip_mask, fast_infer=fast_infer)

                depth_inv_res = torch.tanh(depth_guid) * inverse_depth_multiplier
                if clip_mask is None:
                    cur_depth_inv = cur_depth_inv + depth_inv_res
                else:
                    cur_depth_inv = cur_depth_inv + depth_inv_res * clip_mask
                
                if self.sparse_dropout:
                    if self.training:
                        reg_loss_list.append(self.update_block_v2.get_reg())
                    else:
                        clip_mask = self.update_block_v2.gru04.sdrop.clip_mask

                exit_loop = fast_infer and (clip_mask.sum()/clip_mask.numel() < 0.01)
                if self.training or (itr == n_iters - 1) or exit_loop:
                    cur_depth_inv_up = self.upsample_depth(cur_depth_inv.float(), up_mask.float())
                    depth_up = 1 / (cur_depth_inv_up + eps)
                    pred_depths.append(depth_up)

                if exit_loop:
                    break

        confidence_0 = F.interpolate(conf, scale_factor=2**init_ds, mode="bilinear", align_corners=False)
        return {"depth": pred_depths[-1], "confidence": confidence_0}

    def feature_transform_infer(self, feature, proj_inv, ref_aug_feas=None, is_ref=False):
        B,C,H,W = feature.shape

        pos_enc = ray_pos_enc(feature, proj_inv, num=7)
        aug_fea = feature + pos_enc
        aug_fea = aug_fea.flatten(2).permute(0,2,1)
        for i, self_layer, cross_layer in zip(range(4), self.self_trans_layers, self.cross_trans_layers):
            aug_fea = self_layer(aug_fea, aug_fea)
            aug_fea[1:] = cross_layer(aug_fea[1:], aug_fea[:1].repeat(self.nsrc,1,1))
        aug_fea = aug_fea.permute(0,2,1).view(B,C,H,W)
        return aug_fea

    def feature_transform(self, feature, proj_inv, ref_aug_feas=None, is_ref=False):
        if is_ref:
            ref_aug_feas = []

        with torch.no_grad():
            pos_enc = ray_pos_enc(feature, proj_inv, num=7)
        feat = feature
        B,C,H,W = feat.shape
        feat = feat.flatten(2).permute(0,2,1)
        pos_enc = pos_enc.flatten(2).permute(0,2,1)
        aug_fea = feat + pos_enc

        for i, self_layer, cross_layer in zip(range(4), self.self_trans_layers, self.cross_trans_layers):
            aug_fea = self_layer(aug_fea, aug_fea)
            if is_ref:
                ref_aug_feas.append(aug_fea)
            else:
                aug_fea = cross_layer(aug_fea, ref_aug_feas[i])
        aug_fea = aug_fea.permute(0,2,1).view(B,C,H,W)
        return aug_fea, ref_aug_feas

    def context_transform(self, feature, proj_inv, ref_aug_feas=None, is_ref=False):
        with torch.no_grad():
            pos_enc = ray_pos_enc(feature, proj_inv, num=7)
        B,C,H,W = feature.shape
        aug_fea = (feature + pos_enc).flatten(2).permute(0,2,1)
        for self_layer in self.context_trans_layers:
            aug_fea = self_layer(aug_fea, aug_fea)
        aug_fea = aug_fea.permute(0,2,1).view(B,C,H,W)
        return aug_fea, None

    def forward(self, ref_img, src_imgs, ref_in, src_in, ref_ex, src_ex, depth_range):

        proj_l2, proj_l3, cam_invs_l3, ref_cam_inv_l3 = cal_cam_proj(ref_in, src_in, ref_ex, src_ex) 
        imgs = torch.cat([ref_img.unsqueeze(1), src_imgs], dim=1)

        with autocast(enabled=self.mixed_precision):
            features = self.feature_extraction(imgs, cam_invs_l3, self.feature_transform, self.feature_transform_infer, is_training=self.training)
            ref_feature = {
                "level1":features['level1'][0],
                "level2":features['level2'][0],
            }
            src_features = {
                "level1": features['level1'][1:],
                "level2": features['level2'][1:],
            }

            ref_context = self.context_extraction_v2(ref_img, ref_cam_inv_l3, self.context_transform,
                                                     is_training=self.training)

        inverse_depth_base = 1 / depth_range[:, 1]    # Bx1
        inverse_depth_multiplier = 1 / depth_range[:,0] - 1 / depth_range[:,1]  # B
        inverse_samps = inverse_depth_base + inverse_depth_multiplier * (self.n_depths - torch.arange(0, self.n_depths, device=depth_range.device) - 1) / self.n_depths 
        depth_samps_flat = 1 / inverse_samps.view(-1,1,1)   # Nx1x1
        inverse_depth_base = inverse_depth_base.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # Bx1x1x1
        inverse_depth_multiplier = inverse_depth_multiplier.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # Bx1x1x1

        outputs = self.compute_depth(ref_feature, src_features, ref_context, depth_range,
                                     inverse_depth_base, inverse_depth_multiplier,
                                     proj_l3, proj_l2, is_training=self.training,
                                     depth_samps_flat=depth_samps_flat)
        return outputs
