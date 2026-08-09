# Anchor-free version by anynomous author

# Adapted from https://github.com/yifanlu0227/CoAlign by
# Yifan Lu <yifan_lu@sjtu.edu.cn>
import torch
import torch.nn as nn
from revqom.models.sub_modules.mean_vfe import MeanVFE
from revqom.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from revqom.models.sub_modules.height_compression import HeightCompression
from revqom.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from revqom.models.sub_modules.focalcomm_transfusion_head import TransFusionHead
from revqom.models.fuse_modules.fuse_utils import regroup
from revqom.models.fuse_modules.fusion_in_one import AttFusion
from revqom.utils.transformation_utils import normalize_pairwise_tfm
from revqom.models.sub_modules.compress import NaiveCompressor, ChannelRVQ_EMA
from revqom.models.sub_modules.downsample_conv import DownsampleConv

class CoAlign(nn.Module):
    """
    V2XViT implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(CoAlign, self).__init__()

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.batch_size = args['batch_size']
        self.max_cav = args['max_cav']
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'],4, args['grid_size'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 256)
        self.voxel_size = args['voxel_size']
        self.num_levels = len(args['base_bev_backbone']['layer_nums'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        self.compression_method = args.get('compression_method', 'naive')
        
        if "compression" in args and args['compression'] > 0:
            self.compression = True
            if self.compression_method == 'naive':
                self.compressor = NaiveCompressor(256, args['compression'])
                print(f"Using NaiveCompressor with ratio {args['compression']}")
            elif self.compression_method == 'channel_rvq_ema':
                print(f"Using ChannelRVQ_EMA with ratio {args['compression']}")
                codebook_size = args.get('codebook_size', 128)
                n_q = args.get('n_q', 3)
                beta_commit = args.get('beta_commit', 0.1)
                ema_decay = args.get('ema_decay', 0.99)
                use_groupnorm = args.get('use_groupnorm', True)
                ortho_reg = args.get('ortho_reg', 1e-4)
                self.compressor = ChannelRVQ_EMA(
                    C=256, 
                    ratio=args['compression'],
                    codebook_size=codebook_size,
                    n_q=n_q,
                    beta_commit=beta_commit,
                    ema_decay=ema_decay,
                    use_groupnorm=use_groupnorm,
                    ortho_reg=ortho_reg
                )

        self.height_compression = HeightCompression(args['height_compression'])

        self.fusion_net = nn.ModuleList()
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            self.fusion_net.append(AttFusion(args['coalign']['feat_dim'][i]))

        # self.fusion_net = AttFusion(args['coalign'])
        head_args = args['dense_head']
        self.head = TransFusionHead(head_args)


    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'batch_size': torch.sum(record_len).cpu().numpy(),
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.mean_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features = batch_dict['spatial_features']

        vq_loss = None
        ortho_loss = None
        # breakpoint()
        if self.compression:
            if self.compression_method == 'naive':
                spatial_features = self.compressor(spatial_features)
            elif self.compression_method == 'channel_rvq_ema':
                # breakpoint()
                spatial_features, loss_dict = self.compressor(spatial_features)
                vq_loss = loss_dict.get('vq_loss', None)
                ortho_loss = loss_dict.get('ortho_loss', None)

        # multiscale fusion
        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(fuse_module(feature_list[i], record_len, normalized_affine_matrix))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list) 

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        # Add spatial_features_2d for TransFusion head compatibility
        batch_dict['spatial_features_2d'] = fused_feature.contiguous()
        # breakpoint()
        output_dict = self.head(batch_dict)


        if vq_loss is not None:
            output_dict['vq_loss'] = vq_loss
        if ortho_loss is not None:
            output_dict['ortho_loss'] = ortho_loss    

        return output_dict
