# Anchor-free version by anynomous author

import torch
import torch.nn as nn
from revqom.models.sub_modules.mean_vfe import MeanVFE
from revqom.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from revqom.models.sub_modules.height_compression import HeightCompression
from revqom.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from revqom.models.sub_modules.focalcomm_transfusion_head import TransFusionHead
from revqom.models.fuse_modules.fuse_utils import regroup
from revqom.models.fuse_modules.fusion_in_one import CoBEVT
from revqom.utils.transformation_utils import normalize_pairwise_tfm
from revqom.models.sub_modules.downsample_conv import DownsampleConv
from revqom.models.sub_modules.compress import NaiveCompressor, ChannelRVQ, ChannelRVQ_EMA, ResidualFSQ


class Cobevt(nn.Module):
    """
    Co-Event Fusion implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(Cobevt, self).__init__()

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

        self.height_compression = HeightCompression(args['height_compression'])
        # breakpoint()
        # self.out_channel = self.height_compression.feature_num
        # print("self.backbone", self.backbone)
        self.out_channel = 256
        self.compression = False
        if 'compression' in args:
            self.compression = True
            compression_config = args['compression']
            method = compression_config['compression_method']              
            print(f"Using {method} compression with config: {compression_config}")
            
            if method == 'NaiveCompressor':
                self.compressor = NaiveCompressor(self.out_channel, compression_config['channel_reduction_ratio'])
            elif method == 'ChannelRVQ':
                self.compressor = ChannelRVQ(
                    C=self.out_channel,
                    ratio=compression_config['channel_reduction_ratio'],
                    codebook_size=compression_config.get('codebook_size', 64),
                    n_q=compression_config.get('n_q', 3),
                    beta=compression_config.get('beta', 0.05),
                    ortho_reg=compression_config.get('ortho_reg', 0.0001)
                )
            elif method == 'ChannelRVQ_EMA':
                self.compressor = ChannelRVQ_EMA(
                    C=self.out_channel,
                    C_rr=compression_config['channel_reduction_ratio'],
                    codebook_size=compression_config.get('codebook_size', 64),
                    n_q=compression_config.get('n_q', 3),
                    beta_commit=compression_config.get('beta_commit', 0.05),
                    ema_decay=compression_config.get('ema_decay', 0.95),
                    eps=compression_config.get('eps', 1e-5),
                    use_groupnorm=compression_config.get('use_groupnorm', True),
                    ortho_reg=compression_config.get('ortho_reg', 0.0001)
                )
            elif method == 'ResidualFSQ':
                self.compressor = ResidualFSQ(
                    C=self.out_channel,
                    ratio=compression_config['channel_reduction_ratio'],
                    levels=compression_config.get('levels', [8, 8, 8, 8]),
                    num_quantizers=compression_config.get('num_quantizers', 4),
                    beta_commit=compression_config.get('beta_commit', 0.05),
                    ortho_reg=compression_config.get('ortho_reg', 0.0001)
                )
            else:
                raise ValueError(f"Unknown compression method: {method}")
                
            print(f"Initialized {method} compressor successfully")

        

        self.fusion_net = CoBEVT(args['cobevt'])

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
        # breakpoint()
        batch_dict = self.height_compression(batch_dict)
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features = batch_dict['spatial_features']
        # breakpoint()
        # Apply compression if enabled
        vq_loss = None
        ortho_loss = None
        
        if self.compression and self.compressor is not None:
            # Check if compressor returns losses (VQ-based methods)
            if hasattr(self.compressor, 'forward') and 'NaiveCompressor' not in str(type(self.compressor)):
                spatial_features, loss_dict = self.compressor(spatial_features)
                vq_loss = loss_dict.get('vq_loss', None)
                ortho_loss = loss_dict.get('ortho_loss', None)
                # Handle additional losses like perplexity for debugging
                if 'perplexity' in loss_dict:
                    # Could be logged or added to output_dict if needed
                    pass
            else:
                # NaiveCompressor doesn't return losses
                spatial_features = self.compressor(spatial_features)

        fused_features = self.fusion_net(spatial_features, record_len, normalized_affine_matrix)
        if self.shrink_flag:
            fused_features = self.shrink_conv(fused_features)

        # Add spatial_features_2d for TransFusion head compatibility
        batch_dict['spatial_features_2d'] = fused_features.contiguous()
        
        output_dict = self.head(batch_dict)
        
        # Add compression losses to output if present
        if vq_loss is not None:
            output_dict['vq_loss'] = vq_loss
        if ortho_loss is not None:
            output_dict['ortho_loss'] = ortho_loss
        
        
        return output_dict
