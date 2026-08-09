import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from revqom.models.sub_modules.mean_vfe import MeanVFE
from revqom.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from revqom.models.sub_modules.height_compression import HeightCompression
from revqom.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from revqom.models.sub_modules.focalcomm_transfusion_head import TransFusionHead
from revqom.models.sub_modules.torch_transformation_utils import warp_affine_simple
from revqom.utils.transformation_utils import normalize_pairwise_tfm
from revqom.models.sub_modules.naive_compress import NaiveCompressor
from revqom.models.sub_modules.downsample_conv import DownsampleConv

from revqom.models.fuse_modules.where2comm_attn import Where2comm 

class Where2Comm(nn.Module):
    def __init__(self, args):
        super(Where2Comm, self).__init__()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.max_cav = args['max_cav']
        
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        # if len(args.get('backbone_3d', [])) > 0:
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'], 4, args['grid_size'])
        self.height_compression = HeightCompression(args['height_compression'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], args['height_compression']['feature_num'])
        
        self.voxel_size = args['voxel_size']
        self.num_levels = len(args['base_bev_backbone']['layer_nums'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]   

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(64, args['compression'])

        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']
        # self.fusion_net = AttFusion(args['coalign'])
        head_args = args['dense_head']       
        self.head = TransFusionHead(head_args)


    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']        
        record_len = data_dict['record_len']
        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'batch_size': torch.sum(record_len).cpu().numpy(),
            'record_len': record_len
        }
        
        batch_dict = self.mean_vfe(batch_dict)
        
        # if hasattr(self, 'backbone_3d'):
        batch_dict = self.backbone_3d(batch_dict)        
        batch_dict = self.height_compression(batch_dict)
        # batch_dict = self.base_bev_backbone(batch_dict)  
        spatial_features_2d = batch_dict['spatial_features']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)


        psm_single = None # only matters if comm is enabled (in the config -- skipped in where2comm attn)
        

        # print('spatial_features_2d: ', spatial_features_2d.shape)
        if self.multi_scale:
            # For multi-scale, pass the unprocessed features from height compression
            fused_feature, communication_rates, result_dict = self.fusion_net(batch_dict['spatial_features'],
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.backbone)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix)





        # Add spatial_features_2d for TransFusion head compatibility
        batch_dict['spatial_features_2d'] = fused_feature.contiguous()
        # breakpoint()
        output_dict = self.head(batch_dict)

        
        return output_dict

