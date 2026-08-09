# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Adapted for EigenMAP anchor-free detection

import random
import math
from collections import OrderedDict
import numpy as np
import torch
import h5py

from revqom.utils import box_utils
from revqom.utils import pcd_utils
from revqom.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
from revqom.utils.common_utils import merge_features_to_dict
from revqom.utils.transformation_utils import x1_to_x2, get_pairwise_transformation
from revqom.utils.pcd_utils import mask_ego_points, shuffle_points, downsample_lidar_minimum
from revqom.hypes_yaml.yaml_utils import load_yaml


class OPV2VIntermediateFusionDataset(OPV2VBaseDataset):
    def __init__(self, params, visualize, train=True):
        super().__init__(params, visualize, train)

        self.proj_first = params['fusion']['args'].get('proj_first', False)
        self.supervise_single = params['fusion']['args'].get('supervise_single', False)

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        timestamp_keys = list(scenario_database.items())[0][1]
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]
        return timestamp_key

    def retrieve_base_data(self, idx):
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
            idx -= ele

        scenario_database = self.scenario_database[scenario_index]
        timestamp_index = idx
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)

        data = OrderedDict()

        for j, (cav_id, cav_content) in enumerate(scenario_database.items()):
            if cav_id == 'scene_len':
                continue
            if j >= self.max_cav:
                break

            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = (j == 0)

            cav_yaml = load_yaml(cav_content[timestamp_key]['yaml'])
            if 'lidar_pose' not in cav_yaml:
                cav_yaml['lidar_pose'] = cav_yaml.get('true_ego_pos', [0, 0, 0, 0, 0, 0])
            data[cav_id]['params'] = cav_yaml

            if self.load_lidar_file or self.visualize:
                lidar_path = cav_content[timestamp_key]['lidar']
                if self.use_hdf5:
                    hdf5_path = lidar_path.replace('.pcd', '.hdf5')
                    with h5py.File(hdf5_path, 'r') as f:
                        lidar_np = f['lidar'][()]
                else:
                    lidar_np = pcd_utils.pcd_to_np(lidar_path)
                data[cav_id]['lidar_np'] = lidar_np

        return data

    def get_item_single_car(self, selected_cav_base, ego_cav_base):
        selected_cav_processed = {}

        ego_pose = ego_cav_base['params']['lidar_pose']
        transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_pose)

        if self.load_lidar_file or self.visualize:
            lidar_np = selected_cav_base['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            lidar_np = mask_ego_points(lidar_np)

            projected_lidar = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)

            if self.proj_first:
                lidar_np[:, :3] = projected_lidar

            if self.visualize:
                selected_cav_processed['projected_lidar'] = projected_lidar

            processed_lidar = self.pre_processor.preprocess(lidar_np)
            selected_cav_processed['processed_features'] = processed_lidar

        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
            [selected_cav_base], ego_pose
        )

        selected_cav_processed.update({
            'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
            'object_bbx_mask': object_bbx_mask,
            'object_ids': object_ids,
            'transformation_matrix': transformation_matrix
        })

        return selected_cav_processed

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []
        ego_cav_base = None

        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_cav_base = cav_content
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        processed_features = []
        object_stack = []
        object_id_stack = []
        too_far = []
        lidar_pose_list = []
        cav_id_list = []

        if self.visualize:
            projected_lidar_stack = []

        for cav_id, selected_cav_base in base_data_dict.items():
            distance = math.sqrt(
                (selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 +
                (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2
            )

            if distance > self.params['comm_range']:
                too_far.append(cav_id)
                continue

            lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])
            cav_id_list.append(cav_id)

        for cav_id in too_far:
            base_data_dict.pop(cav_id)

        pairwise_t_matrix = get_pairwise_transformation(base_data_dict, self.max_cav, self.proj_first)
        lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)
        cav_num = len(cav_id_list)

        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]
            selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_cav_base)

            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']

            if self.load_lidar_file:
                processed_features.append(selected_cav_processed['processed_features'])

            if self.visualize:
                projected_lidar_stack.append(selected_cav_processed['projected_lidar'])

        unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack) if object_stack else np.zeros((0, 7))
        object_stack = object_stack[unique_indices] if len(unique_indices) > 0 else object_stack

        if self.load_lidar_file:
            merged_feature_dict = merge_features_to_dict(processed_features)
            processed_data_dict['ego']['processed_lidar'] = merged_feature_dict

        processed_data_dict['ego'].update({
            'gt_boxes': object_stack,
            'object_ids': [object_id_stack[i] for i in unique_indices],
            'cav_num': cav_num,
            'pairwise_t_matrix': pairwise_t_matrix,
            'lidar_poses': lidar_poses,
            'sample_idx': idx,
            'cav_id_list': cav_id_list
        })

        if self.visualize:
            processed_data_dict['ego']['origin_lidar'] = np.vstack(projected_lidar_stack)

        return processed_data_dict

    def collate_batch_train(self, batch):
        output_dict = {'ego': {}}

        gt_boxes_list = []
        object_ids_list = []
        processed_lidar_list = []
        record_len = []
        lidar_pose_list = []
        pairwise_t_matrix_list = []
        origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            gt_boxes_list.append(ego_dict['gt_boxes'])
            object_ids_list.append(ego_dict['object_ids'])
            lidar_pose_list.append(ego_dict['lidar_poses'])
            record_len.append(ego_dict['cav_num'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

            if self.load_lidar_file:
                processed_lidar_list.append(ego_dict['processed_lidar'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        gt_boxes_list = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in gt_boxes_list]
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        if self.load_lidar_file:
            merged_feature_dict = merge_features_to_dict(processed_lidar_list)
            processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_feature_dict)
            output_dict['ego']['processed_lidar'] = processed_lidar_torch_dict

        output_dict['ego'].update({
            'gt_boxes': gt_boxes_list,
            'object_ids': object_ids_list,
            'record_len': record_len,
            'lidar_pose': lidar_pose,
            'pairwise_t_matrix': pairwise_t_matrix
        })

        if self.visualize:
            origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            output_dict['ego']['origin_lidar'] = torch.from_numpy(origin_lidar)

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None

        transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()

        output_dict['ego'].update({
            'transformation_matrix': transformation_matrix_torch,
            'transformation_matrix_clean': transformation_matrix_torch,
            'sample_idx': batch[0]['ego']['sample_idx'],
            'cav_id_list': batch[0]['ego']['cav_id_list']
        })

        return output_dict

    def post_process(self, data_dict, output_dict):
        pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor, gt_label_tensor = self.post_processor.generate_gt_bbx(data_dict)
        return pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor
