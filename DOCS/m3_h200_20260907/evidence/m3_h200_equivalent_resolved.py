checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/work_dirs/gmo_radar_m3_h200720_w2_2g_b1_acc2_seed0_fcos3d'
load_from = '/storage/data/metaiot_data/huayiming/RadarFlowOcc/checkpoints/r101_dcn_fcos3d_pretrain_l40s_reference.pth'
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
occ_path = '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/openoccupancy/nuScenes-Occupancy-v0.1'
use_fine_occ = True
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 0.2]
occ_size = [512, 512, 40]
plan_grid_conf = dict(
    xbound=[-50.0, 50.0, 0.5],
    ybound=[-50.0, 50.0, 0.5],
    zbound=[-10.0, 10.0, 20.0])
queue_length = 2
memory_queue_len = 1
future_queue_length_train = 4
future_pred_frame_num_train = 4
future_queue_length_test = 4
future_pred_frame_num_test = 4
future_decoder_layer_num = 3
frame_loss_weight = [[1], [1], [0]]
only_generate_dataset = False
supervise_all_future = True
load_frame_interval = None
turn_on_flow = False
turn_on_plan = False
world_head_pred_history_frame_num = 0
world_head_pred_future_frame_num = 0
world_head_per_frame_loss_weight = (1.0, )
ida_aug_conf = dict(
    reisze=[720], crop=(0, 0, 1600, 900), H=900, W=1600, rand_flip=True)
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
use_separate_classes = False
use_background_classes = False
class_names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction', 'motorcycle',
    'pedestrian', 'trafficcone', 'trailer', 'truck', 'driveable_surface',
    'other', 'sidewalk', 'terrain', 'mannade', 'vegetation'
]
empty_idx = 0
num_cls = 2
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=True)
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
pred_height = 16
model = dict(
    type='Drive_OccWorld',
    turn_on_flow=False,
    turn_on_plan=False,
    memory_queue_len=1,
    use_grid_mask=True,
    video_test_mode=True,
    only_generate_dataset=False,
    supervise_all_future=True,
    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    bev_h=200,
    bev_w=200,
    future_pred_frame_num=4,
    test_future_frame_num=4,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    future_pred_head=dict(
        type='WorldHeadV1',
        num_classes=2,
        history_queue_length=2,
        memory_queue_len=1,
        soft_weight=False,
        turn_on_flow=False,
        obj_motion_norm=False,
        pred_history_frame_num=0,
        pred_future_frame_num=0,
        per_frame_loss_weight=(1.0, ),
        num_pred_fcs=1,
        num_pred_height=16,
        use_can_bus=True,
        use_plan_traj=False,
        use_command=True,
        use_vel_steering=False,
        use_vel=True,
        use_steering=False,
        use_fourier=False,
        condition_ca_add='ca',
        can_bus_norm=True,
        can_bus_dims=(0, 1, 2, 17),
        bev_h=200,
        bev_w=200,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        loss_weight=[[1], [1], [0]],
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=200,
            col_num_embed=200),
        prev_render_neck=dict(
            type='ConditionalNorm',
            occ_flow='occ',
            embed_dims=256,
            sem_norm=False,
            sem_gt_train=False,
            ego_motion_ln=True,
            obj_motion_ln=False,
            pred_height=16,
            num_cls=2,
            num_pred_fcs=0),
        transformer=dict(
            type='PredictionTransformer',
            embed_dims=256,
            decoder=dict(
                type='WorldDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='PredictionTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='PredictionMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='PredictionMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='GroupMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'cross_attn_action', 'norm', 'ffn',
                                     'norm'))))),
    pts_bbox_head=dict(
        type='WorldBEVFormerHead',
        bev_h=200,
        bev_w=200,
        num_query=900,
        num_classes=2,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=256,
            encoder=dict(
                type='CustomBEVFormerEncoder',
                num_layers=6,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayerV2',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=4),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 0.2],
            num_classes=2),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=200,
            col_num_embed=200),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 0.2],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))),
    radar_encoder=dict(
        in_channels=8,
        hidden_channels=64,
        out_channels=256,
        num_groups=8,
        zero_init_residual=True),
    doppler_advection=None,
    doppler_posterior=dict(
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        bev_h=200,
        bev_w=200,
        embed_dims=256,
        hidden_dims=64,
        transport_dims=32,
        time_step=0.5,
        sensor_std=1.5,
        prior_std_bounds=(0.5, 15.0),
        initial_prior_std=5.0,
        max_time_lag=0.15,
        holdout_fraction=0.2,
        gate_init=0.0,
        transport_mode='sigma',
        use_doppler_update=True),
    doppler_flow_loss=None,
    doppler_nll_weight=0.05,
    scientific_eval=True)
dataset_type = 'NuScenesWorldDatasetV1'
data_root = '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/nuscenes_driveocc/'
cam4docc_dataset_path = 'data/cam4docc/'
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='CropResizeFlipImage',
        data_aug_conf=dict(
            reisze=[720],
            crop=(0, 0, 1600, 900),
            H=900,
            W=1600,
            rand_flip=True),
        training=True,
        debug=False),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='LoadOccupancy',
        to_float32=True,
        occ_path=
        '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/openoccupancy/nuScenes-Occupancy-v0.1',
        grid_size=[512, 512, 40],
        unoccupied=0,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        use_fine_occ=True,
        use_separate_classes=False,
        use_background_classes=False,
        time_history_field=2,
        time_future_field=4,
        test_mode=False),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'barrier', 'bicycle', 'bus', 'car', 'construction', 'motorcycle',
            'pedestrian', 'trafficcone', 'trailer', 'truck',
            'driveable_surface', 'other', 'sidewalk', 'terrain', 'mannade',
            'vegetation'
        ]),
    dict(
        type='CustomCollect3D',
        keys=[
            'img', 'aug_param', 'gt_occ', 'vel_steering', 'sdc_planning',
            'sdc_planning_mask', 'command', 'sample_traj', 'gt_future_boxes'
        ])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='LoadOccupancy',
        to_float32=True,
        occ_path=
        '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/openoccupancy/nuScenes-Occupancy-v0.1',
        grid_size=[512, 512, 40],
        unoccupied=0,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        use_fine_occ=True,
        use_separate_classes=False,
        use_background_classes=False,
        time_history_field=2,
        time_future_field=4,
        test_mode=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'barrier', 'bicycle', 'bus', 'car', 'construction', 'motorcycle',
            'pedestrian', 'trafficcone', 'trailer', 'truck',
            'driveable_surface', 'other', 'sidewalk', 'terrain', 'mannade',
            'vegetation'
        ]),
    dict(
        type='CustomCollect3D',
        keys=[
            'img', 'gt_occ', 'vel_steering', 'sdc_planning',
            'sdc_planning_mask', 'command', 'sample_traj', 'segmentation_bev'
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='NuScenesWorldDatasetV1',
        data_root=
        '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/nuscenes_driveocc/',
        ann_file=
        '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/nuscenes_driveocc/nuscenes_infos_temporal_train_new.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='CropResizeFlipImage',
                data_aug_conf=dict(
                    reisze=[720],
                    crop=(0, 0, 1600, 900),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=True,
                debug=False),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='LoadOccupancy',
                to_float32=True,
                occ_path=
                '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/openoccupancy/nuScenes-Occupancy-v0.1',
                grid_size=[512, 512, 40],
                unoccupied=0,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                use_fine_occ=True,
                use_separate_classes=False,
                use_background_classes=False,
                time_history_field=2,
                time_future_field=4,
                test_mode=False),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'barrier', 'bicycle', 'bus', 'car', 'construction',
                    'motorcycle', 'pedestrian', 'trafficcone', 'trailer',
                    'truck', 'driveable_surface', 'other', 'sidewalk',
                    'terrain', 'mannade', 'vegetation'
                ]),
            dict(
                type='CustomCollect3D',
                keys=[
                    'img', 'aug_param', 'gt_occ', 'vel_steering',
                    'sdc_planning', 'sdc_planning_mask', 'command',
                    'sample_traj', 'gt_future_boxes'
                ])
        ],
        classes=[
            'barrier', 'bicycle', 'bus', 'car', 'construction', 'motorcycle',
            'pedestrian', 'trafficcone', 'trailer', 'truck',
            'driveable_surface', 'other', 'sidewalk', 'terrain', 'mannade',
            'vegetation'
        ],
        use_separate_classes=False,
        use_fine_occ=True,
        turn_on_flow=False,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=True,
            use_map=False,
            use_external=True),
        test_mode=False,
        use_valid_flag=True,
        bev_size=(200, 200),
        queue_length=2,
        future_length=4,
        ego_mask=(-0.8, -1.5, 0.8, 2.5),
        load_frame_interval=None,
        plan_grid_conf=dict(
            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0]),
        box_type_3d='LiDAR',
        radar_cfg=None,
        future_metadata_only=True,
        radar_observation_cfg=dict(
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            bev_h=200,
            bev_w=200,
            nsweeps=1,
            max_returns=4096,
            max_time_lag=0.15,
            cache_dir=
            '/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/cache/radar_m3_obs_trainval_1sweep_200x200_v1',
            cache_readonly=True)),
    val=dict(
        type='NuScenesWorldDatasetV1',
        data_root=
        '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/nuscenes_driveocc/',
        ann_file=
        '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/nuscenes_driveocc/nuscenes_infos_temporal_val_new.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='LoadOccupancy',
                to_float32=True,
                occ_path=
                '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/openoccupancy/nuScenes-Occupancy-v0.1',
                grid_size=[512, 512, 40],
                unoccupied=0,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                use_fine_occ=True,
                use_separate_classes=False,
                use_background_classes=False,
                time_history_field=2,
                time_future_field=4,
                test_mode=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'barrier', 'bicycle', 'bus', 'car', 'construction',
                    'motorcycle', 'pedestrian', 'trafficcone', 'trailer',
                    'truck', 'driveable_surface', 'other', 'sidewalk',
                    'terrain', 'mannade', 'vegetation'
                ]),
            dict(
                type='CustomCollect3D',
                keys=[
                    'img', 'gt_occ', 'vel_steering', 'sdc_planning',
                    'sdc_planning_mask', 'command', 'sample_traj',
                    'segmentation_bev'
                ])
        ],
        bev_size=(200, 200),
        use_separate_classes=False,
        use_fine_occ=True,
        turn_on_flow=False,
        classes=[
            'barrier', 'bicycle', 'bus', 'car', 'construction', 'motorcycle',
            'pedestrian', 'trafficcone', 'trailer', 'truck',
            'driveable_surface', 'other', 'sidewalk', 'terrain', 'mannade',
            'vegetation'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=True,
            use_map=False,
            use_external=True),
        samples_per_gpu=1,
        queue_length=2,
        future_length=4,
        ego_mask=(-0.8, -1.5, 0.8, 2.5),
        plan_grid_conf=dict(
            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0]),
        radar_cfg=None,
        radar_observation_cfg=dict(
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            bev_h=200,
            bev_w=200,
            nsweeps=1,
            max_returns=4096,
            max_time_lag=0.15,
            cache_dir=
            '/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/cache/radar_m3_obs_trainval_1sweep_200x200_v1',
            cache_readonly=True)),
    test=dict(
        type='NuScenesWorldDatasetV1',
        data_root=
        '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/nuscenes_driveocc/',
        ann_file=
        '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/nuscenes_driveocc/nuscenes_infos_temporal_val_new.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='LoadOccupancy',
                to_float32=True,
                occ_path=
                '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/openoccupancy/nuScenes-Occupancy-v0.1',
                grid_size=[512, 512, 40],
                unoccupied=0,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                use_fine_occ=True,
                use_separate_classes=False,
                use_background_classes=False,
                time_history_field=2,
                time_future_field=4,
                test_mode=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'barrier', 'bicycle', 'bus', 'car', 'construction',
                    'motorcycle', 'pedestrian', 'trafficcone', 'trailer',
                    'truck', 'driveable_surface', 'other', 'sidewalk',
                    'terrain', 'mannade', 'vegetation'
                ]),
            dict(
                type='CustomCollect3D',
                keys=[
                    'img', 'gt_occ', 'vel_steering', 'sdc_planning',
                    'sdc_planning_mask', 'command', 'sample_traj',
                    'segmentation_bev'
                ])
        ],
        bev_size=(200, 200),
        use_separate_classes=False,
        use_fine_occ=True,
        turn_on_flow=False,
        classes=[
            'barrier', 'bicycle', 'bus', 'car', 'construction', 'motorcycle',
            'pedestrian', 'trafficcone', 'trailer', 'truck',
            'driveable_surface', 'other', 'sidewalk', 'terrain', 'mannade',
            'vegetation'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=True,
            use_map=False,
            use_external=True),
        queue_length=2,
        future_length=4,
        ego_mask=(-0.8, -1.5, 0.8, 2.5),
        plan_grid_conf=dict(
            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0]),
        radar_cfg=None,
        radar_observation_cfg=dict(
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            bev_h=200,
            bev_w=200,
            nsweeps=1,
            max_returns=4096,
            max_time_lag=0.15,
            cache_dir=
            '/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/cache/radar_m3_obs_trainval_1sweep_200x200_v1',
            cache_readonly=True)),
    shuffler_sampler=dict(
        type='VirtualDistributedGroupSampler', reference_world_size=4),
    nonshuffler_sampler=dict(type='DistributedSampler'))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(
    type='EquivalentCumulativeOptimizerHook',
    cumulative_iters=2,
    reference_world_size=4,
    expected_optimizer_steps_per_epoch=5983,
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='EquivalentCosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001,
    cumulative_iters=2,
    by_epoch=True)
total_epochs = 24
evaluation = dict(
    interval=1,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='LoadOccupancy',
            to_float32=True,
            occ_path=
            '/storage/data/metaiot_data/huayiming/RadarFlowOcc/datasets/openoccupancy/nuScenes-Occupancy-v0.1',
            grid_size=[512, 512, 40],
            unoccupied=0,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            use_fine_occ=True,
            use_separate_classes=False,
            use_background_classes=False,
            time_history_field=2,
            time_future_field=4,
            test_mode=True),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'barrier', 'bicycle', 'bus', 'car', 'construction',
                'motorcycle', 'pedestrian', 'trafficcone', 'trailer', 'truck',
                'driveable_surface', 'other', 'sidewalk', 'terrain', 'mannade',
                'vegetation'
            ]),
        dict(
            type='CustomCollect3D',
            keys=[
                'img', 'gt_occ', 'vel_steering', 'sdc_planning',
                'sdc_planning_mask', 'command', 'sample_traj',
                'segmentation_bev'
            ])
    ])
runner = dict(type='EpochBasedRunner', max_epochs=24)
custom_hooks = [dict(type='SetEpochInfoHook')]
radar_observation_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    bev_h=200,
    bev_w=200,
    nsweeps=1,
    max_returns=4096,
    max_time_lag=0.15,
    cache_dir=
    '/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/cache/radar_m3_obs_trainval_1sweep_200x200_v1',
    cache_readonly=True)
seed = 0
gpu_ids = range(0, 2)
fp16 = None
tf32_policy = dict(matmul=True, cudnn=True)
close_tf32 = False
cudnn_benchmark = False
m3_h200_requires_resolution = False
m3_protocol = dict(
    status='HOST_RUNTIME_GPU_SMOKE',
    reference_config=
    '/home/wangning/Workspace/RadarFlowOcc-m3/DOCS/m3_h200_20260907/l40s_resolved_reference.py',
    initialization='same FCOS3D pretrained artifact as L40S; start epoch 1',
    reference_checkpoint_bytes=225215819,
    physical_world_size=2,
    physical_microbatch=1,
    cumulative_iters=2,
    effective_global_batch=4,
    optimizer_updates_per_epoch=5983,
    total_optimizer_updates=143592,
    raw_microsteps_per_epoch=11966,
    warmup_optimizer_updates=500,
    epochs=24,
    seed=0,
    train_image_resize=[720],
    validation_during_training=False,
    precision='FP32 with TF32 matmul and cuDNN enabled, as observed on L40S',
    ego_condition='given future ground-truth ego transforms and actions',
    method_deltas=[
        'M3 one sweep versus M1 five sweeps',
        'per-return posterior, held-out NLL and sigma-point transport'
    ],
    equivalence=
    'shared optimization/data protocol and effective batch; not bitwise reproduction',
    checkpoint_iter_unit='raw_microstep',
    lr_warmup_unit='optimizer_update',
    radar_cache=dict(
        directory=
        '/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/cache/radar_m3_obs_trainval_1sweep_200x200_v1',
        status_file=
        '/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/cache/radar_m3_obs_trainval_1sweep_200x200_v1/build_status.json',
        status='COMPLETE',
        tokens_processed=34149,
        tokens_required=34149,
        covered_splits=['train', 'val'],
        train_val_test_readonly=True,
        validation=
        'build record and production manifest; per-entry validation at read time'
    ))
