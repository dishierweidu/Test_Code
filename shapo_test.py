import argparse
import pathlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt
import os
import time
import pytorch_lightning as pl
import _pickle as cPickle
import os, sys
sys.path.append('shapo')
from simnet.lib.net import common
from simnet.lib import camera
from simnet.lib.net.panoptic_trainer import PanopticModel
from utils.nocs_utils import load_img_NOCS, create_input_norm
from utils.viz_utils import depth2inv, viz_inv_depth
from utils.transform_utils import get_gt_pointclouds, transform_coordinates_3d, calculate_2d_projections
from utils.transform_utils import project, get_pc_absposes, transform_pcd_to_canonical
from utils.viz_utils import save_projected_points, draw_bboxes, line_set_mesh, display_gird, draw_geometries, show_projected_points
from sdf_latent_codes.get_surface_pointcloud import get_surface_pointclouds_octgrid_viz, get_surface_pointclouds
from sdf_latent_codes.get_rgb import get_rgbnet, get_rgb_from_rgbnet

import pyrealsense2 as rs

if __name__ == '__main__':
    sys.argv = ['', '@/home/dishierweidu/Desktop/Grasp/shapo/configs/net_config.txt']
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    common.add_train_args(parser)
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--app_output', default='inference', type=str)
    app_group.add_argument('--result_name', default='shapo_inference', type=str)
    app_group.add_argument('--data_dir', default='/home/dishierweidu/Desktop/Grasp/shapo/test_data', type=str)

    hparams = parser.parse_args()
    min_confidence = 0.50
    use_gpu=True
    hparams.checkpoint = '/home/dishierweidu/Desktop/Grasp/shapo/ckpts/shapo_real.ckpt'
    model = PanopticModel(hparams, 0, None, None)
    model.eval()
    if use_gpu:
        model.cuda()
    data_path = open(os.path.join(hparams.data_dir, 'Real', 'test_list_subset.txt')).read().splitlines()
    _CAMERA = camera.NOCS_Real()
    sdf_pretrained_dir = os.path.join(hparams.data_dir, 'sdf_rgb_pretrained')
    rgb_model_dir = os.path.join(hparams.data_dir, 'sdf_rgb_pretrained', 'rgb_net_weights')

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: color and depth
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Display the color and depth images
            cv2.imshow('Color Image', color_image)
            cv2.imshow('Depth Image', depth_image)

            # Save color and depth images when Ctrl + S is pressed
            key = cv2.waitKey(1)
            
            #num from 0 to 3 (small subset of data)
            # img_full_path = os.path.join(hparams.data_dir, 'Real', data_path[num])
            # print(img_full_path)
            # img_vis = cv2.imread(img_full_path + '_color.png')
            # img_vis = cv2.resize(img_vis, (640, 480))

            # left_linear, depth, actual_depth = load_img_NOCS(img_full_path + '_color.png' , img_full_path + '_depth.npy')
            left_linear = color_image
            depth = depth_image
            actual_depth = depth_image.astype(np.uint16)
            input = create_input_norm(left_linear, depth)[None, :, :, :]

            if use_gpu:
                input = input.to(torch.device('cuda:0'))

            with torch.no_grad():
                seg_output, _, _ , pose_output = model.forward(input)
                _, _, _ , pose_output = model.forward(input)
                shape_emb_outputs, appearance_emb_outputs, abs_pose_outputs, peak_output, scores_out, output_indices = pose_output.compute_shape_pose_and_appearance(min_confidence,is_target = False)

            # display_gird(img_vis, depth, peak_output)

            rotated_pcds = []
            points_2d = []
            box_obb = []
            axes = []
            lod = 3 # Choose from LOD 3-7 here, going higher means more memory and finer details

            # Here we visualize the output of our network
            for j in range(len(shape_emb_outputs)):
                shape_emb = shape_emb_outputs[j]
                # appearance_emb = appearance_emb_putputs[j]
                appearance_emb = appearance_emb_outputs[j]
                is_oct_grid = True
                if is_oct_grid:
                    # pcd_dsdf_actual = get_surface_pointclouds_octgrid_sparse(shape_emb, sdf_latent_code_dir = sdf_pretrained_dir, lods=[2,3,4,5,6])
                    pcd_dsdf, nrm_dsdf = get_surface_pointclouds_octgrid_viz(shape_emb, lod=lod, sdf_latent_code_dir=sdf_pretrained_dir)
                else:
                    pcd_dsdf = get_surface_pointclouds(shape_emb)

                rgbnet = get_rgbnet(rgb_model_dir)
                pred_rgb = get_rgb_from_rgbnet(shape_emb, pcd_dsdf, appearance_emb, rgbnet)
                rotated_pc, rotated_box, _ = get_pc_absposes(abs_pose_outputs[j], pcd_dsdf)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.copy(rotated_pc))
                pcd.colors = o3d.utility.Vector3dVector(pred_rgb.detach().cpu().numpy())
                pcd.normals = o3d.utility.Vector3dVector(nrm_dsdf)
                rotated_pcds.append(pcd)

                cylinder_segments = line_set_mesh(rotated_box)
                # draw 3D bounding boxes around the object
                for k in range(len(cylinder_segments)):
                    rotated_pcds.append(cylinder_segments[k])

                # draw 3D coordinate frames around each object
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                T = abs_pose_outputs[j].camera_T_object
                mesh_t = mesh_frame.transform(T)
                rotated_pcds.append(mesh_t)

                points_mesh = camera.convert_points_to_homopoints(rotated_pc.T)
                points_2d.append(project(_CAMERA.K_matrix, points_mesh).T)
                #2D output
                points_obb = camera.convert_points_to_homopoints(np.array(rotated_box).T)
                box_obb.append(project(_CAMERA.K_matrix, points_obb).T)
                xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                sRT = abs_pose_outputs[j].camera_T_object @ abs_pose_outputs[j].scale_matrix
                transformed_axes = transform_coordinates_3d(xyz_axis, sRT)
                axes.append(calculate_2d_projections(transformed_axes, _CAMERA.K_matrix[:3,:3]))
            # draw_geometries(rotated_pcds)
    except KeyboardInterrupt:
        pipeline.stop()
        
    except Exception as e:
        print(e)
        pipeline.stop()
        
    # finally:
    #     # Stop streaming
    #     pipeline.stop()

    # Close all windows
    cv2.destroyAllWindows()

    
