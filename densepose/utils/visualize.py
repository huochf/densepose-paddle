# -------------------------------------------------------------------
# Modified from densepose(https://github.com/facebookresearch/DensePose)
# -------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -------------------------------------------------------------------
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def plt_detect_result(im, boxes, thresh=0.9, kp_thresh=2, dpi=300, box_alpha=0.8, show_class=True):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        
        # show box (off by default)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                fill=False, edgecolor='g', linewidth=0.5, alpha=box_alpha)
        )

        if show_class:
            ax.text(bbox[0], bbox[1] - 2, 'person', fontsize=3, family='serif',
                bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')
    
    return fig


def plot_bodypose(im, boxes, body_uv, thresh=0.9, kp_thresh=2, dpi=300,):
    #   DensePose Visualization Starts !!
    ##  Get full IUV image out
    IUV_fields = body_uv
    #
    All_Coords = np.zeros(im.shape)
    All_inds = np.zeros([im.shape[0], im.shape[1]])
    K = 26
    ##
    inds = np.argsort(boxes[:, 4])

    ##
    for i, ind in enumerate(inds):
        entry = boxes[ind, :]
        if entry[4] > 0.65:
            entry = entry[0:4].astype(int)
            #####
            output = IUV_fields[ind]
            #####
            All_Coords_Old = All_Coords[entry[1] : entry[1] + output.shape[1], entry[0] : entry[0] + output.shape[2], :]
            All_Coords_Old[All_Coords_Old == 0] = output.transpose([1, 2, 0])[All_Coords_Old==0]
            All_Coords[entry[1] : entry[1] + output.shape[1], entry[0] : entry[0] + output.shape[2], :] = All_Coords_Old
            ###
            CurrentMask = (output[0, :, :] > 0).astype(np.float32)
            All_inds_old = All_inds[entry[1] : entry[1] + output.shape[1], entry[0] : entry[0] + output.shape[2]]
            All_inds_old[All_inds_old==0] = CurrentMask[All_inds_old==0] * i
            All_inds[entry[1]: entry[1] + output.shape[1], entry[0] : entry[0] + output.shape[2]] = All_inds_old

    #
    All_Coords[:, :, 1:3] = 255. * All_Coords[:, :, 1:3]
    All_Coords[All_Coords > 255] = 255.
    All_Coords = All_Coords.astype(np.uint8)
    All_inds = All_inds.astype(np.uint8)

    return All_Coords, All_inds


def vis_one_image(im, im_name, output_dir, boxes, body_uv, thresh=0.9, 
    kp_thresh=2, dpi=200, box_alpha=0.8, show_class=True, ext='pdf'):
    """ Visual debugging of detections. """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset_keypoints, _ = get_keypoints()

    color_list = colormap(rgb=True) / 255

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        
        # show box (off by default)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                fill=False, edgecolor='g', linewidth=0.5, alpha=box_alpha)
        )

        if show_class:
            ax.text(bbox[0], bbox[1] - 2, 'person', fontsize=3, family='serif',
                bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')
        
    #   DensePose Visualization Starts !!
    ##  Get full IUV image out
    IUV_fields = body_uv
    #
    All_Coords = np.zeros(im.shape)
    All_inds = np.zeros([im.shape[0], im.shape[1]])
    K = 26
    ##
    inds = np.argsort(boxes[:, 4])

    ##
    for i, ind in enumerate(inds):
        entry = boxes[ind, :]
        if entry[4] > 0.65:
            entry = entry[0:4].astype(int)
            #####
            output = IUV_fields[ind]
            #####
            All_Coords_Old = All_Coords[entry[1] : entry[1] + output.shape[1], entry[0] : entry[0] + output.shape[2], :]
            All_Coords_Old[All_Coords_Old == 0] = output.transpose([1, 2, 0])[All_Coords_Old==0]
            All_Coords[entry[1] : entry[1] + output.shape[1], entry[0] : entry[0] + output.shape[2], :] = All_Coords_Old
            ###
            CurrentMask = (output[0, :, :] > 0).astype(np.float32)
            All_inds_old = All_inds[entry[1] : entry[1] + output.shape[1], entry[0] : entry[0] + output.shape[2]]
            All_inds_old[All_inds_old==0] = CurrentMask[All_inds_old==0] * i
            All_inds[entry[1]: entry[1] + output.shape[1], entry[0] : entry[0] + output.shape[2]] = All_inds_old

    #
    All_Coords[:, :, 1:3] = 255. * All_Coords[:, :, 1:3]
    All_Coords[All_Coords > 255] = 255.
    All_Coords = All_Coords.astype(np.uint8)
    All_inds = All_inds.astype(np.uint8)
    #
    IUV_SaveName = os.path.basename(im_name).split('.')[0] + '_IUV.png'
    INDS_SaveName = os.path.basename(im_name).split('.')[0] + '_INDS.png'
    cv2.imwrite(os.path.join(output_dir, '{}'.format(IUV_SaveName)), All_Coords)
    cv2.imwrite(os.path.join(output_dir, '{}'.format(INDS_SaveName)), All_inds)
    print("IUV written to: ", os.path.join(output_dir, '{}'.format(IUV_SaveName)))
    ###
    ### DensePose Visualization Dowe !!
    #
    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    plt.close('all')



def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    return keypoints, keypoint_flip_map



def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list























