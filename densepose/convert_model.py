import pickle
with open('/home/aistudio/densepose/pretrained_models/DensePose_ResNet101_FPN_32x8d_s1x-e2e.pkl', 'rb+') as f:
     model_weights = pickle.load(f, encoding='iso-8859-1')



weight_dict = model_weights['blobs']

rename_dict = {
    'conv1_w': "backbone.bottom_up.stem.conv1.weight",
    'conv1_b': "backbone.bottom_up.stem.conv1.bias",
    'res_conv1_bn_s': "backbone.bottom_up.stem.conv1_norm.scale",
    'res_conv1_bn_b': "backbone.bottom_up.stem.conv1_norm.bias",
    'conv_rpn_fpn2_w': "proposal_generator.rpn_head.conv.weight",
    'conv_rpn_fpn2_b': "proposal_generator.rpn_head.conv.bias",
    'rpn_bbox_pred_fpn2_w': "proposal_generator.rpn_head.anchor_deltas.weight",
    'rpn_bbox_pred_fpn2_b': "proposal_generator.rpn_head.anchor_deltas.bias",
    'rpn_cls_logits_fpn2_w': "proposal_generator.rpn_head.objectness_logits.weight",
    'rpn_cls_logits_fpn2_b': "proposal_generator.rpn_head.objectness_logits.bias",
    'AnnIndex_lowres_w': "roi_body_uv_heads.AnnIndex_lowres.weight",
    'AnnIndex_lowres_b': "roi_body_uv_heads.AnnIndex_lowres.bias",
    'AnnIndex_w': "roi_body_uv_heads.AnnIndex.weight",
    'AnnIndex_b': "roi_body_uv_heads.AnnIndex.bias",
    'Index_UV_lowres_w': "roi_body_uv_heads.Index_UV_lowres.weight",
    'Index_UV_lowres_b': "roi_body_uv_heads.Index_UV_lowres.bias",
    'Index_UV_w': "roi_body_uv_heads.Index_UV.weight",
    'Index_UV_b': "roi_body_uv_heads.Index_UV.bias",
    'U_lowres_w': "roi_body_uv_heads.U_lowres.weight",
    'U_lowres_b': "roi_body_uv_heads.U_lowres.bias",
    'U_estimated_w': "roi_body_uv_heads.U_estimated.weight",
    'U_estimated_b': "roi_body_uv_heads.U_estimated.bias",
    'V_lowres_w': "roi_body_uv_heads.V_lowres.weight",
    'V_lowres_b': "roi_body_uv_heads.V_lowres.bias",
    'V_estimated_w': "roi_body_uv_heads.V_estimated.weight",
    'V_estimated_b': "roi_body_uv_heads.V_estimated.bias",
    'b': 'bias',
    'w': 'weight',
    's': 'scale',
}

new_weight_dict = {}
for k, v in weight_dict.items():
    if 'momentum' in k:
        pass

    elif k in rename_dict:
        new_k = rename_dict[k]
        new_weight_dict[new_k] = v

    elif 'res' in k and 'fpn' not in k:
        header = ['backbone', 'bottom_up']
        if 'bn' in k:
            stage, sub_block, branch, _, is_bias = k.split('_')
            branch += '_norm'
        else:
            stage, sub_block, branch, is_bias = k.split('_')
        
        newname = '.'.join(header + [stage, sub_block, branch, rename_dict[is_bias]])
        new_weight_dict[newname] = v
    
    elif 'fpn' in k:
        if 'inner' in k:
            if len(k.split('_')) == 6:
                _, _, layer, _, _, is_bias = k.split('_')
            else:
                _, _, layer, _, _, _, is_bias = k.split('_')
            layer = 'fpn_lateral' + layer[-1]
            new_k = '.'.join(['backbone', layer, rename_dict[is_bias]])
            new_weight_dict[new_k] = v
        else:
            _, layer, _, _, is_bias = k.split('_')
            layer = 'fpn_output' + layer[-1]
            new_k = '.'.join(['backbone', layer, rename_dict[is_bias]])
            new_weight_dict[new_k] = v

    elif 'fc' in k and '1000' not in k and 'fcn' not in k:
        fc, is_bias = k.split('_')
        new_k = '.'.join(['roi_fast_rcnn_heads', fc, rename_dict[is_bias]])

        if is_bias == 'w':
            new_weight_dict[new_k] = v.transpose((1, 0))
        else:
            new_weight_dict[new_k] = v
    
    elif k == 'cls_score_w':
        new_weight_dict['roi_fast_rcnn_heads.cls_score.weight'] = v.transpose((1, 0))
    elif k == 'cls_score_b':
        new_weight_dict['roi_fast_rcnn_heads.cls_score.bias'] = v
    elif k == 'bbox_pred_w':
        new_weight_dict['roi_fast_rcnn_heads.bbox_pred.weight'] = v.transpose((1, 0))
    elif k == 'bbox_pred_b':
        new_weight_dict['roi_fast_rcnn_heads.bbox_pred.bias'] = v
    
    elif 'body_conv' in k:
        is_bias = k.split('_')[-1]
        new_k = '.'.join(['roi_body_uv_heads', k[:-2], rename_dict[is_bias]])
        new_weight_dict[new_k] = v
    else:
        print(k)


import sys
sys.path.append('/home/aistudio/')
import numpy as np 

import paddle.fluid.dygraph as dg

from densepose.config.config import get_cfg
from densepose.modeling.build import build_model


cfg = get_cfg()

densepose = build_model(cfg)

densepose.load_dict(new_weight_dict)

import paddle.fluid.dygraph as dg 
dg.save_dygraph(densepose.state_dict(), '/home/aistudio/densepose/pretrained_models/DensePose_ResNet101_FPN_32x8d_s1x-e2e')
