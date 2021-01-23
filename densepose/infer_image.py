import argparse
import numpy as np
import sys
sys.path.append('/home/aistudio')
import matplotlib
matplotlib.use('Agg') 

import paddle.fluid.dygraph as dg
from densepose.utils.image_reader import read_image
from densepose.modeling.build import build_model
from densepose.config.config import get_cfg
from densepose.utils.visualize import vis_one_image


def main(args):
    im, im_info, im_ori, im_ori_shape = read_image(args.im_path)
    cfg = get_cfg()
    model = build_model(cfg)
    
    image = dg.to_variable(im)
    output = model(image, im_info, im_ori_shape=im_ori_shape)

    vis_one_image(im_ori[:, :, ::-1], args.im_path.split('/')[-1], '/home/aistudio/densepose/outputs', output['cls_boxes'], output['cls_bodys'], ext='jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='densepose inference')
    parser.add_argument('--im_path', type=str, default='/home/aistudio/densepose/images/demo_im.jpg', help='Image Path.')
    args = parser.parse_args()
    main(args)
