import sys
sys.path.append('/home/aistudio/')
import argparse
import numpy as np 

import paddle.fluid as F
import paddle.fluid.dygraph as dg

from densepose.config.config import get_cfg
from densepose.modeling.build import build_model
from densepose.modeling.backbone import build_backbone


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config-file", default="/home/aistudio/retinanet/config/Base-RetinaNet.yaml",
    #     metavar="FILE", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line",
        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_cfg()
    # cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    with dg.guard():
        # resnet = build_backbone(cfg, )

        # for k, v in resnet.state_dict().items():
        #     print(k + ": " + str(v.shape))

        # image = dg.to_variable(np.random.rand(4, 3, 256, 256).astype('float32'))
        # features = resnet(image)
        # if isinstance(features, dict):
        #     for k, v in features.items():
        #         print(k + ": " + str(v.shape))
        # else:
        #     for f in features:
        #         print(f.shape)
        #         # [4, 256, 4, 4]
        #         # [4, 256, 8, 8]
        #         # [4, 256, 16, 16]
        #         # [4, 256, 32, 32]
        #         # [4, 256, 64, 64]
        densepose = build_model(cfg)
        for k, v in densepose.state_dict().items():
            print(k + ": " + str(v.shape))
        
        state_dict, _ = F.load_dygraph('/home/aistudio/densepose/pretrained_models/DensePose_ResNet101_FPN_32x8d_s1x-e2e.pdparams')
        densepose.load_dict(state_dict)

        image = dg.to_variable(np.random.rand(4, 3, 256, 256).astype("float32"))
        im_info = np.array([[256, 256, 1],
                            [256, 256, 1],
                            [256, 256, 1],
                            [256, 256, 1]
                            ]).astype("float32")
        results = densepose(image, im_info, im_ori_shape=(256, 256, 3))
        for k, v in results.items():
            if not isinstance(v, list):
                print(k + ": " + str(v.shape))
        print(results['lod'])

