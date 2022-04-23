import argparse
import os
from pathlib import Path
from functools import partial
import cv2
import numpy as np

import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '/'))
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.cv_core import (Config, load_checkpoint, FeatureMapVis, show_tensor, imdenormalize, show_img, imwrite,
                           traverse_file_paths)
from mmdet.models import build_detector
from mmdet.datasets.builder import build_dataset
from mmdet.datasets.pipelines import Compose
import torch
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img_dir', type=str, default='/home/shm/data/coco/val2017', help='show img dir')
    parser.add_argument('--show', type=bool, default=False, help='show results')
    parser.add_argument(
        '--output_dir', help='directory where painted images will be saved')
    args = parser.parse_args()
    return args


def forward(self, img, img_metas=None, return_loss=False, **kwargs):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    return outs


def create_model(cfg, use_gpu=True):
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    if use_gpu:
        model = model.cuda()
    return model


def create_featuremap_vis(cfg, use_gpu=True, init_shape=(320, 320, 3)):
    model = create_model(cfg, use_gpu)
    model.forward = partial(forward, model)
    featurevis = FeatureMapVis(model, use_gpu)
    featurevis.set_hook_style(init_shape[2], init_shape[:2])
    return featurevis


def _show_save_data(featurevis, img, img_orig, feature_indexs, filepath, is_show, output_dir,item=None,dataset=None):
    show_datas = []
    for feature_index in feature_indexs:
        feature_map = featurevis.run(img.copy(), feature_index=feature_index)[0]
        data = show_tensor(feature_map[0], resize_hw=img.shape[:2], show_split=False, is_show=False)[0]
        am_data = cv2.addWeighted(data, 0.5, img_orig, 0.5, 0)
        if item is not None:
          am_data=imshow_det_bboxes(
              am_data,
              item['gt_bboxes'].data,
              item['gt_labels'].data,
              None,
              class_names=dataset.CLASSES,
              show=False)
        show_datas.append(am_data)
    if is_show:
        show_img(show_datas)
    if output_dir is not None:
        filename = os.path.join(output_dir,
                                Path(filepath).name
                                )
        if len(show_datas) == 1:
            imwrite(show_datas[0], filename)
        else:
            for i in range(len(show_datas)):
                fname, suffix = os.path.splitext(filename)
                imwrite(show_datas[i], fname + '_{}'.format(str(i)) + suffix)


def show_featuremap_from_imgs(featurevis, feature_indexs, img_dir, mean, std, is_show, output_dir):
    if not isinstance(feature_indexs, (list, tuple)):
        feature_indexs = [feature_indexs]
    img_paths = traverse_file_paths(img_dir, 'jpg')
    for path in img_paths:
        data = dict(img_info=dict(filename=path), img_prefix=None)
        test_pipeline = Compose(cfg.data.test.pipeline)
        item = test_pipeline(data)
        img_tensor = item['img']
        img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)  
        img_orig = imdenormalize(img, np.array(mean), np.array(std)).astype(np.uint8)
        _show_save_data(featurevis, img, img_orig, feature_indexs, path, is_show, output_dir)


def show_featuremap_from_datalayer(featurevis, feature_indexs, is_show, output_dir):
    if not isinstance(feature_indexs, (list, tuple)):
        feature_indexs = [feature_indexs]
    dataset = build_dataset(cfg.data.test)
    for item in dataset:
        img_tensor = item['img'].data
        img_metas = item['img_metas'].data
        filename = img_metas['filename']
        img_norm_cfg = img_metas['img_norm_cfg']
        img = img_tensor.cpu().numpy().transpose(1, 2, 0) 
        img_orig = imdenormalize(img, img_norm_cfg['mean'], img_norm_cfg['std']).astype(np.uint8)
        _show_save_data(featurevis, img, img_orig, feature_indexs, filename, is_show, output_dir,item,dataset)


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    use_gpu = False
    is_show = args.show
    print(is_show)
    init_shape = (320, 320, 3)  
    #feature_index = [625,265,536]  
    #feature_index = [669,672,675]  
    #feature_index = [755,765,775] #head
    #feature_index = [655, 277,562] #shuffle out
    #feature_index = [697,686,738,749] #FPN -Attention
    #feature_index = [669,672,675,707,717,727,759,769,779] #FPN lateral connection
    #feature_index = [669,672,675,697,686,738,749] #FPN lateral connection
    #feature_index = [669,672,675,697,686,738,749,707,717,727,759,769,779]
    #feature_index=[669,672,675,686,698,740,752,763,773,783]
    
    #feature_index=[295,209,302]
    #feature_index=[584,595,499,510]
    #feature_index=[42,121,162]
    #feature_index=[23,47,79,103,127,135]
    #feature_index=[138]
    #feature_index=[163,164,165]
    #feature_index=[643,651,659]
    #feature_index=[515,217,442]
    feature_index=[530,531,532]
    featurevis = create_featuremap_vis(cfg, use_gpu, init_shape)

    mean = cfg.img_norm_cfg['mean']
    std = cfg.img_norm_cfg['std']
    show_featuremap_from_imgs(featurevis, feature_index, args.img_dir, mean, std, False, args.output_dir)
    #show_featuremap_from_datalayer(featurevis, feature_index,False,args.output_dir)
