# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import os
import time
import sys
sys.path.append('PaddleDetection')
import json
import yaml
from functools import reduce
import multiprocessing

from PIL import Image
import cv2
import numpy as np
import paddle
# import paddleseg.transforms as T
from paddle.inference import Config
from paddle.inference import create_predictor
# from paddleseg.cvlibs import manager


from deploy.python.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
from deploy.python.picodet_postprocess import PicoDetPostProcess
from deploy.python.utils import argsparser, Timer, get_current_memory_mb
from deploy.python.visualize import visualize_box_mask


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.tracker = None
        if 'tracker' in yml_conf:
            self.tracker = yml_conf['tracker']
        if 'NMS' in yml_conf:
            self.nms = yml_conf['NMS']
        if 'fpn_stride' in yml_conf:
            self.fpn_stride = yml_conf['fpn_stride']
        self.print_config()

    def print_config(self):
        print('%s: %s' % ('Model Arch', self.arch))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))


def get_test_images(infer_file):
    with open(infer_file, 'r') as f:
        dirs = f.readlines()
    images = []
    for dir in dirs:
        images.append(eval(repr(dir.replace('\n',''))).replace('\\', '/'))
    assert len(images) > 0, "no image found in {}".format(infer_file)
    return images

def load_predictor(model_dir):
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    # initial GPU memory(M), device ID
    config.enable_use_gpu(2000, 0)
    # optimize graph and fuse op
    config.switch_ir_optim(True)
    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, config



def create_inputs(imgs, im_info):
    inputs = {}

    im_shape = []
    scale_factor = []
    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    origin_scale_factor = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    padding_imgs_shape = []
    padding_imgs_scale = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = np.array(img, dtype=np.float32)
        padding_imgs.append(padding_im)
        padding_imgs_shape.append(
            np.array([max_shape_h, max_shape_w]).astype('float32'))
        rescale = [float(max_shape_h) / float(im_h), float(max_shape_w) / float(im_w)]
        padding_imgs_scale.append(np.array(rescale).astype('float32'))
    inputs['image'] = np.stack(padding_imgs, axis=0)
    inputs['im_shape'] = np.stack(padding_imgs_shape, axis=0)
    inputs['scale_factor'] = origin_scale_factor
    return inputs


def visualize(image_list, results, labels, output_dir='output/', threshold=0.5):
    # visualize the predict result
    start_idx = 0
    for idx, image_file in enumerate(image_list):
        im_bboxes_num = results['boxes_num'][idx]
        im_results = {}
        if 'boxes' in results:
            im_results['boxes'] = results['boxes'][start_idx:start_idx +
                                                   im_bboxes_num, :]
        if 'masks' in results:
            im_results['masks'] = results['masks'][start_idx:start_idx +
                                                   im_bboxes_num, :]
        if 'segm' in results:
            im_results['segm'] = results['segm'][start_idx:start_idx +
                                                 im_bboxes_num, :]
        if 'label' in results:
            im_results['label'] = results['label'][start_idx:start_idx +
                                                   im_bboxes_num]
        if 'score' in results:
            im_results['score'] = results['score'][start_idx:start_idx +
                                                   im_bboxes_num]

        start_idx += im_bboxes_num
        im = visualize_box_mask(
            image_file, im_results, labels, threshold=threshold)
        img_name = os.path.split(image_file)[-1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = os.path.join(output_dir, img_name)
        im.save(out_path, quality=95)
        print("save result to: " + out_path)

class Detector(object):

    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(model_dir)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.preprocess_ops = self.get_ops()

    def get_ops(self):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            print(op_type)
            preprocess_ops.append(eval(op_type)(**new_op_info))
        return preprocess_ops

    def postprocess(self,
                    np_boxes,
                    np_masks,
                    inputs,
                    np_boxes_num,
                    threshold=0.5):
        # postprocess output of predictor
        results = {}
        results['boxes'] = np_boxes
        results['boxes_num'] = np_boxes_num
        if np_masks is not None:
            results['masks'] = np_masks
        return results

    def predict(self, inputs):
        # preprocess
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        np_score_list, np_boxes_list = [], []

        # model_prediction
        # self.predictor.run()
        # np_score_list.clear()
        # np_boxes_list.clear()
        # output_names = self.predictor.get_output_names()
        # num_outs = int(len(output_names) / 2)

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        boxes_num = self.predictor.get_output_handle(output_names[1])
        np_boxes_num = boxes_num.copy_to_cpu()


        # for out_idx in range(num_outs):
        #     np_score_list.append(
        #         self.predictor.get_output_handle(output_names[out_idx])
        #         .copy_to_cpu())
        #     np_boxes_list.append(
        #         self.predictor.get_output_handle(output_names[
        #             out_idx + num_outs]).copy_to_cpu())

        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            results = {'boxes': np.zeros([0, 6]), 'boxes_num': [0]}
        else:
            np_masks = None
            results = self.postprocess(
                np_boxes, np_masks, inputs, np_boxes_num, threshold=threshold)

        return results

        #postprocess
        # self.postprocess = PicoDetPostProcess(
        #     inputs['image'].shape[2:],
        #     inputs['im_shape'],
        #     inputs['scale_factor'],
        #     strides=self.pred_config.fpn_stride,
        #     nms_threshold=self.pred_config.nms['nms_threshold'])
        # np_boxes, np_boxes_num = self.postprocess(np_score_list, np_boxes_list)

        # return dict(boxes=np_boxes, boxes_num=np_boxes_num)




def predict_image(detector, image_list, result_path, threshold):
    c_results = {"result": []}

    for index in range(len(image_list)):
        # 检测模型图像预处理
        input_im_lst = []
        input_im_info_lst = []

        im_path = image_list[index]
        im, im_info = preprocess(im_path, detector.preprocess_ops)


        input_im_lst.append(im)
        input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)

        image_id = int(os.path.basename(im_path).split('.')[0])


        # 检测模型预测结果

        det_results = detector.predict(inputs)
        # visualize(
        #         [im_path],
        #         det_results,
        #         detector.pred_config.labels,
        #         output_dir='out_put',
        #         threshold=0.5)
        # 检测模型写结果
        im_bboxes_num = det_results['boxes_num'][0]
        if im_bboxes_num > 0:
            bbox_results = det_results['boxes'][0:im_bboxes_num, 2:]
            id_results = det_results['boxes'][0:im_bboxes_num, 0]
            score_results = det_results['boxes'][0:im_bboxes_num, 1]
            for idx in range(im_bboxes_num):
                if float(score_results[idx]) >= threshold:
                    c_results["result"].append({"image_id": image_id,
                                                "type": int(id_results[idx]) + 1,
                                                "x": float(bbox_results[idx][0]),
                                                "y": float(bbox_results[idx][1]),
                                                "width": float(bbox_results[idx][2]) - float(bbox_results[idx][0]),
                                                "height": float(bbox_results[idx][3]) - float(bbox_results[idx][1]),
                                                "segmentation": []})


    # 写文件
    with open(result_path, 'w') as ft:
        json.dump(c_results, ft)

def main(infer_txt, result_path, det_model_path, threshold):
    pred_config = PredictConfig(det_model_path)
    detector = Detector(pred_config, det_model_path)

    # predict from image
    img_list = get_test_images(infer_txt)
    predict_image(detector, img_list, result_path, threshold)

if __name__ == '__main__':
    start_time = time.time()
    det_model_path = "model/ppyoloe_test"

    threshold = 0.4

    paddle.enable_static()
    infer_txt = sys.argv[1]
    result_path = sys.argv[2]

    main(infer_txt, result_path, det_model_path, threshold)
    print('total time:', time.time() - start_time)
