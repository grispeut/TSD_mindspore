import sys
import os
print('================')
print(os.getcwd())
print('================')
from mindspore import context
devid = int(os.getenv('DEVICE_ID', '0'))
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)

import cv2
import time
import numpy as np
from PIL import Image
from collections import defaultdict

from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore as ms

from src.yolo import YOLOV4CspDarkNet53
from src.transforms import statistic_normalize_img
from src.ensemble_boxes_wbf import weighted_boxes_fusion

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from model_service.model_service import SingleNodeService
    Image.MAX_IMAGE_PIXELS = 1000000000000000
    offline_flag = False
except:
    offline_flag = True
    SingleNodeService = object
    pass

label_mapping = {
    '0': 'red_stop',
    '1': 'green_go',
    '2': 'yellow_back',
    '3': 'pedestrian_crossing',
    '4': 'speed_limited',
    '5': 'speed_unlimited'
}

def letterbox(img, new_shape=(640, 640), color=(128, 128, 128), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2
    return boxes

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords = clip_coords(coords, img0_shape)
    return coords


class YOLO_service(SingleNodeService):
    def __init__(self, model_name, model_path='.', mn_list=['stage1.ckpt','stage2.ckpt']):
        self.model_name = model_name
        self.model_path_list = []
        self.model_list = []
        if offline_flag:
            print('================')
        else:
            logger.info('================')
        for i, model_name in enumerate(mn_list):
            mn_path = os.path.join(model_path, model_name)
            self.model_path_list.append(mn_path)
            if offline_flag:
                print(f'model_path:{mn_path}')
            else:
                logger.info(f'model_path:{mn_path}')
        if offline_flag:
            print('================')
        else:
            logger.info('================')

        self.img_size = [[416, 768],
                         [448, 800],
                         [480, 864],
                         [512, 928]]  # Net input  # Net input
        self.stride = 32
        self.conf_thre = 0.01
        self.nms_thre = 0.1
        self.labels = ['red_stop', 'green_go', 'yellow_back', 'pedestrian_crossing', 'speed_limited', 'speed_unlimited']

        for mn_path in self.model_path_list:
            model = self.load_model(mn_path)
            self.model_list.append(model)


    def load_model(self, model_path):
        model = YOLOV4CspDarkNet53(is_training=False)
        param_dict = load_checkpoint(model_path)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(model, param_dict_new)
        model.set_train(False)
        self.network_warmup(model)
        if offline_flag:
            print(f"load {model_path} successfully ! \n")
        else:
            logger.info(f"load {model_path} successfully ! \n")
        return model

    def network_warmup(self, model):
        logger.info("warmup network ... \n")
        for img_size in self.img_size:
            input_shape_h, input_shape_w = img_size
            input_shape = (input_shape_w, input_shape_h)
            input_shape = Tensor(input_shape, ms.float32)
            images = np.array(np.random.randn(1, 3, input_shape_h, input_shape_w), dtype=np.float32)
            inputs = Tensor(images, ms.float32)
            model(inputs, input_shape)
        logger.info("warmup network successfully ! \n")

    def _preprocess(self, img0, img_size):
        preprocessed_data = {}
        img = letterbox(img0, img_size, stride=self.stride, auto=True)[0]
        img = statistic_normalize_img(img, statistic_norm=True)
        # print(f'=========={time.time()-t0}')
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.concatenate([img, img, img], axis=-1)
        img = img[:, :, :].transpose(2, 0, 1).astype(np.float32)
        img = np.expand_dims(img, axis=0)  # BGR to RGB and HWC to CHW
        img = np.ascontiguousarray(img)
        preprocessed_data["input_img"] = Tensor(img)
        return (preprocessed_data, img0)

    def _diou_nms(self, dets, thresh=0.5):
        # conver xywh -> xmin ymin xmax ymax
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            center_x1 = (x1[i] + x2[i]) / 2
            center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
            center_y1 = (y1[i] + y2[i]) / 2
            center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
            inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
            out_max_x = np.maximum(x2[i], x2[order[1:]])
            out_max_y = np.maximum(y2[i], y2[order[1:]])
            out_min_x = np.minimum(x1[i], x1[order[1:]])
            out_min_y = np.minimum(y1[i], y1[order[1:]])
            outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
            diou = ovr - inter_diag / outer_diag
            diou = np.clip(diou, -1, 1)
            inds = np.where(diou <= thresh)[0]
            order = order[inds + 1]
        return keep

    def normal_nms(self, all_boxes, all_scores, thresh=0.5, max_boxes=100):
        """Apply NMS to bboxes."""
        x1 = all_boxes[:, 0]
        y1 = all_boxes[:, 1]
        x2 = all_boxes[:, 2]
        y2 = all_boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = all_scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if len(keep) >= max_boxes:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep


    def _inference(self, model, in_data, img1_shape):
        data, img0 = in_data
        img = data["input_img"]
        image_shape = img0.shape[:2]
        input_shape_h, input_shape_w = img1_shape
        input_shape = (input_shape_w, input_shape_h)
        input_shape = Tensor(input_shape, ms.float32)
        prediction = model(img, input_shape)
        output_big, output_me, output_small = prediction
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        outputs = [output_small, output_me, output_big]

        single_pic = {"detection_classes":[], "detection_boxes":[], "detection_scores":[]}
        results = defaultdict(list)
        for out_id in range(len(outputs)):
            out_item = outputs[out_id]
            out_item_single = out_item[0, :]
            # get number of items in one head, [B, gx, gy, anchors, 5+80]
            dimensions = out_item_single.shape[:-1]
            out_num = 1
            for d in dimensions:
                out_num *= d
            ori_h, ori_w = image_shape
            out_item_single[..., 0] = out_item_single[..., 0]*img1_shape[1]
            out_item_single[..., 1] = out_item_single[..., 1]*img1_shape[0]
            out_item_single[..., 2] = out_item_single[..., 2]*img1_shape[1]
            out_item_single[..., 3] = out_item_single[..., 3]*img1_shape[0]
            coords = out_item_single[..., 0:4].reshape(-1,4)
            boxes = coords.copy()
            boxes[:, 0] = coords[:, 0] - coords[:, 2]/2
            boxes[:, 1] = coords[:, 1] - coords[:, 3]/2
            boxes[:, 2] = coords[:, 0] + coords[:, 2]/2
            boxes[:, 3] = coords[:, 1] + coords[:, 3]/2
            boxes = scale_coords(img1_shape, boxes, (ori_h, ori_w))
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            x = boxes[:, 0] + w/2
            y = boxes[:, 1] + h/2
            conf = out_item_single[..., 4:5]
            cls_emb = out_item_single[..., 5:]
            cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
            x = x.reshape(-1)
            y = y.reshape(-1)
            w = w.reshape(-1)
            h = h.reshape(-1)
            cls_emb = cls_emb.reshape(-1, len(self.labels))
            conf = conf.reshape(-1)
            cls_argmax = cls_argmax.reshape(-1)

            idx = conf > self.conf_thre
            x = x[idx]
            y = y[idx]
            w = w[idx]
            h = h[idx]
            cls_emb = cls_emb[idx,:]
            conf = conf[idx]
            cls_argmax = cls_argmax[idx]

            x_top_left = x - w / 2.
            y_top_left = y - h / 2.
            # creat all False
            flag = np.random.random(cls_emb.shape) > sys.maxsize
            for i in range(flag.shape[0]):
                c = cls_argmax[i]
                flag[i, c] = True
            confidence = cls_emb[flag] * conf
            for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left, w, h, confidence, cls_argmax):
                if confi < self.conf_thre:
                    continue
                x_lefti = max(0, x_lefti)
                y_lefti = max(0, y_lefti)
                wi = min(wi, ori_w)
                hi = min(hi, ori_h)
                results[clsi].append([x_lefti, y_lefti, wi, hi, confi])
        for clsi in results:
            dets = results[clsi]
            dets = np.array(dets)
            keep_index = self._diou_nms(dets, thresh=self.nms_thre)
            dets_keep = dets[keep_index]
            if dets_keep.shape[0] > 0:
                for det in dets_keep:
                    x_lefti, y_lefti, wi, hi, confi = det
                    hw_box_coord = [float(x_lefti), float(y_lefti), float(x_lefti+wi), float(y_lefti+hi)]
                    score = float(confi)
                    label = int(clsi)
                    single_pic['detection_boxes'].append(hw_box_coord)
                    single_pic['detection_classes'].append(label)
                    single_pic['detection_scores'].append(score)
        hw_results = single_pic
        return hw_results
    
    def inference(self, data):
        t0 = time.time()
        hw_results = {"detection_classes":[], "detection_boxes":[], "detection_scores":[]}
        img = None
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = np.array(Image.open(file_content).convert("RGB"))
        ori_h, ori_w = img.shape[0:2]
        for image_size in self.img_size:
            prep_out = self._preprocess(img, image_size)
            for model in self.model_list:
                results = self._inference(model, prep_out, image_size)
                hw_results['detection_boxes'] = hw_results['detection_boxes'] + results['detection_boxes']
                hw_results['detection_classes'] = hw_results['detection_classes'] + results['detection_classes']
                hw_results['detection_scores'] = hw_results['detection_scores'] + results['detection_scores']

        boxes_xyxy = np.array(hw_results['detection_boxes'])
        scores = np.array(hw_results['detection_scores'])
        l_all = np.array(hw_results['detection_classes'])
        detections, scores, l_all = weighted_boxes_fusion([boxes_xyxy.tolist()], [scores.tolist()], [l_all.tolist()], weights=None,
                                                          iou_thr=0.55, skip_box_thr=0.0, conf_type='max')

        nms_index = self.normal_nms(detections, scores, thresh=self.nms_thre)
        detections = detections[nms_index]
        scores = scores[nms_index]
        l_all = l_all[nms_index]

        detections = np.hstack((detections,scores.reshape(-1,1)))
        hw_results = {"detection_classes":[], "detection_boxes":[], "detection_scores":[]}
        for di, bbox in enumerate(detections):
            x1, y1, x2, y2, confi = bbox
            hw_box_coord = [float(y1), float(x1), float(y2), float(x2)]
            score = float(confi)
            label = int(l_all[di])
            label_cls = label_mapping[str(label)]
            if label_cls in ['speed_limited', 'speed_unlimited']:
                if (y2-y1)/(x2-x1) > 3 or (y2-y1)/(x2-x1) < 0.5: continue
            if label_cls in ['red_stop', 'green_go', 'yellow_back']:
                if (x2-x1) > (y2-y1): continue
                if (y2-y1)/(x2-x1) > 8.:continue

            hw_results['detection_boxes'].append(hw_box_coord)
            hw_results['detection_classes'].append(label_cls)
            hw_results['detection_scores'].append(score)
        if offline_flag:
            print(f'inferencing time:{time.time()-t0}')
        else:
            logger.info(f'inferencing time:{time.time()-t0}')

        return hw_results



if __name__ == "__main__":
    if offline_flag:
        hwcld = YOLO_service('-1',mn_list=['weights/stage1.ckpt','weights/stage2.ckpt'])
        path = 'samples'
        out_dir = 'outputs'
        import shutil
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        filelist = sorted(os.listdir(path))
        for i, item in enumerate(filelist):
            if item.endswith('.jpg'):
                img_path = os.path.join(path, item)
                input = {"input_img":{'test': img_path}}
                res = hwcld.inference(input)

                det = res['detection_boxes']
                cls = res['detection_classes']
                conf = res['detection_scores']
                img = cv2.imread(img_path)
                thickness = round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
                bbox_color=(255,0,0)
                for di, box in enumerate(det):
                    left_top = (int(box[1]), int(box[0]))
                    right_bottom = (int(box[3]), int(box[2]))
                    confi = conf[di]
                    if confi < 0.1:
                        continue
                    cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=thickness)
                    c1, c2 = left_top, right_bottom
                    tf = max(thickness - 1, 1)
                    label_text = cls[di] + str(round(confi,2))
                    t_size = cv2.getTextSize(label_text, 0, fontScale=thickness / 4, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(img, c1, c2, bbox_color, -1)
                    cv2.putText(img, label_text, (c1[0], c1[1] - 2), 0, thickness / 4, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

                img = cv2.resize(img, (int(img.shape[1]/1), int(img.shape[0]/1)))
                cv2.imwrite(f'{out_dir}/{item}', img)
                print(f'finished {i} image')

    

