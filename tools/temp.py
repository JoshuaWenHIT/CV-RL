# import os
# IMG_PATH = "/home/joshuawen/WorkSpace/datasets/RSOD/train/images"
# RESULTS_PATH = {
#     "all": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/all-train/labels",
#     "aircraft": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/aircraft-train/labels",
#     "oiltank": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/oiltank-train/labels",
#     "overpass": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/overpass-train/labels",
#     "playground": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/playground-train/labels"
# }
#
#
# def get_detection(model, sequence, index):
#     dets = []
#     i = 0
#     det_txt = os.path.join(RESULTS_PATH[model], sequence[index].replace('.jpg', '.txt'))
#     print(det_txt)
#     if os.path.exists(det_txt):
#         with open(det_txt, 'r') as file:
#             for line in file:
#                 det_line = line.strip('\n').split()
#                 dets.append(list(det_line))
#     else:
#         return dets
#     return dets
#
#
# dets = get_detection(model="aircraft", sequence=os.listdir(IMG_PATH), index=0)
# print(dets)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from tempfile import NamedTemporaryFile
import sys
import os

# # coco格式的json文件，原始标注数据
# anno_file = '/media/joshuawen/JoshuaWorkSpace2/Datasets/Det/coco128/annotations/instances_val2017.json'
# coco_gt = COCO(anno_file)
#
# # 用GT框作为预测框进行计算，目的是得到detection_res
# with open(anno_file, 'r') as f:
#     json_file = json.load(f)
# annotations = json_file['annotations']
# detection_res = []
# for anno in annotations:
#     detection_res.append({
#         'score': 1.,
#         'category_id': anno['category_id'],
#         'bbox': anno['bbox'],
#         'image_id': anno['image_id']
#     })
#
# with NamedTemporaryFile(suffix='.json') as tf:
#     # 由于后续需要，先将detection_res转换成二进制后写入json文件
#     content = json.dumps(detection_res).encode(encoding='utf-8')
#     tf.write(content)
#     res_path = tf.name
#     print(res_path)
#
#     # loadRes会在coco_gt的基础上生成一个新的COCO类型的instance并返回
#     coco_dt = coco_gt.loadRes(res_path)
#
#     cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()
#
# print(cocoEval.stats)


class HiddenPrints:
    def __init__(self, activated=True):
        # activated参数表示当前修饰类是否被激活
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # 这里的os.devnull实际上就是Linux系统中的“/dev/null”
        # /dev/null会使得发送到此目标的所有数据无效化，就像“被删除”一样
        # 这里使用/dev/null对sys.stdout输出流进行重定向

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()


if __name__ == '__main__':

    coco_gt = COCO(annotation_file='/home/joshuawen/WorkSpace/datasets/RSOD/train/json/aircraft_4.json')
    coco_pre = coco_gt.loadRes('/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/all-train/json/aircraft_4.json')

    HiddenPrints().close()
    coco_evaluator = COCOeval(cocoGt=coco_gt, cocoDt=coco_pre, iouType="bbox")
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    print(coco_evaluator.stats[2])



