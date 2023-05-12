import os
import random

import cv2
from tqdm import tqdm
import json
import shutil
import xml.etree.ElementTree as ET



#  复制文件
def copyfile(old_file_path, new_folder_path):
    shutil.copy(old_file_path, new_folder_path)


def test():
    # 需要修改dir路径，其子文件夹需要有annotations和images
    dir = '/usr/ldw/visdrone2coco/'
    train_dir = os.path.join(dir, "annotations")
    print(train_dir)
    id_num = 0
    categories = [
        {"id": 0, "name": "ignored regions"},
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "bicycle"},
        {"id": 4, "name": "car"},
        {"id": 5, "name": "van"},
        {"id": 6, "name": "truck"},
        {"id": 7, "name": "tricycle"},
        {"id": 8, "name": "awning-tricycle"},
        {"id": 9, "name": "bus"},
        {"id": 10, "name": "motor"},
        {"id": 11, "name": "others"}
    ]
    images = []
    annotations = []
    # 需要修改annotations_path,指向annotations
    # annotations_path = r'J:\Dataset\visdrone\Task 2_ Object Detection in Videos\VisDrone2019-VID-train\annotations'
    annotations_path = '/usr/ldw/visdrone2coco/annotations/'
    set = os.listdir(annotations_path)
    # images_path,指向images
    # images_path = r'J:\Dataset\visdrone\Task 2_ Object Detection in Videos\VisDrone2019-VID-train\images'
    images_path = '/usr/ldw/visdrone2coco/images/'
    print()
    for i in tqdm(set):
        print(annotations_path + "/" + i, "r")
        f = open(annotations_path + "/" + i, "r")
        name = i.replace(".txt", "")
        image = {}
        height, width = cv2.imread(images_path + "/" + name + ".jpg").shape[:2]
        file_name = name + ".jpg"
        image["file_name"] = file_name
        image["height"] = height
        image["width"] = width
        image["id"] = name
        images.append(image)
        for line in f.readlines():
            annotation = {}
            line = line.replace("\n", "")
            if line.endswith(","):  # filter data
                line = line.rstrip(",")
            line_list = [int(i) for i in line.split(",")]
            bbox_xywh = [line_list[0], line_list[1], line_list[2], line_list[3]]
            annotation["image_id"] = name
            annotation["score"] = line_list[4]
            annotation["bbox"] = bbox_xywh
            annotation["category_id"] = int(line_list[5])
            annotation["id"] = id_num
            annotation["iscrowd"] = 0
            annotation["segmentation"] = []
            annotation["area"] = bbox_xywh[2] * bbox_xywh[3]
            id_num += 1
            annotations.append(annotation)
        dataset_dict = {"images": images, "annotations": annotations, "categories": categories}
        json_str = json.dumps(dataset_dict)
        # 修改url，后缀名为json
        url = '/usr/ldw/visdrone2coco/annotations/a1.json'
        with open(url, 'w') as json_file:
            json_file.write(json_str)
    print("json file write done...")


class VisVID2VisDET:
    def __init__(self, src_path, dst_path):
        self.src_path = src_path
        self.dst_path = dst_path
        self.img_id = 0
        pass

    def run(self):
        for root, dirs, files in os.walk(self.src_path, topdown=True):
            for file in files:
                if '.jpg' in file and 'train' in root:
                    self._rename_img(os.path.join(root, file), os.path.join(self.dst_path, 'train'))
                if '.jpg' in file and 'val' in root:
                    self._rename_img(os.path.join(root, file), os.path.join(self.dst_path, 'val'))
                if '.txt' in file and 'train' in root:
                    self._rename_ann(os.path.join(root, file),
                                     os.path.join(self.dst_path, 'annotations/train_txt'))
                if '.txt' in file and 'val' in root:
                    self._rename_ann(os.path.join(root, file),
                                     os.path.join(self.dst_path, 'annotations/val_txt'))

    @staticmethod
    def _rename_img(src_img_path, dst_dir_path):
        # 0000001.jpg -> uav0000079_00480_v_0000001.jpg
        seq_name = src_img_path.split('/')[-2]
        img_name = src_img_path.split('/')[-1]
        dst_img_path = os.path.join(dst_dir_path, seq_name + '_' + img_name)
        shutil.copy(src_img_path, dst_img_path)
        print("{} -> {}".format(src_img_path, dst_img_path))

    def _rename_ann(self, src_ann_path, dst_dir_path):
        # uav0000079_00480_v.txt -> uav0000079_00480_v_0000001.txt
        seq_name = src_ann_path.split('/')[-1]
        w = 1
        frame_id = 1
        while w:
            w = 0
            with open(src_ann_path, 'r') as f:
                for line in f.readlines():
                    # line = line.rstrip('\n')
                    line_list = [int(i) for i in line.split(",")]
                    if line_list[0] == frame_id:
                        single_txt = os.path.join(dst_dir_path, seq_name.split('.')[0] + '_%07d' % frame_id + '.txt')
                        with open(single_txt, 'a') as p:
                            p.write(line)
                            w += 1
                            print("{} is created".format(single_txt))
                frame_id += 1

    def _transformer(self):
        pass


class VisVID2COCO:
    def __init__(self, save_path, train_ratio, is_mode="train"):
        # 12 classes
        # self.category_list = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van',
        #                       'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
        # actually 10 classes
        # self.category_list = ['pedestrian', 'people', 'bicycle', 'car', 'van',
        #                       'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
        # seq1 4 classes
        # self.category_list = ['pedestrian', 'people', 'bicycle', 'motor']
        # seq2 9 classes
        # self.category_list = ['pedestrian', 'people', 'bicycle', 'car', 'van',
        #                       'truck', 'tricycle', 'awning-tricycle', 'motor']
        # seq3 6 classes
        # self.category_list = ['pedestrian', 'car', 'van', 'truck', 'bus', 'motor']
        # seq4 4 classes
        self.category_list = ['pedestrian', 'car', 'van', 'truck']
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.save_path = save_path
        self.train_ratio = train_ratio
        self.is_mode = is_mode
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def to_coco(self, ann_dir, img_dir, restriction=None):
        self._init_categories()
        img_list_org = os.listdir(img_dir)
        img_list = []
        if restriction:
            for res_str in restriction:
                img_list += list(filter(lambda x: res_str in x, img_list_org))
        else:
            img_list = img_list_org

        for img_name in img_list:
            ann_path = os.path.join(ann_dir, img_name.replace(os.path.splitext(img_name)[-1], '.txt'))
            if not os.path.isfile(ann_path):
                print('this ann file ({}) is not exist !'.format(ann_path))
                continue
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            self.images.append(self._image(img_path, h, w))
            if self.img_id % 10 == 0:
                print("processing No.{} image".format(self.img_id))

            with open(ann_path, 'r') as f:
                for line in f.readlines():
                    try:
                        if ',' in line:
                            _, _, xmin, ymin, w, h, score, category, trunc, occlusion = line.split(',')
                        else:
                            _, _, xmin, ymin, w, h, score, category, trunc, occlusion = line.split()
                    except:
                        print('error: ', ann_path, 'line: ', line)
                        continue
                    # all
                    # if int(category) in [0, 11] or int(w) < 4 or int(h) < 4:
                    #     continue
                    # seq1
                    # if int(category) in [0, 4, 5, 6, 7, 8, 9, 11] or int(w) < 4 or int(h) < 4:
                    #     continue
                    # if int(category) == 1:
                    #     category = '0'
                    # if int(category) == 2:
                    #     category = '1'
                    # if int(category) == 3:
                    #     category = '2'
                    # if int(category) == 10:
                    #     category = '3'
                    # seq2
                    # if int(category) in [0, 9, 11] or int(w) < 4 or int(h) < 4:
                    #     continue
                    # if int(category) in [1, 2, 3, 4, 5, 6, 7, 8]:
                    #     category = str(int(category) - 1)
                    # if int(category) == 10:
                    #     category = '8'
                    # seq3
                    # if int(category) in [0, 2, 3, 7, 8, 11] or int(w) < 4 or int(h) < 4:
                    #     continue
                    # if int(category) == 1:
                    #     category = '0'
                    # if int(category) in [4, 5, 6]:
                    #     category = str(int(category) - 3)
                    # if int(category) in [9, 10]:
                    #     category = str(int(category) - 5)
                    # seq4
                    if int(category) in [0, 2, 3, 7, 8, 9, 10, 11] or int(w) < 4 or int(h) < 4:
                        continue
                    if int(category) == 1:
                        category = '0'
                    if int(category) in [4, 5, 6]:
                        category = str(int(category) - 3)
                    label, bbox = int(category), [int(xmin), int(ymin), int(w), int(h)]
                    annotation = self._annotation(label, bbox)
                    self.annotations.append(annotation)
                    self.ann_id += 1
            self.img_id += 1
        instance = {
            "info": 'VisDrone2019-VID',
            "license": ['none'],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        return instance

    def _init_categories(self):
        cls_num = len(self.category_list)
        for v in range(0, cls_num):
            category = {
                "id": v,
                "name": self.category_list[v],
                "supercategory": self.category_list[v]
            }
            self.categories.append(category)

    def _image(self, path, h, w):
        image = {
            "height": h,
            "width": w,
            "id": self.img_id,
            "file_name": os.path.basename(path)
        }
        return image

    def _annotation(self, label, bbox):
        area = bbox[2] * bbox[3]
        annotation = {
            "id": self.ann_id,
            "image_id": self.img_id,
            "category_id": label,
            "segmentation": [],
            "bbox": bbox,
            "iscrowd": 0,
            "ignore": 0,
            "area": area
        }
        return annotation

    @staticmethod
    def save_coco_json(instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=4, separators=(',', ': '))


class RSOD2YOLO:

    def __init__(self, src_path, dst_path, train_ratio=0.6):
        self.category_list = ['aircraft', 'oiltank', 'overpass', 'playground']
        self.src_path = src_path
        self.dst_path = dst_path
        self.train_ratio = train_ratio

    @staticmethod
    def convert(bbox, img_size):
        x_c = (bbox[0] + bbox[2]) / 2.0
        y_c = (bbox[1] + bbox[3]) / 2.0
        x = x_c / img_size[0]  # img_size[0] width
        y = y_c / img_size[1]  # img_size[1] height
        w = (bbox[2] - bbox[0]) / img_size[0]
        h = (bbox[3] - bbox[1]) / img_size[1]

        return x, y, w, h

    def create_yolo_annotation(self):
        for cls in self.category_list:
            out_labels_path = os.path.join(self.src_path, cls, 'Annotation/yolo_labels')
            if not os.path.exists(out_labels_path):
                os.makedirs(out_labels_path)
            xml_files_path = os.path.join(self.src_path, cls, 'Annotation/xml')
            xml_files = os.listdir(xml_files_path)
            for xml_name in xml_files:
                xml_file = os.path.join(xml_files_path, xml_name)
                out_txt_path = os.path.join(out_labels_path, xml_name.split('.')[0] + '.txt')
                out_txt = open(out_txt_path, 'w')
                tree = ET.parse(xml_file)
                root = tree.getroot()
                img_size = (int(root.find('size').find('width').text), int(root.find('size').find('height').text))

                for obj in root.iter('object'):
                    # difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    cls_id = self.category_list.index(cls)
                    bndbox = obj.find('bndbox')
                    bbox = (
                        float(bndbox.find('xmin').text),
                        float(bndbox.find('ymin').text),
                        float(bndbox.find('xmax').text),
                        float(bndbox.find('ymax').text)
                    )
                    label = tuple(self.convert(bbox, img_size))
                    out_txt.write(str(cls_id) + ' ' + ' '.join([str(l) for l in label]) + '\n')
                    out_txt.close()

    def create_sets(self):
        for cls in self.category_list:
            out_labels_path = os.path.join(self.src_path, cls, 'Annotation/yolo_labels')
            out_labels_train_path = os.path.join(self.src_path, cls, 'Annotation/yolo_labels_train')
            out_labels_val_path = os.path.join(self.src_path, cls, 'Annotation/yolo_labels_val')
            out_labels_test_path = os.path.join(self.src_path, cls, 'Annotation/yolo_labels_test')
            if not os.path.exists(out_labels_path):
                os.makedirs(out_labels_path)
            if not os.path.exists(out_labels_train_path):
                os.makedirs(out_labels_train_path)
            if not os.path.exists(out_labels_val_path):
                os.makedirs(out_labels_val_path)
            if not os.path.exists(out_labels_test_path):
                os.makedirs(out_labels_test_path)

            images_path = os.path.join(self.src_path, cls, 'JPEGImages')
            images_train_path = os.path.join(self.src_path, cls, 'images_train')
            images_val_path = os.path.join(self.src_path, cls, 'images_val')
            images_test_path = os.path.join(self.src_path, cls, 'images_test')
            if not os.path.exists(images_train_path):
                os.makedirs(images_train_path)
            if not os.path.exists(images_val_path):
                os.makedirs(images_val_path)
            if not os.path.exists(images_test_path):
                os.makedirs(images_test_path)

            labels_list = os.listdir(out_labels_path)
            num = len(labels_list)
            num_list = range(num)
            train_num = int(num * self.train_ratio)
            train_samples = random.sample(num_list, train_num)

            for i in num_list:
                if i in train_samples:
                    shutil.copyfile(os.path.join(out_labels_path, labels_list[i]),
                                    os.path.join(out_labels_train_path, labels_list[i]))
                    shutil.copyfile(os.path.join(images_path, labels_list[i].split('.')[0] + '.jpg'),
                                    os.path.join(images_train_path, labels_list[i].split('.')[0] + '.jpg'))
                else:
                    shutil.copyfile(os.path.join(out_labels_path, labels_list[i]),
                                    os.path.join(out_labels_val_path, labels_list[i]))
                    shutil.copyfile(os.path.join(out_labels_path, labels_list[i]),
                                    os.path.join(out_labels_test_path, labels_list[i]))
                    shutil.copyfile(os.path.join(images_path, labels_list[i].split('.')[0] + '.jpg'),
                                    os.path.join(images_val_path, labels_list[i].split('.')[0] + '.jpg'))
                    shutil.copyfile(os.path.join(images_path, labels_list[i].split('.')[0] + '.jpg'),
                                    os.path.join(images_test_path, labels_list[i].split('.')[0] + '.jpg'))


class HRRSD2YOLO:

    def __init__(self, src_path, dst_path, train_ratio=None):
        self.category_list = ['bridge',
                              'airplane',
                              'ground track field',
                              'vehicle',
                              'parking lot',
                              'T junction',
                              'baseball diamond',
                              'tennis court',
                              'basketball court',
                              'ship',
                              'crossroad',
                              'harbor',
                              'storage tank']
        self.src_path = src_path
        self.dst_path = dst_path
        self.train_ratio = train_ratio

    @staticmethod
    def convert(bbox, img_size):
        x_c = (bbox[0] + bbox[2]) / 2.0
        y_c = (bbox[1] + bbox[3]) / 2.0
        x = x_c / img_size[0]  # img_size[0] width
        y = y_c / img_size[1]  # img_size[1] height
        w = (bbox[2] - bbox[0]) / img_size[0]
        h = (bbox[3] - bbox[1]) / img_size[1]

        return x, y, w, h

    @staticmethod
    def txt2list(txt_file):
        txt_list = []
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                txt_list.append(line.strip('\n'))
        return txt_list

    def create_yolo_annotation(self):
        xml_files_path = os.path.join(self.src_path, 'xmls')
        out_labels_path = os.path.join(self.src_path, 'yolo_labels')
        if not os.path.exists(out_labels_path):
            os.makedirs(out_labels_path)
        xml_files = os.listdir(xml_files_path)
        for xml_name in xml_files:
            xml_file = os.path.join(xml_files_path, xml_name)
            out_txt_path = os.path.join(out_labels_path, xml_name.split('.')[0] + '.txt')
            out_txt = open(out_txt_path, 'w')
            tree = ET.parse(xml_file)
            root = tree.getroot()
            img_size = (int(root.find('size').find('width').text), int(root.find('size').find('height').text))

            for obj in root.iter('object'):
                # difficult = obj.find('difficult').text
                cls = obj.find('name').text
                cls_id = self.category_list.index(cls)
                bndbox = obj.find('bndbox')
                bbox = (
                    float(bndbox.find('xmin').text),
                    float(bndbox.find('ymin').text),
                    float(bndbox.find('xmax').text),
                    float(bndbox.find('ymax').text)
                )
                label = tuple(self.convert(bbox, img_size))
                out_txt.write(str(cls_id) + ' ' + ' '.join([str(l) for l in label]) + '\n')
            out_txt.close()

    def create_sets(self):
        image_sets_main = os.path.join(self.src_path, 'ImageSets/Main')
        train_list = self.txt2list(os.path.join(image_sets_main, 'train.txt'))
        trainval_list = self.txt2list(os.path.join(image_sets_main, 'trainval.txt'))
        val_list = self.txt2list(os.path.join(image_sets_main, 'val.txt'))
        test_list = self.txt2list(os.path.join(image_sets_main, 'test.txt'))
        images_train_path = os.path.join(self.src_path, 'images_train')
        images_trainval_path = os.path.join(self.src_path, 'images_trainval')
        images_val_path = os.path.join(self.src_path, 'images_val')
        images_test_path = os.path.join(self.src_path, 'images_test')
        if not os.path.exists(images_train_path):
            os.makedirs(images_train_path)
        if not os.path.exists(images_trainval_path):
            os.makedirs(images_trainval_path)
        if not os.path.exists(images_val_path):
            os.makedirs(images_val_path)
        if not os.path.exists(images_test_path):
            os.makedirs(images_test_path)
        labels_train_path = os.path.join(self.src_path, 'labels_train')
        labels_trainval_path = os.path.join(self.src_path, 'labels_trainval')
        labels_val_path = os.path.join(self.src_path, 'labels_val')
        labels_test_path = os.path.join(self.src_path, 'labels_test')
        if not os.path.exists(labels_train_path):
            os.makedirs(labels_train_path)
        if not os.path.exists(labels_trainval_path):
            os.makedirs(labels_trainval_path)
        if not os.path.exists(labels_val_path):
            os.makedirs(labels_val_path)
        if not os.path.exists(labels_test_path):
            os.makedirs(labels_test_path)
        images_path = os.path.join(self.src_path, 'images')
        out_labels_path = os.path.join(self.src_path, 'yolo_labels')
        for i in train_list:
            shutil.copyfile(os.path.join(out_labels_path, i + '.txt'),
                            os.path.join(labels_train_path, i + '.txt'))
            shutil.copyfile(os.path.join(images_path, i + '.jpg'),
                            os.path.join(images_train_path, i + '.jpg'))
        for i in trainval_list:
            shutil.copyfile(os.path.join(out_labels_path, i + '.txt'),
                            os.path.join(labels_trainval_path, i + '.txt'))
            shutil.copyfile(os.path.join(images_path, i + '.jpg'),
                            os.path.join(images_trainval_path, i + '.jpg'))
        for i in val_list:
            shutil.copyfile(os.path.join(out_labels_path, i + '.txt'),
                            os.path.join(labels_val_path, i + '.txt'))
            shutil.copyfile(os.path.join(images_path, i + '.jpg'),
                            os.path.join(images_val_path, i + '.jpg'))
        for i in test_list:
            shutil.copyfile(os.path.join(out_labels_path, i + '.txt'),
                            os.path.join(labels_test_path, i + '.txt'))
            shutil.copyfile(os.path.join(images_path, i + '.jpg'),
                            os.path.join(images_test_path, i + '.jpg'))


class YOLO2COCO:
    def __init__(self, src_img_path, src_ann_path, dst_path, category_list):
        self.src_img_path = src_img_path
        self.src_ann_path = src_ann_path
        self.src_img = sorted(os.listdir(src_img_path))
        self.src_ann = sorted(os.listdir(src_ann_path))
        self.dst_ann_path = dst_path
        self.category_list = category_list

    @staticmethod
    def convert(label, img_size, ann_id_cnt, k):
        cls_id = int(label[0])
        x = float(label[1])
        y = float(label[2])
        w = float(label[3])
        h = float(label[4])
        H, W, _ = img_size
        x1 = (x - w / 2) * W
        y1 = (y - h / 2) * H
        x2 = (x + w / 2) * W
        y2 = (y + h / 2) * H
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        ann = {
            'area': width * height,
            'bbox': [x1, y1, width, height],
            'category_id': cls_id,
            'id': ann_id_cnt,
            'image_id': k,
            'iscrowd': 0,
            # 'score': float(label[5]),
            # mask, 矩形是从左上角点按顺时针的四个顶点
            'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
        }
        return ann

    def create_coco_annotation(self, reset=False):
        if not reset:
            anns = {
                'categories': [],
                'annotations': [],
                'images': []
            }
            for i, cls in enumerate(self.category_list, 0):
                anns['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

        ann_id_cnt = 0
        for k, txt in enumerate(tqdm(self.src_ann)):
            if reset:
                ann_id_cnt = 0
                anns = {
                    'categories': [],
                    'annotations': [],
                    'images': []
                }
                for i, cls in enumerate(self.category_list, 0):
                    anns['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            img = txt.replace('.txt', '.jpg')
            image = cv2.imread(os.path.join(self.src_img_path, img))
            image_size = image.shape
            anns['images'].append({
                'file_name': img,
                'id': k,
                'width': image_size[1],
                'height': image_size[0]
            })

            with open(os.path.join(self.src_ann_path, txt), 'r') as f:
                labels = f.readlines()
                for label in labels:
                    label = label.strip().split()
                    ann = self.convert(label, image_size, ann_id_cnt, k)
                    anns['annotations'].append(ann)
                    ann_id_cnt += 1
            if reset:
                json_name = os.path.join(self.dst_ann_path, '{}.json'.format(txt.strip('.txt')))
                with open(json_name, 'w') as f:
                    json.dump(anns, f)
        if not reset:
            name = 'train'
            json_name = os.path.join(self.dst_ann_path, '{}.json'.format(name))
            with open(json_name, 'w') as f:
                json.dump(anns, f)
                print('{} saved !'.format(json_name))

    def create_coco_results(self):
        # results = []
        ann_id_cnt = 0
        for k, txt in enumerate(tqdm(self.src_ann)):
            results = []
            img = txt.replace('.txt', '.jpg')
            image = cv2.imread(os.path.join(self.src_img_path, img))
            image_size = image.shape
            with open(os.path.join(self.src_ann_path, txt), 'r') as f:
                labels = f.readlines()
                for label in labels:
                    label = label.strip().split()
                    ann = self.convert(label, image_size, ann_id_cnt, k)
                    results.append({
                        'image_id': k,
                        'category_id': ann['category_id'],
                        'bbox': ann['bbox'],
                        'score': float(label[5])
                    })
            json_name = os.path.join(self.dst_ann_path, '{}.json'.format(txt.strip('.txt')))
            with open(json_name, 'w') as f:
                json.dump(results, f)


if __name__ == '__main__':
    SRC_IMG_PATH = '/home/joshuawen/WorkSpace/datasets/RSOD/train/images'
    SRC_ANN_PATH = '/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/all-train/labels'
    DST_PATH = '/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/all-train/json'
    category_list = ['aircraft', 'oiltank', 'overpass', 'playground']
    trans = YOLO2COCO(src_img_path=SRC_IMG_PATH,
                      src_ann_path=SRC_ANN_PATH,
                      dst_path=DST_PATH,
                      category_list=category_list)
    # trans.create_coco_annotation(reset=True)
    trans.create_coco_results()
    # VID -> DET
    # SRC_PATH = '/media/joshuawen/JoshuaWS3/Datasets/RL/VisDrone2019_VID'
    # DST_PATH = '/media/joshuawen/JoshuaWS3/Datasets/RL/VisDrone2019_VID_COCO'
    # trans = VisVID2VisDET(src_path=SRC_PATH, dst_path=DST_PATH)
    # trans.run()
    # DET -> COCO
    # mode = 'train'
    # SEQ = {"seq1_train_4c": ["uav0000079_00480_v", "uav0000084_00000_v"],
    #        "seq1_val_4c": ["uav0000086_00000_v"],
    #        "seq2_train_9c": ["uav0000150_02310_v", "uav0000222_03150_v", "uav0000316_01288_v", "uav0000357_00920_v",
    #                          "uav0000361_02323_v"],
    #        "seq2_val_9c": ["uav0000137_00458_v", "uav0000182_00000_v"],
    #        "seq3_train_6c": ["uav0000326_01035_v"],
    #        "seq3_val_6c": ["uav0000305_00000_v"],
    #        "seq4_train_4c": ["uav0000145_00000_v", "uav0000218_00001_v", "uav0000263_03289_v", "uav0000266_03598_v"],
    #        "seq4_val_4c": ["uav0000268_05773_v"],
    #        }
    # SAVE_PATH = '/media/joshuawen/JoshuaWS3/Datasets/RL/VisDrone2019_VID_COCO/annotations'
    # ANN_PATH = '/media/joshuawen/JoshuaWS3/Datasets/RL/VisDrone2019_VID_COCO/annotations/{}_txt'.format(mode)
    # IMG_PATH = '/media/joshuawen/JoshuaWS3/Datasets/RL/VisDrone2019_VID_COCO/{}'.format(mode)
    # creator = VisVID2COCO(save_path=SAVE_PATH, train_ratio=1.0, is_mode=mode)
    # # ins = creator.to_coco(ann_dir=ANN_PATH, img_dir=IMG_PATH)
    # ins = creator.to_coco(ann_dir=ANN_PATH, img_dir=IMG_PATH, restriction=SEQ["seq4_train_4c"])
    # # creator.save_coco_json(instance=ins,
    # #                        save_path=os.path.join(SAVE_PATH, 'VisDrone2019_VID_{}_coco.json'.format(mode)))
    # creator.save_coco_json(instance=ins,
    #                        save_path=os.path.join(SAVE_PATH, 'VisDrone2019_VID_{}_coco.json'.format("seq4_train_4c")))
    # convert YOLO format
    # SRC_PATH = '/media/joshuawen/JoshuaWS3/Datasets/CV/RGB/RS/HRRSD'
    # DST_PATH = ''
    # creator = RSOD2YOLO(src_path=SRC_PATH, dst_path=DST_PATH)
    # creator = HRRSD2YOLO(src_path=SRC_PATH, dst_path=DST_PATH)
    # creator.create_yolo_annotation()
    # creator.create_sets()
