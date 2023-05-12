import os

from pycocotools.coco import COCO
import xml.etree.ElementTree as ET

# # COCO format
# annFile = "/media/joshuawen/JoshuaWS3/Datasets/RL/VisDrone2019_VID_COCO/annotations/VisDrone2019_VID_seq4_val_4c_coco.json"
#
# # initialize COCO api for instance annotations
# coco = COCO(annFile)
#
# # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# cat_nms = [cat['name'] for cat in cats]
# print('number of categories: ', len(cat_nms))
# print('COCO categories: \n', cat_nms)
#
# # 统计各类的图片数量和标注框数量
# for cat_name in cat_nms:
#     catId = coco.getCatIds(catNms=cat_name)  # 1~90
#     imgId = coco.getImgIds(catIds=catId)  # 图片的id
#     annId = coco.getAnnIds(catIds=catId)  # 标注框的id
#     print("class name: {}, {}".format(catId[0], cat_name))
#     print("image number: {}".format(len(imgId)))
#     print("annotation number: {}".format(len(annId)))


# XML format
# xml_dir = "/media/joshuawen/JoshuaWS3/Datasets/CV/UA-DETRAC/trainlabel/DETRAC-Train-Annotations-XML-v3"
# xml_dir = "/media/joshuawen/JoshuaWS3/Datasets/CV/UA-DETRAC/testlabel/DETRAC-Test-Annotations-XML"

# i = 0
# res = {
#     'sunny': 0,
#     'sunny_un': 0,
#     'night': 0,
#     'night_un': 0,
#     'cloudy': 0,
#     'cloudy_un': 0,
#     'rainy': 0,
#     'rainy_un': 0,
# }
# for file in sorted(os.listdir(xml_dir)):
#     ann = ET.parse(os.path.join(xml_dir, file)).getroot()
#     name = ann.attrib['name']
#     sa = ann.find('sequence_attribute').attrib
#     if sa['sence_weather'] == 'sunny' and sa['camera_state'] == 'stable':
#         res['sunny'] += 1
#         # print(sa, name)
#     if sa['sence_weather'] == 'night' and sa['camera_state'] == 'stable':
#         res['night'] += 1
#         # print(sa, name)
#     if sa['sence_weather'] == 'cloudy' and sa['camera_state'] == 'stable':
#         res['cloudy'] += 1
#         # print(sa, name)
#     if sa['sence_weather'] == 'rainy' and sa['camera_state'] == 'stable':
#         res['rainy'] += 1
#         # print(sa, name)
#     if sa['sence_weather'] == 'sunny' and sa['camera_state'] == 'unstable':
#         res['sunny_un'] += 1
#         # print(sa, name)
#     if sa['sence_weather'] == 'night' and sa['camera_state'] == 'unstable':
#         res['night_un'] += 1
#         # print(sa, name)
#     if sa['sence_weather'] == 'cloudy' and sa['camera_state'] == 'unstable':
#         res['cloudy_un'] += 1
#         # print(sa, name)
#     if sa['sence_weather'] == 'rainy' and sa['camera_state'] == 'unstable':
#         res['rainy_un'] += 1
#         print(sa, name)
#     i += 1
#
# print(res)

# txt_dir = "/media/joshuawen/JoshuaWS3/Datasets/CV/UAVDT_M/M_attr/train"
txt_dir = "/media/joshuawen/JoshuaWS3/Datasets/CV/UAVDT_M/M_attr/test"

i = 0
res = {
    'daylight': {'num': 0, 'seq': []},
    'night': {'num': 0, 'seq': []},
    'fog': {'num': 0, 'seq': []},
    'low-alt': {'num': 0, 'seq': []},
    'medium-alt': {'num': 0, 'seq': []},
    'high-alt': {'num': 0, 'seq': []},
    'front-view': {'num': 0, 'seq': []},
    'side-view': {'num': 0, 'seq': []},
    'bird-view': {'num': 0, 'seq': []},
    'long-term': {'num': 0, 'seq': []},
}
for txt in sorted(os.listdir(txt_dir)):
    with open(os.path.join(txt_dir, txt), 'r') as f:
        for line in f.readlines():
            try:
                if ',' in line:
                    w1, w2, w3, a1, a2, a3, v1, v2, v3, t = line.split(',')
            except:
                print('ERROR')
                continue
        if int(w1) == 1:
            res['daylight']['num'] += 1
            res['daylight']['seq'].append(txt)
        if int(w2) == 1:
            res['night']['num'] += 1
            res['night']['seq'].append(txt)
        if int(w3) == 1:
            res['fog']['num'] += 1
            res['fog']['seq'].append(txt)
        if int(a1) == 1:
            res['low-alt']['num'] += 1
            res['low-alt']['seq'].append(txt)
        if int(a2) == 1:
            res['medium-alt']['num'] += 1
            res['medium-alt']['seq'].append(txt)
        if int(a3) == 1:
            res['high-alt']['num'] += 1
            res['high-alt']['seq'].append(txt)
        if int(v1) == 1:
            res['front-view']['num'] += 1
            res['front-view']['seq'].append(txt)
        if int(v2) == 1:
            res['side-view']['num'] += 1
            res['side-view']['seq'].append(txt)
        if int(v3) == 1:
            res['bird-view']['num'] += 1
            res['bird-view']['seq'].append(txt)
        if int(t) == 1:
            res['long-term']['num'] += 1
            res['long-term']['seq'].append(txt)

print("w1")
print(res['daylight'])
print("w2")
print(res['night'])
print("w3")
print(res['fog'])
print("a1")
print(res['low-alt'])
print("a2")
print(res['medium-alt'])
print("a3")
print(res['high-alt'])
print("v1")
print(res['front-view'])
print("v2")
print(res['side-view'])
print("v3")
print(res['bird-view'])
print("t")
print(res['long-term'])
