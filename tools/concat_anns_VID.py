import os
import json
import cv2
import numpy as np

# path init
SET_LIST = ['train', 'val', 'test-dev', 'test-challenge']
SET = SET_LIST[1]
ROOT_PATH = "/media/ljb/Joshua_SSD3/Datasets/Det/VisDrone2019-VID-{}".format(SET)
ANN_PATH = os.path.join(ROOT_PATH, "annotations")
FRAME_PATH = os.path.join(ROOT_PATH, "sequences")
# SEQ_LIST = ['uav0000071_03240_v', 'uav0000084_00000_v', 'uav0000150_02310_v', 'uav0000326_01035_v']
SEQ_LIST = ['uav0000326_01035_v', 'uav0000150_02310_v', 'uav0000071_03240_v', 'uav0000084_00000_v']
# SEQ_LIST = ['uav0000086_00000_v', 'uav0000117_02622_v', 'uav0000137_00458_v', 'uav0000305_00000_v']
# SEQ_LIST = ['uav0000305_00000_v', 'uav0000137_00458_v', 'uav0000117_02622_v', 'uav0000086_00000_v']
OUT_PATH = "/media/ljb/Joshua_SSD3/Datasets/Det/{}_concat.json".format(SET)

# CLASS_NAME = {'ignored regions': 0, 'pedestrian': 1, 'people': 2, 'bicycle': 3, 'car': 4, 'van': 5,
#               'truck': 6, 'tricycle': 7, 'awning-tricycle': 8, 'bus': 9, 'motor': 10, 'others': 11}

if __name__ == '__main__':
    out = {'images': [], 'annotations': [],
           'categories': [{'id': 1, 'name': 'pedestrian'}, {'id': 2, 'name': 'car'}],
           'videos': []}
    # out = {'images': [], 'annotations': [],
    #        'categories': [{'id': 1, 'name': 'pedestrian'}, {'id': 2, 'name': 'people'}, {'id': 3, 'name': 'bicycle'},
    #                       {'id': 4, 'name': 'car'}, {'id': 5, 'name': 'van'}, {'id': 6, 'name': 'truck'},
    #                       {'id': 7, 'name': 'tricycle'}, {'id': 8, 'name': 'awning-tricycle'},
    #                       {'id': 9, 'name': 'bus'}, {'id': 10, 'name': 'motor'}, {'id': 11, 'name': 'others'},
    #                       {'id': 0, 'name': 'ignored regions'}],
    #        'videos': []}
    frame_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    h = 0
    w = 0
    # seq = SEQ
    for seq in SEQ_LIST:
        video_cnt += 1
        out['videos'].append({'id': video_cnt, 'seq_name': seq})
        frame_list = os.listdir(os.path.join(FRAME_PATH, seq))
        # get frame information
        num_frames = len(frame_list)
        for i in range(num_frames):
            if i == 0:
                img = cv2.imread('{}/{}/{:07d}.jpg'.format(FRAME_PATH, seq, i+1))
                h, w = img.shape[0:2]
            frame_info = {
                'file_name': '{}/{}/{:07d}.jpg'.format(FRAME_PATH, seq, i+1),
                'id': frame_cnt + i + 1,
                'height': int(h),
                'width': int(w),
            }
            out['images'].append(frame_info)
        print('{}: {} images'.format(seq, num_frames))
        # get annotation information
        ann_path = '{}/{}.txt'.format(ANN_PATH, seq)
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
        category_id = None
        for i in range(anns.shape[0]):
            frame_id = int(anns[i][0])
            track_id = int(anns[i][1])
            cat_id = int(anns[i][7])
            ann_cnt += 1
            if not (int(anns[i][6] == 1)):
                continue
            if int(anns[i][9]) == 2:
                continue
            if int(anns[i][7]) in [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]:
                continue
            if int(anns[i][7]) == 1:
                category_id = 1
            if int(anns[i][7]) == 4:
                category_id = 2
            seg = []
            seg.append(float(anns[i][2]))
            seg.append(float(anns[i][3]))
            seg.append(float(anns[i][2]))
            seg.append(float(anns[i][3] + anns[i][5]))
            seg.append(float(anns[i][2] + anns[i][4]))
            seg.append(float(anns[i][3] + anns[i][5]))
            seg.append(float(anns[i][2] + anns[i][4]))
            seg.append(float(anns[i][3]))
            ann = {
                'area': float(anns[i][4]) * float(anns[i][5]),
                'iscrowd': 0,
                'image_id': frame_cnt + int(anns[i][0]),
                'bbox': anns[i][2:6].tolist(),
                'category_id': category_id,
                'id': i + 1,
                'ignore': 0,
                'segmentation': []
            }
            # ann['segmentation'].append(seg)
            out['annotations'].append(ann)
        frame_cnt += num_frames
    print('loaded {} for {} images and {} samples'.format(
        SET, len(out['images']), len(out['annotations'])))
    json.dump(out, open(OUT_PATH, 'w'))
