import json
import os
import cv2
from loguru import logger
import numpy as np
import gym
from gym import spaces
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tools.utils.misc import HiddenPrints

N_DISCRETE_ACTIONS = 5  # model0: all, model1: aircraft, model2: oiltank, model3: overpass, model4: playground
FPS = 30
FRAME_WIDTH = 128
FRANE_HEIGHT = 128
N_CHANNELS = 3

# RSOD
IMG_PATH = "/home/joshuawen/WorkSpace/datasets/RSOD/train/images"
# GT_PATH = "/home/joshuawen/WorkSpace/datasets/RSOD/train/labels"
GT_PATH = "/home/joshuawen/WorkSpace/datasets/RSOD/train/json"
MODEL_PATH = {
    "all": "/home/joshuawen/WorkSpace/yolov8_rl/runs/detect/RSOD/all/train6-all/weights/last.pt",
    "aircraft": "/home/joshuawen/WorkSpace/yolov8_rl/runs/detect/RSOD/aircraft/train6-0-2/weights/last.pt",
    "oiltank": "/home/joshuawen/WorkSpace/yolov8_rl/runs/detect/RSOD/oiltank/train6-1-2/weights/last.pt",
    "overpass": "/home/joshuawen/WorkSpace/yolov8_rl/runs/detect/RSOD/overpass/train6-2-1/weights/last.pt",
    "playground": "/home/joshuawen/WorkSpace/yolov8_rl/runs/detect/RSOD/playground/train6-3-1/weights/last.pt"
}
# RESULTS_PATH = {
#     "all": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/all-train/labels",
#     "aircraft": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/aircraft-train/labels",
#     "oiltank": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/oiltank-train/labels",
#     "overpass": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/overpass-train/labels",
#     "playground": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/playground-train/labels"
# }
RESULTS_PATH = {
    "all": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/all-train/json",
    "aircraft": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/aircraft-train/json",
    "oiltank": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/oiltank-train/json",
    "overpass": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/overpass-train/json",
    "playground": "/media/joshuawen/JoshuaWS3/Exp/yolov8_rl/RSOD/res/playground-train/json"
}


class DetQuest(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DetQuest, self).__init__()
        self.sequence = sorted(os.listdir(IMG_PATH))
        self.gt = sorted(os.listdir(GT_PATH))
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation = np.zeros((FRANE_HEIGHT, FRAME_WIDTH, N_CHANNELS))
        self.observation_space = spaces.Box(low=0, high=255, shape=(N_CHANNELS, FRANE_HEIGHT, FRAME_WIDTH,))
        self.state_init = self.sequence[0]

        self.current_frame_index = 0
        self.step_count = 0

        # RSOD
        self.mean_vector = np.asarray([0.451, 0.449, 0.433] * 255, dtype=np.float32).reshape(1, 1, 3)
        self.std_vector = np.asarray([0.197, 0.192, 0.189] * 255, dtype=np.float32).reshape(1, 1, 3)

        self.cv_model = None

    def step(self, action):
        if action == 0:
            self.cv_model = "all"
        elif action == 1:
            self.cv_model = "aircraft"
        elif action == 2:
            self.cv_model = "oiltank"
        elif action == 3:
            self.cv_model = "overpass"
        elif action == 4:
            self.cv_model = "playground"
        else:
            assert False

        if self.current_frame_index == (len(self.sequence) - 1):
            self.current_frame_index = 0
        else:
            self.current_frame_index += 1

        detections = self.get_detection(self.cv_model, self.current_frame_index)
        reward = self.get_reward(detections)
        self.observation = self.state_preprocess(self.current_frame_index)

        self.step_count += 1
        done = False
        info = {
            "episode": {
                "a": action,
                "r": reward,
                "l": self.current_frame_index,
                "n": self.sequence[self.current_frame_index]
            }
        }
        return self.observation, reward, done, info

    def reset(self):
        self.current_frame_index = np.random.randint(0, len(self.sequence))
        self.observation = self.state_preprocess(self.current_frame_index)
        print(self.observation.shape)
        return self.observation

    def render(self):
        pass

    def close(self):
        pass

    @staticmethod
    def seed(seed):
        np.random.seed(seed)

    def state_preprocess(self, index):
        current_frame_path = os.path.join(IMG_PATH, self.sequence[index])
        current_frame = cv2.imread(current_frame_path)
        observation = cv2.resize(current_frame, (FRANE_HEIGHT, FRAME_WIDTH))
        # observation = (observation / 255. - self.mean_vector) / self.std_vector
        observation = observation.transpose(2, 0, 1).reshape(3, FRANE_HEIGHT, FRAME_WIDTH)
        return observation

    def get_detection(self, model, index):
        dets = []
        i = 0
        det_txt = os.path.join(RESULTS_PATH[model], self.sequence[index].replace('.jpg', '.txt'))
        if os.path.exists(det_txt):
            with open(det_txt, 'r') as file:
                for line in file:
                    det_line = line.strip('\n').split()
                    dets.append(list(det_line))
        else:
            return dets
        return dets

    @staticmethod
    def get_reward(dets):
        reward = 0
        if len(dets) == 0:
            reward = -100
            return reward
        for det in dets:
            if float(det[5]) < 0.4:
                continue
            elif 0.4 <= float(det[5]) < 0.5:
                reward += 1
            elif 0.5 <= float(det[5]) < 0.7:
                reward += 4
            elif 0.7 <= float(det[5]) < 1.0:
                reward += 8
        # print(reward)
        return reward

    def get_reward_map(self, model, index):
        gt = os.path.join(GT_PATH, self.gt[index])
        dets = os.path.join(RESULTS_PATH[model], self.sequence[index].replace('.jpg', '.json'))
        reward = self.get_map(gt, dets)
        return reward

    @staticmethod
    def get_map(gt, dets):
        HiddenPrints().close()
        coco_gt = COCO(annotation_file=gt)
        coco_pred = coco_gt.loadRes(dets)
        coco_evaluator = COCOeval(cocoGt=coco_gt, cocoDt=coco_pred, iouType="bbox")
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        return coco_evaluator.stats[2]

    def __str__(self):
        return 'DetQuestEnvironment'


