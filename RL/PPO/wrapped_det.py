import json
import os
import cv2
from loguru import logger
import numpy as np
import gym
from gym import spaces

N_DISCRETE_ACTIONS = 5  # model0: all, model1: seq1, model2: seq2, model3: seq3, model4: seq4
FPS = 30
FRAME_WIDTH = 64
FRANE_HEIGHT = 64
N_CHANNELS = 3
SEQ_PATH = "/media/joshuawen/JoshuaWS3/Datasets/RL/VisDrone2019_VID_COCO/val"
MODEL_PATH = {
    "all": "/home/joshuawen/WorkSpace/YOLOX/YOLOX_outputs/visdrone_vid_l/best_ckpt.pth",
    "seq1": "/home/joshuawen/WorkSpace/YOLOX/YOLOX_outputs/visdrone_vid_l/best_ckpt.pth",
    "seq2": "/home/joshuawen/WorkSpace/YOLOX/YOLOX_outputs/visdrone_vid_l/best_ckpt.pth",
    "seq3": "/home/joshuawen/WorkSpace/YOLOX/YOLOX_outputs/visdrone_vid_l/best_ckpt.pth",
    "seq4": "/home/joshuawen/WorkSpace/YOLOX/YOLOX_outputs/visdrone_vid_l/best_ckpt.pth"
}
RESULTS_PATH = {
    "all": "/home/joshuawen/WorkSpace/YOLOX/YOLOX_outputs/visdrone_vid_l/vis_res/visdrone_vid_l.json",
    "seq1": "/home/joshuawen/WorkSpace/YOLOX/YOLOX_outputs/visdrone_vid_l_seq1/vis_res/visdrone_vid_l_seq1.json",
    "seq2": "/home/joshuawen/WorkSpace/YOLOX/YOLOX_outputs/visdrone_vid_l_seq2/vis_res/visdrone_vid_l_seq2.json",
    "seq3": "/home/joshuawen/WorkSpace/YOLOX/YOLOX_outputs/visdrone_vid_l_seq3/vis_res/visdrone_vid_l_seq3.json",
    "seq4": "/home/joshuawen/WorkSpace/YOLOX/YOLOX_outputs/visdrone_vid_l_seq4/vis_res/visdrone_vid_l_seq4.json"
}


class DetQuest(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DetQuest, self).__init__()
        self.sequence = sorted(os.listdir(SEQ_PATH))
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation = np.zeros((FRANE_HEIGHT, FRAME_WIDTH, N_CHANNELS))
        self.observation_space = spaces.Box(low=0, high=255, shape=(N_CHANNELS, FRANE_HEIGHT, FRAME_WIDTH,))
        self.state_init = self.sequence[0]

        self.current_frame_index = 0
        self.step_count = 0

        self.mean_vector = np.asarray([109, 114, 131], dtype=np.float32).reshape(1, 1, 3)
        self.std_vector = np.asarray([109, 114, 131], dtype=np.float32).reshape(1, 1, 3)

        self.cv_model = None

    def step(self, action):
        if action == 0:
            self.cv_model = "all"
        elif action == 1:
            self.cv_model = "seq1"
        elif action == 2:
            self.cv_model = "seq2"
        elif action == 3:
            self.cv_model = "seq3"
        elif action == 4:
            self.cv_model = "seq4"
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
                "n": self.sequence[self.current_frame_index],
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
        current_frame_path = os.path.join(SEQ_PATH, self.sequence[index])
        current_frame = cv2.imread(current_frame_path)
        observation = cv2.resize(current_frame, (FRANE_HEIGHT, FRAME_WIDTH))
        # observation = (observation / 255. - self.mean_vector) / self.std_vector
        observation = observation.transpose(2, 0, 1).reshape(3, FRANE_HEIGHT, FRAME_WIDTH)
        return observation

    def get_detection(self, model, index):
        dets = []
        i = 0
        # det_txt = os.path.join(RESULTS_PATH[model], self.sequence[index].replace('.jpg', '.txt'))
        with open(RESULTS_PATH[model], "r") as js:
            # print(model)
            res = json.load(js)
            for r in res:
                if r["image_name"] == self.sequence[index]:
                    dets.append(r)
                    i += 1
                    # print("we got {} detections ~".format(i))
        # with open(det_txt, 'r') as file:
        #     for line in file:
        #         det_line = line.strip('\n').split()
        #         det.append(tuple(det_line))
        return dets

    @staticmethod
    def get_reward(dets):
        reward = 0
        if len(dets) <= 20:
            reward = -100
            return reward
        for det in dets:
            if det["score"] < 0.1:
                continue
            elif 0.1 <= det["score"] < 0.5:
                reward += 1
            elif 0.5 <= det["score"] < 0.7:
                reward += 3
            elif 0.7 <= det["score"] < 1.0:
                reward += 6
        # print(reward)
        return reward

    def __str__(self):
        return 'DetQuestEnvironment'


