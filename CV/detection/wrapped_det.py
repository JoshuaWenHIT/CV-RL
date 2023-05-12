import os
import cv2

import numpy as np
import CV.detection.det_utils
import gym
from gym import spaces

N_DISCRETE_ACTIONS = 3  # model0: sc1 + sc4, model1: sc1, model2: sc4
FPS = 30
FRAME_WIDTH = 640
FRANE_HEIGHT = 640
N_CHANNELS = 3
SEQ_PATH = "../.././data/datasets/VisDrone-demo/images"
MODEL_PATH = {
    "sc1+sc4": "../.././data/datasets/VisDrone-demo/weights/sc1+sc4.pth",
    "sc1": "../.././data/datasets/VisDrone-demo/weights/sc1.pth",
    "sc4": "../.././data/datasets/VisDrone-demo/weights/sc4.pth"
}
RESULTS_PATH = {
    "sc1+sc4": "../.././data/datasets/VisDrone-demo/detection/sc1+sc4",
    "sc1": "../.././data/datasets/VisDrone-demo/detection/sc1",
    "sc4": "../.././data/datasets/VisDrone-demo/detection/sc4"
}


class DetQuest(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DetQuest, self).__init__()
        self.sequence = sorted(os.listdir(SEQ_PATH))
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation = np.zeros((FRANE_HEIGHT, FRAME_WIDTH, N_CHANNELS))
        self.observation_space = spaces.Box(low=0, high=255, shape=(FRANE_HEIGHT, FRAME_WIDTH, N_CHANNELS))
        self.state_init = self.sequence[0]

        self.current_frame_index = 0
        self.step_count = 0

        self.mean_vector = np.asarray([109, 114, 131], dtype=np.float32).reshape(1, 1, 3)
        self.std_vector = np.asarray([109, 114, 131], dtype=np.float32).reshape(1, 1, 3)

        self.cv_model = None

    def step(self, action):
        if action == 0:
            self.cv_model = "sc1+sc4"
        elif action == 1:
            self.cv_model = "sc1"
        elif action == 2:
            self.cv_model = "sc4"
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
        info = {}
        return self.observation, reward, done, info

    def reset(self):
        self.current_frame_index = np.random.randint(0, len(self.sequence))
        self.observation = self.state_preprocess(self.current_frame_index)
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
        observation = (observation - self.mean_vector) / self.std_vector
        return observation

    def get_detection(self, model, index):
        det = []
        det_txt = os.path.join(RESULTS_PATH[model], self.sequence[index].replace('.jpg', '.txt'))
        with open(det_txt, 'r') as file:
            for line in file:
                det_line = line.strip('\n').split()
                det.append(tuple(det_line))
        return det

    @staticmethod
    def get_reward(detections):
        reward = 0
        if len(detections) == 0:
            reward = -100
            return reward
        for det in detections:
            if float(det[5]) < 0.1:
                continue
            elif 0.1 <= float(det[5]) < 0.5:
                reward += 1
            elif 0.5 <= float(det[5]) < 0.7:
                reward += 3
            elif 0.7 <= float(det[5]) < 1.0:
                reward += 6
        return reward

    def __str__(self):
        return 'DetQuestEnvironment'


