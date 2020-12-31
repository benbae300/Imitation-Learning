import argparse
from os import close
from sim import Action, WallESim
import pybullet as p
import torch
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from torchvision.models import resnet18
from torch import nn, optim

CLOSE_DISTANCE_THRESHOLD = 0.3

class ImgProcessingActionPredictor:
    def __init__(self):
        self.prevLeft = 0 
        self.prevRight = 0 
        self.prevAction = Action.FORWARD

    def predict_action(self, img):
        '''
        i = Image.fromarray(img, 'RGB')
        i.show()
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if not np.array_equal(img[x, y], np.array([0,0,0])):
                    print(x)
        print('hi')
        '''
        #https://stackoverflow.com/questions/41971663/use-numpy-to-convert-rgb-pixel-array-into-grayscale
        grayscale = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        gray_slice = grayscale[223]
        begin = 0
        end = 0
        idx = 0

        while idx < gray_slice.shape[0] - 1:
            while (gray_slice[idx] != 0.0) and (idx < gray_slice.shape[0] - 1):
                if begin == 0: begin = idx
                else: end = idx
                idx += 1 
            idx += 1 
        mid = (begin + end) / 2

        if mid == 0: action = Action.RIGHT
        elif mid > 120: action = Action.RIGHT
        elif mid < 100: action = Action.LEFT
        else: action = Action.FORWARD 
        # TODO: 
        # ===============================================
        # ===============================================
        if action == Action.FORWARD:
            self.prevLeft = 0
            self.prevRight = 0 
        
        if action == Action.LEFT and self.prevAction == Action.RIGHT:
            self.prevLeft += 1 
        if action == Action.RIGHT and self.prevAction == Action.LEFT:
            self.prevRight += 1 
        self.prevAction = action 
        if self.prevLeft + self.prevRight > 8:
            action = action.FORWARD
            self.prevLeft = 0
            self.prevRight = 0
        return action


class ImitationLearningActionPredictor:
    def __init__(self, model_path, transform=None):
        # TODO: Load model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()


        # ===============================================

    def predict_action(self, img):

        action = Action.FORWARD
        #forward - 0
        # convert img to PIL 
        PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
        '''
        transform = transforms.Compose([
            # resize to 224, toTensor
        transforms.Resize(224),
        transforms.ToTensor(),
        ])'''

        transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transformed = torch.unsqueeze(transform(PIL_image), 0)
        
        output = self.model(transformed)
        #_, output = torch.max(output, 1) #torch.max
        output = torch.argmax(output)
        print(output)
        if output == 0: return Action.BACKWARD
        elif output == 1: return Action.FORWARD
        elif output == 2: return Action.LEFT
        else: return Action.RIGHT 



if __name__ == "__main__":
    parser = argparse.ArgumentParser("HW4: Testing line following algorithms")
    parser.add_argument("--use_imitation_learning", "-uip", action="store_true",
                        help="Algorithm to use: 0->image processing, 1->trained model")
    parser.add_argument("--map_path", "-m", type=str, default="maps/test/map1",
                        help="path to map directory. eg: maps/test/map2")
    parser.add_argument("--model_path", type=str, default="following_model.pth",
                        help="Path to trained imitation learning based action predictor model")
    args = parser.parse_args()

    env = WallESim(args.map_path, load_landmarks=True)

    if args.use_imitation_learning:
        # TODO: Provide transform arguments if any to the constructor
        # =================================================================
        actionPredictor = ImitationLearningActionPredictor(args.model_path)
        # =================================================================
    else:
        actionPredictor = ImgProcessingActionPredictor()

    landmarks_reached = np.zeros(len(env.landmarks), dtype=np.bool)
    assert len(landmarks_reached) != 0
    iteration = 1
    while True:
        env.set_landmarks_visibility(False)
        rgbImg = env.get_robot_view()
        env.set_landmarks_visibility(True)
        action = actionPredictor.predict_action(rgbImg)
        env.move_robot(action)

        position, _ = p.getBasePositionAndOrientation(env.robot_body_id)
        distance_from_landmarks = np.linalg.norm(env.landmarks - position, axis=1)
        closest_landmark_index = np.argmin(distance_from_landmarks)
        if distance_from_landmarks[closest_landmark_index] < CLOSE_DISTANCE_THRESHOLD and not landmarks_reached[closest_landmark_index]:
            landmarks_reached[closest_landmark_index] = True
        print(
            f"[{iteration}] {np.sum(landmarks_reached)} / {len(landmarks_reached)} landmarks reached. "
            f"{distance_from_landmarks[closest_landmark_index]:.2f} distance away from nearest landmark"
        )
        if np.all(landmarks_reached):
            print("All landmarks reached!")
            break
        iteration += 1