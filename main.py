from movenet import Movenet
import numpy as np
import pandas as pd 
import os
from movenet import Movenet
import wget
from data import BodyPart
import tensorflow as tf
import cv2
import keras
import joblib
from parserargs import parse_args
from processdata import ProcessData
from colorama import Fore, Style


class Predictor(ProcessData):
    
    def __init__(self,algorithm,movenet,model):
        super().__init__()
        
        self.keypoint_pairs_coco = [
        (0, 1),  # Nose to Left Eye
        (0, 2),  # Nose to Right Eye
        (1, 3),  # Left Eye to Left Ear
        (2, 4),  # Right Eye to Right Ear
        (5, 6),  # Left Shoulder to Right Shoulder
        (5, 7),  # Left Shoulder to Left Elbow
        (7, 9),  # Left Elbow to Left Wrist
        (6, 8),  # Right Shoulder to Right Elbow
        (8, 10),  # Right Elbow to Right Wrist
        (11, 12),  # Left Hip to Right Hip
        (11, 13),  # Left Hip to Left Knee
        (13, 15),  # Left Knee to Left Ankle
        (12, 14),  # Right Hip to Right Knee
        (14, 16)  # Right Knee to Right Ankle
        ]
        self.algorithm=algorithm
        self.movenet_thresh=0.3
        self.pose_thresh=0.90
        self.movenet=movenet
        self.model=model
    
    
    def detect(self,input_tensor):
        detection=self.movenet.detect(input_tensor, reset_crop_region=True)
        return detection #it returns the key points
    
    def predict_pose(self,person):
        # image=cv2.imread(image_path)
        # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        classname_to_index={0:"downdog",
                            1:"nopose",
                            2:"tree",
                            3:"warrior"}
        
        pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y] 
                                for keypoint in person.keypoints],dtype=np.float32)
        pose_landmarks = self.preprocess_single_data(pose_landmarks)
        pose_landmarks=tf.reshape(pose_landmarks,(-1,34))
        if (self.algorithm=="nn"):
            predictions=model.predict(pose_landmarks,verbose=False)         
        else:
            predictions=model.predict_proba(pose_landmarks)
        if (predictions[0,:].max()<=self.pose_thresh):
            return "nopose"
        text=classname_to_index[predictions.argmax()]
        return text
        

    def detect_single_image(self,im_path):
        
        frame=cv2.imread(im_path)
        # frame=cv2.flip(frame,1)
        # frame=cv2.resize(frame,(1280,720))
        resized_frame=cv2.resize(frame,(256,256))
        resized_frame=cv2.cvtColor(resized_frame,cv2.COLOR_BGR2RGB)
        

        person = self.detect(resized_frame)
        keypoints=person.keypoints
        
        coords_array=np.zeros((17,3)) #Score of each coordinate

        for i,keypoint in enumerate(keypoints):
            x=keypoint.coordinate.x*frame.shape[1]/256
            y=keypoint.coordinate.y*frame.shape[0]/256
            score=keypoint.score
            coords_array[i][0]=x
            coords_array[i][1]=y
            coords_array[i][2]=score

        text=self.predict_pose(person)
        if (text!="nopose"):
            kp_color=(0,255,0)
        else:
            kp_color=(255,255,255)
        
        for pair in self.keypoint_pairs_coco:
            
            point1=coords_array[pair[0]][:2].astype(int)
            point2=coords_array[pair[1]][:2].astype(int)
            
            if (coords_array[pair[0]][2]>self.movenet_thresh and coords_array[pair[1]][2]>self.movenet_thresh):
                cv2.circle(frame,tuple(point1), 3, kp_color, -1)
                cv2.circle(frame,tuple(point2), 3, kp_color, -1)
                cv2.line(frame, tuple(point1), tuple(point2), kp_color, 2)
            

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255,255,255)
        thickness = 2
        spacing=5

        cv2.rectangle(frame, (50, 50), (200,100), (0, 0, 0), -1)
        cv2.putText(frame, text, (70, 80), font, font_scale, color, thickness,spacing)
        cv2.imshow('Camera Frame', frame)  # Display the frame
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def detect_camera(self,video_path):
        
        video_capture = cv2.VideoCapture(video_path)
        frame_counter=0
        text="None"
        kp_color=(255,255,255)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            if (video_path==0): # the frame is flippped as cv2 reads the frame which is flipped so to correct the frame we flip it
                frame=cv2.flip(frame,1)
            # frame=cv2.resize(frame,(1024,1024))
            resized_frame=cv2.resize(frame,(256,256))
            resized_frame=resized_frame
            resized_frame=cv2.cvtColor(resized_frame,cv2.COLOR_BGR2RGB)
            
            person = self.detect(resized_frame)
            keypoints=person.keypoints
            
            coords_array=np.zeros((17,3))

            for i,keypoint in enumerate(keypoints):
                x=keypoint.coordinate.x*frame.shape[1]/256
                y=keypoint.coordinate.y*frame.shape[0]/256
                score=keypoint.score
                coords_array[i][0]=x
                coords_array[i][1]=y
                coords_array[i][2]=score
                    
            
            if (frame_counter%5==0):
                text=self.predict_pose(person)
                if (text!="nopose"):
                    kp_color=(0,255,0)
                else:
                    kp_color=(255,255,255)
            

            for pair in self.keypoint_pairs_coco:
                
                point1=coords_array[pair[0]][:2].astype(int)
                point2=coords_array[pair[1]][:2].astype(int)
                
                if (coords_array[pair[0]][2]>self.movenet_thresh and coords_array[pair[1]][2]>self.movenet_thresh):
                    cv2.circle(frame,tuple(point1), 3, kp_color, -1)
                    cv2.circle(frame,tuple(point2), 3, kp_color, -1)
                    cv2.line(frame, tuple(point1), tuple(point2), kp_color, 2)
                

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            color = (255,255,255)
            thickness = 2
            spacing=5

            cv2.rectangle(frame, (50, 50), (200,100), (0, 0, 0), -1)
            cv2.putText(frame, text, (70, 80), font, font_scale, color, thickness,spacing)
            cv2.imshow('Camera Frame', frame)  # Display the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
            frame_counter+=1
        cv2.destroyAllWindows()
        
    def drawkeypoints(self,person):
        print(person)
            

if __name__=="__main__":
    args = parse_args()
    
    if('movenet_thunder.tflite' not in os.listdir("./models")):
        wget.download('https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite', './models/movenet_thunder.tflite')

    movenet = Movenet('./models/movenet_thunder')
    
    algorithm=args.algorithm
        
        
    if (algorithm=="nn"):
        model = keras.models.load_model('./models/weights.best.hdf5')
        print(f"{Fore.YELLOW}using nn...")
        Style.RESET_ALL
    else:
        model=joblib.load(f"./models/{args.algorithm}.pkl")
        print(f"{Fore.YELLOW}using {algorithm}...")
        Style.RESET_ALL
        
        
        
        
    predictor=Predictor(algorithm,movenet,model)

        
    if (args.image):
        predictor.detect_single_image(args.image)
    elif (args.video):
        predictor.detect_camera(args.video)
    elif (args.camera):
        predictor.detect_camera(0)
    else:
        print("No input selected")
        print("Exitting...")
        exit(0)
        