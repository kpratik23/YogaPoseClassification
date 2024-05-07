
from data import BodyPart
import tensorflow as tf
import cv2
import keras
import pandas as pd


class ProcessData:
    
    def __init__(self):
        pass

    def draw_keypoints(self,keypoints,image):
        
        for keypoint in keypoints:
            x,y=keypoint.coordinate.x,keypoint.coordinate.y
            cv2.circle(image, (x,y), 3, (0, 255, 0), -1)
        cv2.imshow("Image",image)
        cv2.waitKey(0)
    
    def load_csv(self,csv_path):
        df = pd.read_csv(csv_path)
        df.drop(['filename'],axis=1, inplace=True)
        classes = df.pop('class_name').unique()
        y = df.pop('class_no')
        
        X = df.astype('float64')
        # y = keras.utils.to_categorical(y)
        return X, y, classes


        
    def get_center_point(self,landmarks, left_bodypart, right_bodypart):
        
        left = tf.gather(landmarks, left_bodypart.value, axis=1)
        right = tf.gather(landmarks, right_bodypart.value, axis=1)
        center = left * 0.5 + right * 0.5
        return center


    def get_pose_size(self,landmarks, torso_size_multiplier=2.5):
        
        hips_center = self.get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                    BodyPart.RIGHT_HIP)
        shoulders_center = self.get_center_point(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
        torso_size = tf.linalg.norm(shoulders_center - hips_center)
        pose_center_new = self.get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
        pose_center_new = tf.expand_dims(pose_center_new, axis=1)
        pose_center_new = tf.broadcast_to(pose_center_new,[tf.size(landmarks) // (17*2), 17, 2])
        d = tf.gather(landmarks - pose_center_new, 0, axis=0,name="dist_to_pose_center")
        max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))
        pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
        return pose_size



    def normalize_pose_landmarks(self,landmarks):
        
        pose_center = self.get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                    BodyPart.RIGHT_HIP)

        pose_center = tf.expand_dims(pose_center, axis=1)
        pose_center = tf.broadcast_to(pose_center, 
                                    [tf.size(landmarks) // (17*2), 17, 2])
        landmarks = landmarks - pose_center
        pose_size = self.get_pose_size(landmarks)
        landmarks /= pose_size
        return landmarks


    def landmarks_to_embedding(self,landmarks_and_scores):
        
        reshaped_inputs = keras.layers.Reshape((17, 2))(landmarks_and_scores)
        landmarks = self.normalize_pose_landmarks(reshaped_inputs[:, :, :2])
        embedding = keras.layers.Flatten()(landmarks)
        return embedding


    def preprocess_single_data(self,X_train):
        
        embedding = self.landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train), (1, 34)))
        embedding=tf.reshape(embedding, (34))
        return embedding
    
    def preprocess_data(self,X_train):
        processed_X_train = []
        for i in range(X_train.shape[0]):
            embedding = self.landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train.iloc[i]), (1, 34)))
            processed_X_train.append(tf.reshape(embedding, (34)))
        return tf.convert_to_tensor(processed_X_train)

