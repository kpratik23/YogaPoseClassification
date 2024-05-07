import enum
from typing import List, NamedTuple

import numpy as np


class BodyPart(enum.Enum):
  NOSE = 0
  LEFT_EYE = 1
  RIGHT_EYE = 2
  LEFT_EAR = 3
  RIGHT_EAR = 4
  LEFT_SHOULDER = 5
  RIGHT_SHOULDER = 6
  LEFT_ELBOW = 7
  RIGHT_ELBOW = 8
  LEFT_WRIST = 9
  RIGHT_WRIST = 10
  LEFT_HIP = 11
  RIGHT_HIP = 12
  LEFT_KNEE = 13
  RIGHT_KNEE = 14
  LEFT_ANKLE = 15
  RIGHT_ANKLE = 16


class Point(NamedTuple):
  x: float
  y: float


class Rectangle(NamedTuple):
  start_point: Point
  end_point: Point


class KeyPoint(NamedTuple):
  body_part: BodyPart
  coordinate: Point
  score: float


class Person(NamedTuple):
  keypoints: List[KeyPoint]
  bounding_box: Rectangle
  score: float
  id: int = None


def person_from_keypoints_with_scores(
    keypoints_with_scores: np.ndarray,
    image_height: float,
    image_width: float,
    keypoint_score_threshold: float = 0.1) -> Person:


    kpts_x = keypoints_with_scores[:, 1]
    kpts_y = keypoints_with_scores[:, 0]
    scores = keypoints_with_scores[:, 2]

    # Convert keypoints to the input image coordinate system.
    keypoints = []
    for i in range(scores.shape[0]):
      keypoints.append(
          KeyPoint(
              BodyPart(i),
              Point(int(kpts_x[i] * image_width), int(kpts_y[i] * image_height)), #To normalize
              scores[i]))

    # Calculate bounding box as SinglePose models don't return bounding box. returning rect
    start_point = Point(
        int(np.amin(kpts_x) * image_width), int(np.amin(kpts_y) * image_height))
    end_point = Point(
        int(np.amax(kpts_x) * image_width), int(np.amax(kpts_y) * image_height))
    bounding_box = Rectangle(start_point, end_point)

    # Calculate person score by averaging keypoint scores.
    # scores indicate the confidence of the model in the pose
    scores_above_threshold = list(
        filter(lambda x: x > keypoint_score_threshold, scores))
    person_score = np.average(scores_above_threshold)

    return Person(keypoints, bounding_box, person_score)


class Category(NamedTuple):
  label: str
  score: float
