import argparse
import queue
import pandas as pd 
import pickle
import imutils
import os
from PIL import Image, ImageDraw
import cv2 
import numpy as np
import torch
import sys

from court_detector import CourtDetector
from Models.tracknet import trackNet
from TrackPlayers.trackplayers import *
from utils import get_video_properties, get_dtype
from detection import *


# parse parameters
parser = argparse.ArgumentParser()

parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default="")

parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--n_classes", type=int)

parser.add_argument("--path_yolo_classes", type=str)
parser.add_argument("--path_yolo_weights", type=str)
parser.add_argument("--path_yolo_config", type=str)

args = parser.parse_args()

input_video_path = args.input_video_path
output_video_path = args.output_video_path

save_weights_path = args.save_weights_path
n_classes = args.n_classes

yolo_classes = args.path_yolo_classes
yolo_weights = args.path_yolo_weights
yolo_config = args.path_yolo_config

if output_video_path == "":
    # output video in same path
    output_video_path = input_video_path.split('.')[0] + "VideoOutput/video_output.mp4"

# get video fps&video size
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
print('fps : {}'.format(fps))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# try to determine the total number of frames in the video file
if imutils.is_cv2() is True :
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
else : 
    prop = cv2.CAP_PROP_FRAME_COUNT
total = int(video.get(prop))

# start from first frame
currentFrame = 0

# width and height in TrackNet
width, height = 640, 360
img, img1, img2 = None, None, None

# load TrackNet model
modelFN = trackNet
m = modelFN(n_classes, input_height=height, input_width=width)
m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
m.load_weights(save_weights_path)

# In order to draw the trajectory of tennis, we need to save the coordinate of previous 7 frames
q = queue.deque()
for i in range(0, 8):
    q.appendleft(None)


# save prediction images as videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# load yolov3 labels
LABELS = open(yolo_classes).read().strip().split("\n")
# yolo net
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# court
court_detector = CourtDetector()

# players tracker
dtype = get_dtype()
detection_model = DetectionModel(dtype=dtype)

# get videos properties
fps, length, v_width, v_height = get_video_properties(video)

frame_i = 0
frames = []

while True:
  ret, frame = video.read()
  frame_i += 1

  if ret:
    if frame_i == 1:
      print('First Frame')
      lines = court_detector.detect(frame)
    else: # then track it
      lines = court_detector.track_court(frame)

    print(frame_i, '\n', lines)

    detection_model.detect_player_1(frame, court_detector)
    detection_model.detect_top_persons(frame, court_detector, frame_i)
    
    for i in range(0, len(lines), 4):
      x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
      cv2.line(frame, (x1,y1),(x2,y2), (0,0,255), 5)
    
    new_frame = cv2.resize(frame, (v_width, v_height))
    frames.append(new_frame)

  else:
    break
video.release()
print('First Video Released')

detection_model.find_player_2_box()

# second part 
player1_boxes = detection_model.player_1_boxes
player2_boxes = detection_model.player_2_boxes
print('Player 1 Boxes :', player1_boxes)


video = cv2.VideoCapture(input_video_path)
frame_i = 0

# while (True):
for img in frames:
    print('Percentage of video processed : {}'.format(round( (currentFrame / total) * 100, 2)))
    # capture frame-by-frame
    video.set(1, currentFrame);
    # ret, img = video.read()
    frame_i += 1

    # detect the ball
    # img is the frame that TrackNet will predict the position
    # since we need to change the size and type of img, copy it to output_img
    output_img = img

    # resize it
    img = cv2.resize(img, (width, height))
    # input must be float type
    img = img.astype(np.float32)

    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    X = np.rollaxis(img, 2, 0)
    # prdict heatmap
    pr = m.predict(np.array([X]))[0]

    # since TrackNet output is ( net_output_height*model_output_width , n_classes )
    # so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
    pr = pr.reshape((height, width, n_classes)).argmax(axis=2)

    # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
    pr = pr.astype(np.uint8)

    # reshape the image size as original input image
    heatmap = cv2.resize(pr, (output_width, output_height))

    # heatmap is converted into a binary image by threshold method.
    ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

    # find the circle in image with 2<=radius<=7
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                               maxRadius=7)


    output_img = mark_player_box(output_img, player1_boxes, currentFrame-1)
    output_img = mark_player_box(output_img, player2_boxes, currentFrame-1)
    
    PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(PIL_image)

    # check if there have any tennis be detected
    if circles is not None:
        # if only one tennis be detected
        if len(circles) == 1:

            x = int(circles[0][0][0])
            y = int(circles[0][0][1])

            # push x,y to queue
            q.appendleft([x, y])
            # pop x,y from queue
            q.pop()

        else:
            # push None to queue
            q.appendleft(None)
            # pop x,y from queue
            q.pop()

    else:
        # push None to queue
        q.appendleft(None)
        # pop x,y from queue
        q.pop()

    # draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
    for i in range(0, 8):
        if q[i] is not None:
            draw_x = q[i][0]
            draw_y = q[i][1]
            bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
            draw = ImageDraw.Draw(PIL_image)
            draw.ellipse(bbox, outline='yellow')
            del draw

    # Convert PIL image format back to opencv image format
    opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
    output_video.write(opencvImage)

    # next frame
    currentFrame += 1


# everything is done, release the video
video.release()
output_video.release()