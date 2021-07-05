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

# sys.path.append(os.path.dirname(__file__))
from modells.mbv2_mlsd_tiny import  MobileV2_MLSD_Tiny
from modells.mbv2_mlsd_large import  MobileV2_MLSD_Large

from utils import  pred_lines

from Models.tracknet import trackNet
from TrackPlayers.trackplayers import *


print(2)
def draw_line(img):
    current_dir = os.getcwd()
    dsize = img.shape[1], img.shape[0]
    # model_path = current_dir+'/modells/mlsd_tiny_512_fp32.pth'
    # model = MobileV2_MLSD_Tiny().cuda().eval()

    model_path = current_dir + '/modells/mlsd_large_512_fp32.pth'
    model = MobileV2_MLSD_Large().cuda().eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    # img_fn = current_dir+'/data/lines.png'

    # img = cv2.imread(img_fn)
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lines = pred_lines(img, model, [512, 512], 0.1, 20)

    for l in lines:
        cv2.line(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0,0,255), 1,16)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, dsize)
    return img

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
    output_video_path = input_video_path.split('.')[0] + "VideoOutput/video_output.avi"

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
# Tutorial: https://stackoverflow.com/questions/33631489/error-during-saving-a-video-using-python-and-opencv
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# load yolov3 labels
LABELS = open(yolo_classes).read().strip().split("\n")
# yolo net
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# players tracker
ct_players = CentroidTracker()
# append players positions at each frame
players_positions = {'x_0': [], 'y_0': [], 'x_1': [], 'y_1': []}

while (True):

    print('Percentage of video processed : {}'.format(round( (currentFrame / total) * 100, 2)))

    # capture frame-by-frame
    video.set(1, currentFrame);
    ret, img = video.read()


    # if there don't have any frame in video, break
    if not ret:
        break

    # detect players
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(img, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    detected_players = predict_players(outs, LABELS, img, 0.8)

    # track players with a unique ID
    formate_detected_players = list(map(update_boxes, list(detected_players)))
    players_objects = ct_players.update(formate_detected_players)

    # players positions frame by frame
    players_positions['x_0'].append(tuple(players_objects[0])[0])
    players_positions['y_0'].append(tuple(players_objects[0])[1])
    players_positions['x_1'].append(tuple(players_objects[1])[0])
    players_positions['y_1'].append(tuple(players_objects[1])[1])


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
    # .argmax( axis=2 ) => select the largest probability as class
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

    # draw court lines
    output_img = draw_line(output_img)

    # draw players boxes
    color_box = (0, 0, 255)
    if len(detected_players) > 0:
        for box in detected_players:
            x, y, w, h = box
            cv2.rectangle(output_img, (x, y), (x + w, y + h), color_box, 2)

    # draw tracking id of each player
    for (objectID, centroid_player)in players_objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(output_img, text, (centroid_player[0] - 50, centroid_player[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.circle(output_img, (centroid_player[0], centroid_player[1]), 1, (0, 255, 0), 2)

    # draw balls circle
    # In order to draw the circle in output_img, we need to used PIL library
    # Convert opencv image format to PIL image format
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
    # write image to output_video
    output_video.write(opencvImage)

    # next frame
    currentFrame += 1


# everything is done, release the video
video.release()
output_video.release()

# players positions
df_players_positions = pd.DataFrame()
df_players_positions['x_0'] = players_positions['x_0']
df_players_positions['y_0'] = players_positions['y_0']
df_players_positions['x_1'] = players_positions['x_1']
df_players_positions['y_1'] = players_positions['y_1']
df_players_positions.to_csv("tracking_players.csv")

