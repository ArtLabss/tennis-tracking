import cv2
import numpy as np
import pandas as pd
import pickle

from copy import deepcopy

from BirdEyeView.bird_eye_view import *

# animation output dimension
output_width = 1000
pad = 0.22
output_height = int(output_width * (1 - pad) * 2 * (1 + pad))

# coordinates of the 4 corners of the court
image_pts = np.array([(574, 307), 
	                  (1338, 307),
	                  (1566, 871),
	                  (363, 871)]).reshape(4, 2)
bev_pts = np.array(court_coor(output_width, pad)).reshape(4, 2)

# homography matrix to go from real world image to bird eye view
M = transition_matrix(image_pts, bev_pts)

# players positions in bird eye view
positions_df = pd.read_csv('tracking_players.csv')
positions_df['cp_0'] = list(zip(positions_df.x_0, positions_df.y_0))
positions_df['cp_1'] = list(zip(positions_df.x_1, positions_df.y_1))
positions_df['coor_bev_0'] = positions_df['cp_0']\
			.apply(lambda x: player_coor(x, M))
positions_df['coor_bev_1'] = positions_df['cp_1']\
			.apply(lambda x: player_coor(x, M))
positions_0 = list(positions_df['coor_bev_0'])
positions_1 = list(positions_df['coor_bev_1'])


output_video_path = '/VideoOutput/output_bird_eye_view.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# better if the fps is the same as the one of the original video
fps = 60

output_video = cv2.VideoWriter(output_video_path, fourcc, fps,
							  (output_width, output_height))

# define a court to trace the player's path
court_base = BirdEyeViewCourt(output_width, pad)

i = 0
while (True):

    if len(positions_0) == i:
        break

    # copy instance in order not to have an inheritance
    court = deepcopy(court_base)

    # players positions at each frame
    court.add_player(positions_0[i], 0,
    				(255, 0, 0), (0, 0, 0))
    court.add_player(positions_1[i], 1,
    				(38, 19, 15), (0, 0, 0))

    # players positions at each frame added to the path
    court_base.add_path_player(positions_0[i])
    court_base.add_path_player(positions_1[i])

    output_video.write(court.court)
    i += 1

output_video.release()

