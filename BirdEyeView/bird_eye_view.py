import cv2
import numpy as np


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def transition_matrix(image_pts, bev_pts):
    rect_image = order_points(image_pts)
    rect_bev = order_points(bev_pts)

    # compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect_image, rect_bev)

    return M


def player_coor(coor_image, M):
    #compute players coordinates on the court
    X = list(coor_image)
    X.append(1)
    X = np.array(X)

    Y = M.dot(X)
    Y = tuple(map(lambda x: round(x / Y[2]), Y))

    return (Y[0], Y[1])


def court_coor(width, pad):
    #compute court court dimensions accorgind to image dimension
    height = int(width * (1 - pad) * 2 * (1 + pad))
    x_1, y_1 = int(width * pad), int(height * pad)
    x_2, y_2 = int(width * (1 - pad)), int(height * pad)
    x_3, y_3 = int(width * (1 - pad)), int(height * (1 - pad))
    x_4, y_4 = int(width * pad), int(height * (1 - pad))

    return [(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)]


class BirdEyeViewCourt:

    def __init__(self, width, pad, line_color=(255, 255, 255), court_color_1=(192, 158, 128),
                 court_color_2=(153, 112, 80)):
        height = int(width * (1 - pad) * 2 * (1 + pad))
        self.court = np.zeros((height, width, 3)).astype(np.uint8)

        cv2.rectangle(self.court, (0, 0), (width, height), court_color_1, -1)

        # baseline
        x_1, y_1 = int(width * pad), int(height * pad)
        x_2, y_2 = int(width * (1 - pad)), int(height * pad)
        x_3, y_3 = int(width * (1 - pad)), int(height * (1 - pad))
        x_4, y_4 = int(width * pad), int(height * (1 - pad))

        cv2.rectangle(self.court, (x_1, y_1), (x_3, y_3), court_color_2, -1)
        cv2.line(self.court, (x_1, y_1), (x_2, y_2), line_color, 2)
        cv2.line(self.court, (x_2, y_2), (x_3, y_3), line_color, 2)
        cv2.line(self.court, (x_3, y_3), (x_4, y_4), line_color, 2)
        cv2.line(self.court, (x_4, y_4), (x_1, y_1), line_color, 2)

        x_ratio = (x_2 - x_1) / 10.97
        y_ratio = (y_3 - y_2) / 23.78

        # doubles sidelines
        xc_1, yc_1 = int(x_1 + x_ratio * 1.372), y_1
        xc_2, yc_2 = int(x_2 - x_ratio * 1.372), y_2
        xc_3, yc_3 = int(x_3 - x_ratio * 1.372), y_3
        xc_4, yc_4 = int(x_4 + x_ratio * 1.372), y_4

        cv2.line(self.court, (xc_1, yc_1), (xc_4, yc_4), line_color, 2)
        cv2.line(self.court, (xc_2, yc_2), (xc_3, yc_3), line_color, 2)

        # service lane
        xs_1, ys_1 = xc_1, int(y_1 + 5.50 * y_ratio)
        xs_2, ys_2 = xc_2, int(y_2 + 5.50 * y_ratio)
        xs_3, ys_3 = xc_3, int(y_3 - 5.50 * y_ratio)
        xs_4, ys_4 = xc_4, int(y_4 - 5.50 * y_ratio)

        cv2.line(self.court, (xs_1, ys_1), (xs_2, ys_2), line_color, 2)
        cv2.line(self.court, (xs_3, ys_3), (xs_4, ys_4), line_color, 2)

        # net
        xnet_1, ynet_1 = x_1, int((y_4 - y_1) / 2 + y_1)
        xnet_2, ynet_2 = x_2, int((y_4 - y_1) / 2 + y_1)

        cv2.line(self.court, (xnet_1, ynet_1), (xnet_2, ynet_2), line_color, 2)

        # center service line
        xv_1, yv_1 = int((x_2 - x_1) / 2 + x_1), ys_1
        xv_2, yv_2 = int((x_2 - x_1) / 2 + x_1), ys_3

        cv2.line(self.court, (xv_1, yv_1), (xv_2, yv_2), line_color, 2)

        # central mark
        xm = int((x_2 - x_1) / 2 + x_1)
        ym_1 = y_1
        ym_2 = int(y_1 + 10)
        ym_3 = int(y_4 - 10)
        ym_4 = y_4

        cv2.line(self.court, (xm, ym_1), (xm, ym_2), line_color, 2)
        cv2.line(self.court, (xm, ym_3), (xm, ym_4), line_color, 2)

    def add_player(self, coor_bev, n_player, color_player_1, color_player_2):
        #display player position with a circle
        x, y = coor_bev
        if n_player == 0:
            cv2.circle(self.court, (x, y), radius=7, color=color_player_1, thickness=-1)
            cv2.circle(self.court, (x, y), radius=7, color=(255, 255, 255), thickness=2)
        elif n_player == 1:
            cv2.circle(self.court, (x, y), radius=7, color=color_player_2, thickness=-1)
            cv2.circle(self.court, (x, y), radius=7, color=(255, 255, 255), thickness=2)

    def add_path_player(self, coor_bev, color_path=(255, 255, 255)):
        #display a new point to the player path
        x, y = coor_bev
        cv2.circle(self.court, (x, y), radius=1, color=color_path, thickness=-1)

