import cv2
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from skimage.morphology import opening, square, closing, erosion
from skimage.feature import blob_log
from random import randint
import config
from collections import Counter

from utils import generate_new_color

import sys
import os



def mean_image_from_video(video, bar_regions = None):
    if not os.path.exists(video):
        print("Error: {} does not exist".format(video))
        sys.exit(2)
    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)

    H,W = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    mean = np.zeros((H, W))
    success, frame = cap.read()   
    count = 0
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if bar_regions is not None:
            for region in bar_regions:
                h1,w1,h2,w2 = region
                frame[h1:h2,w1:w2] = 0
        mean = mean + frame
        success, frame = cap.read()
        count = count + 1
    cap.release()
    mean = mean/count
    return mean

def mean_image(path, H, W, suffix='.bmp'):
    if not os.path.exists(path):
        print("Error: path {} does not exist".format(path))
        sys.exit(2)
    listing = os.listdir(path)
    mean = np.zeros((H, W))
    count = 1
    for name in listing:
        if name.endswith(suffix):
            image = cv2.imread(path+name)
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean = mean + image/255.
            count = count + 1
    mean = mean/count
    return mean

def gmm_unmixing(frame, components=10, covariance_type='full'):
    if len(frame.shape)>2:
        H, W, _ = frame.shape
    
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        H, W = frame.shape
        frame_gray = frame
    frame_gray = frame_gray.astype(float) / 255.
    image_gray = np.array(frame_gray.astype('float64'))

#    image_gray[446: 498, 11: 131] = 0
#    image_gray[2: 12, 173: 336] = 0
    gmm_components = components
    data = image_gray.reshape(-1, 1)
    gmm = mixture.GaussianMixture(
        n_components=gmm_components, max_iter = 20000).fit(data)
    ind = np.argmax(gmm.means_)
    prediction = gmm.predict_proba(data).reshape(H, W, gmm_components)
    return prediction, gmm.means_


def gmm_unmixing_iter(frame, iters):

    prediction, means, score = gmm_unmixing(frame)
    for i in range(iters - 1):
        prediction_temp, means_temp, score_temp = gmm_unmixing(frame)
        if score_temp < score:
            prediction = prediction_temp
            means = means_temp
            score = score_temp
    
    return prediction, means

def find_center(data):
    #bw = opening(gmm_unmix[:,:,ind]*255, square(5))
    ret, thresh = cv2.threshold(np.uint8(data), 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_ind = 0
    areas = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        areas.append(area)
    area_ind = np.argsort(areas)[::-1]
    if areas[area_ind[1]] < 200:
        (x, y), radius = cv2.minEnclosingCircle(contours[area_ind[0]])
    else:
        (x, y), radius = cv2.minEnclosingCircle(np.concatenate(
            (contours[area_ind[0]], contours[area_ind[1]]), axis=0))
    return x, y, radius
 

def refine_center(prediction, means):
    indx = np.argsort(means[:, 0])[::-1]
    x_c, y_c, r, box, rect = find_center(prediction[:, :, indx[0]]*255)
    circular_area = np.pi * (r**2)
    rect_area = rect[1][0] * rect[1][1]
    if circular_area > 2 * rect_area:
        x_c, y_c, r, box, rect = find_center(prediction[:, :, indx[1]]*255)
    return x_c, y_c, r, box


def blob_detection(image, y_c, x_c, radius, radius_e_r=-5, max_sigma = 2):
    image_gray = image.copy()
    if len(image_gray.shape)>2:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = np.array(image_gray.astype('float64')/255.)
    blobs_log = blob_log(image_gray, min_sigma=1, max_sigma=max_sigma,
                         num_sigma=50, threshold=.11, overlap=0.1, log_scale=False)
    height, width = image_gray.shape
    bboxes = []
    colors = []
    features = []
    e_r = 0
    r_w = np.sqrt(2)
    for blob in blobs_log:
        y, x, r = blob
        r = r_w * r
        if np.sqrt((y - y_c)**2+(x - x_c)**2) > radius + radius_e_r:
            features.append((y, x, r))

            y_min = max(y-r-e_r, 0)
            y_max = min(y+r+e_r, height)
            x_min = max(x-r-e_r, 0)
            x_max = min(x+r+e_r, width)
            bboxes.append((y_min, x_min, y_max - y_min, x_max - x_min))
            colors.append(
                (randint(64, 255), randint(64, 255), randint(64, 255)))
    return bboxes, colors, features

def draw_center(frame, y_c, x_c, r, box, replace = False):
    if replace:
        display = frame
    else:
        display = frame.copy()

    x_c_int = int(x_c + 0.5)
    y_c_int = int(y_c + 0.5)
    r_int = int(r + 0.5)

    # draw circle
    cv2.circle(display, (x_c_int, y_c_int), r_int, (0, 255, 0), 1)
    # draw the center of the circle
    cv2.circle(display, (x_c_int, y_c_int), 2, (0, 0, 255), 3)
    # draw the outer rectangular
    cv2.drawContours(display, [box], 0, (255, 0, 0), 1)
    if replace:
        return 
    else:
        return display


def draw_rects(frame, bboxes, colors):
    display = frame.copy()
    count = 0
    for bbox in bboxes:
        # y->rows, x->cols
        y, x, height, width = bbox
        color = colors[count]
        cv2.rectangle(display, (x, y), (x+width, y+height), color, 1)
        count = count + 1
    return display


def draw_blobs(frame, bboxes=None, flags=None):
    i = 0

    #cv2.circle(frame, center, radius, (0, 255, 0), 2)
    display_image = frame.copy()
    i = 0
    if bboxes:
        for bbox in bboxes:
            # y->rows, x->cols
            y, x, height, width = bbox[0:4]
            y = int(y+0.5)
            x = int(x+0.5)
            height = int(height+0.5)
            width = int(width+0.5)
            if config.draw_trust_labels:
                color = config.label_color[bbox[7]]
            else:
                color = bbox[4:7]
            if flags is not None:
                if flags[i] == 1:
                    cv2.rectangle(display_image, (x, y),
                                  (x+width, y+height), color, 1)
            else:
                cv2.rectangle(display_image, (x, y),
                              (x+width, y+height), color, 1)
            i = i + 1
    else:
        for blob in config.bboxes_config:
            y, x, height, width = blob[0:4]
            y = int(y+0.5)
            x = int(x+0.5)
            height = int(height+0.5)
            width = int(width + 0.5)
            if config.draw_trust_labels:
                color = config.label_color[blob[7]]
            else:
                color = blob[4:7]
            if flags is not None:
                if flags[i] == 1:
                    cv2.rectangle(display_image, (x, y),
                                  (x+width, y+height), color, 1)
            else:
                cv2.rectangle(display_image, (x, y),
                              (x+width, y+height), color, 1)
            i = i+1
    cv2.imshow('MultiTracker', display_image)
    cv2.waitKey(100)
    return display_image

def optical_flow(prev_gray, gray, mask=None):
    # Calculate dense optical flow by Farneback method
    # flow[0] is horizontal displacement, flow[1] is vertical displacement
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=1,
                                        winsize=3, iterations=5, poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    angle = angle*180/np.pi/2
    #mask[..., 0] = angle
    #mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    #rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    #normalizedImg = np.zeros_like(gray)
    #normalizedImg = cv2.normalize(magnitude,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    #cv2.imshow('flow', rgb)
    #cv2.waitKey(0)
    return flow


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0


def calculate_blob_flow(prev_gray, flow, threshold=10):
    size = 0
    dx_blob = 0
    dy_blob = 0
    motions = []
    for index in range(config.current_number_blobs):
        y, x, y_len, x_len = config.bboxes_config[index][0:4]
        #for i in range(x, x+x_len):
        #    for j in range(y, y+y_len):
                # Determine whether the pixel is a blob or not
        #        if prev_gray[j, i] > threshold:
        #            size = size + 1
        #            dx, dy = flow[j, i]
        #            dx_blob = dx_blob + dx
        #            dy_blob = dy_blob + dy
#        dx_blob = dx_blob/size
#        dy_blob = dy_blob/size
        center_y = int(y + y_len/2 + 0.5)
        center_x = int(x + x_len/2 + 0.5)
#       dx_blob, dy_blob = np.max(flow[y:y+y_len,x:x+x_len].transpose(2,0,1).reshape(2,-1),1)
        dx_blob, dy_blob = flow[center_y, center_x]
        motions.append((dx_blob, dy_blob))

        size = 0
        dx_blob = 0
        dy_blob = 0

    return motions


def generate_label_mask(gray, bboxes, current_number=0):
    label_mask = np.zeros_like(gray)
    if current_number > 0:
        for i in range(current_number):
            y, x, y_len, x_len = bboxes[i]
            y = int(y+0.5)
            x = int(x+0.5)
            y_len = int(y_len+0.5)
            x_len = int(x_len + 0.5)
            label_mask[y:y+y_len, x:x+x_len] = i + 1
    else:
        i = 0
        for location in bboxes:
            y, x, y_len, x_len = location
            y = int(y+0.5)
            x = int(x+0.5)
            y_len = int(y_len+0.5)
            x_len = int(x_len + 0.5)
            label_mask[y:y+y_len, x:x+x_len] = i + 1
            i = i + 1
    return label_mask


def update_locations(prev_gray, gray, bboxes, motions, saving_excel, index_pattern, overlap_ratio=0.3):
    #prev_label_mask = generate_label_mask(prev_gray, config.locations, config.current_number_blobs)
    label_mask = generate_label_mask(gray, bboxes)

    new_current_number = 0

    # The flags denote that the blob location is updated or not
    previous_flags = np.zeros(config.current_number_blobs)
    current_flags = np.zeros(len(bboxes))

    # trustful blobs labels
    trustful_blobs = []
    # disappear blobs labels, or, blobs that are not tracked successfully
    disappear_blobs = []
    disappear_colors = []
    # new coming blobs
    new_blobs = []

    # tracking flags
    flags = np.ones(config.current_number_blobs)

    # Generate new label mask using the prev_label_mask and the motions of blobs
    for i in range(config.current_number_blobs):
        y, x, y_len, x_len = config.bboxes_config[i][0:4]
        if np.sqrt(motions[i][1]**2 + motions[i][0]**2)< config.blobs_distance and y >= 0:
            new_y = int(np.around(y + motions[i][1]))
            new_x = int(np.around(x + motions[i][0]))
            if new_y >= config.height or new_x >= config.width:
                continue
            new_y = max(0, new_y - config.motion_bias)
            new_x = max(0, new_x - config.motion_bias)
            y_len = y_len + config.motion_bias
            x_len = x_len + config.motion_bias

            # calculate the label of the blob in the current frame, while the original label in the previous
            # frame should be i+1
            # then the blob of label i+1 in the previous should be the blob of new_label in the current frame
            count_label = Counter(
                label_mask[int(new_y+0.5):int(new_y+y_len+0.5), int(new_x+0.5):int(new_x+x_len+0.5)].ravel()).most_common()

            new_label = count_label[0][0]
            if new_label == 0:
                if len(count_label) > 1:
                    new_label = count_label[1][0]
                else:
                    continue

            # Calculate the overlap ratio between the bounding box of the previous and the one in the current
            new_bbox = (new_y, new_x, new_y + y_len, new_x + x_len)
            current_y, current_x, current_y_len, current_x_len = bboxes[new_label - 1]
            current_bbox = (current_y, current_x, current_y +
                            current_y_len, current_x + current_x_len)
            iou = compute_iou(new_bbox, current_bbox)
            if iou >= overlap_ratio:
                # Update the locations
                current_color = config.bboxes_config[i][4:7]
                current_index = config.bboxes_config[i][8]
                config.bboxes_config[i] = bboxes[new_label -
                                                 1] + current_color + (0,) + (current_index,)
#                config.bboxes_config[i][7] = 1 # trustful detection
                #config.locations[i] = bboxes[new_label - 1]
                new_current_number = new_current_number + 1
                previous_flags[i] = 1
                current_flags[new_label - 1] = 1

                trustful_blobs.append(i)

    # For those blobs that are not tracked, use the geometric locations to search for the nearest blobs
    # in the current unmatched blobs, and compare their blob sizes.

    # the average and maximum movement of all the blobs
    #mean_dx, mean_dy = np.mean(motions, 0)
    max_dx, max_dy = np.max(motions, 0)

    for i in range(config.current_number_blobs):
        if previous_flags[i] == 0:
            y, x, y_len, x_len = config.bboxes_config[i][0:4]
            if y >= 0:
                center_y = y + y_len/2
                center_x = x + x_len/2
                distance = 1e10
                new_label = -1
                for j in range(len(bboxes)):
                    if current_flags[j] == 0:
                        y_c, x_c, y_len_c, x_len_c = bboxes[j]
                        if abs(y_c - y) < abs(max_dy) and abs(x_c - x) < abs(max_dx):
                            size_ratio = y_len * x_len / (y_len_c * x_len_c)
                            if size_ratio > 1:
                                size_ratio = 1/size_ratio
                            if size_ratio > config.size_remaining:
                                center_y_c = y_c + y_len_c/2
                                center_x_c = x_c + x_len_c/2
                                dis_c = np.sqrt(
                                    (center_y-center_y_c)**2 + (center_x-center_x_c)**2)
                                if dis_c < distance and dis_c < config.distance_threshold:
                                    new_label = j
                                    distance = dis_c
                if new_label >= 0:
                    current_color = config.bboxes_config[i][4:7]
                    current_index = config.bboxes_config[i][8]
                    config.bboxes_config[i] = bboxes[new_label] + \
                        current_color + (1,) + (current_index,
                                                )  # new matching results
                    #config.locations[i] = bboxes[new_label]
                    new_current_number = new_current_number + 1
                    previous_flags[i] = 1
                    current_flags[new_label] = 1
                else:
                    disappear_blobs.append((y, x, y_len, x_len))

#                    disappear_colors.append(config.colors[i])
                    config.bboxes_config[i] = (-1, -
                                               1, -1, -1, -1, -1, -1, -1, -1)
                    flags[i] = 0
                    #config.locations[i] = (-1,-1,-1,-1)
                    #config.colors[i] = (-1,-1,-1)
                    #config.locations.pop(i)
                    #config.colors.pop(i)
    count = 0
    for location in reversed(config.bboxes_config):
        y, x, y_len, x_len, _, _, _, _, _ = location
        if y < 0:
            config.bboxes_config.remove(location)

    for color in reversed(config.colors):
        b, g, r = color
        if b < 0:
            config.colors.remove(color)

    for i in range(len(bboxes)):
        if current_flags[i] == 0:
            new_color = tuple(generate_new_color(
                config.colors, pastel_factor=0.5))
            config.colors.append(new_color)
            config.bboxes_config.append(
                bboxes[i] + tuple(256*x for x in new_color) + (2,) + (config.index,))
            saving_excel.write(config.index + 1, 0,
                               config.index, index_pattern)
            config.index = config.index + 1

            # new coming blobs
#            config.locations.append(bboxes[i])
#            config.colors.append(
#                (randint(64, 255), randint(64, 255), randint(64, 255)))
            new_blobs.append(new_current_number)
            new_current_number = new_current_number + 1
            current_flags[i] = 1
    config.current_number_blobs = new_current_number

    return trustful_blobs, disappear_blobs, new_blobs, new_current_number, flags
