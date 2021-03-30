import cv2 as cv
import numpy as np
from random import randint

from diffraction_analysis import gmm_unmixing
from diffraction_analysis import find_center
from diffraction_analysis import draw_center
from diffraction_analysis import draw_rects
from diffraction_analysis import draw_blobs

from diffraction_analysis import blob_detection
from diffraction_analysis import optical_flow
from diffraction_analysis import calculate_blob_flow
from diffraction_analysis import update_locations
from diffraction_analysis import mean_image
from diffraction_analysis import mean_image_from_video


from utils import generate_new_color
from utils import update_excel_results
from utils import tupletostring

import config

import xlsxwriter

import sys

save_path = ''
input_video = ""
output_video = save_path + ""

#bar_region0 = (0,0,1,512)
#bar_region1 = (0,172,12,337)
#bar_region2 = (452,13,500,125)
bar_region2 = (470,2,504,79)
#bar_region3 = (0,0,512,1)
#bar_region4 = (511,0,512,512)
#bar_region5 = (0,511,512,512)
regions = []
#regions.append(bar_region0)
#regions.append(bar_region1)
regions.append(bar_region2)
#regions.append(bar_region3)
#regions.append(bar_region4)
#regions.append(bar_region5)

center_results = open(save_path + 'centers.txt', "w")

# Calculate the mean image
mean_img = mean_image_from_video(input_video,regions)
mean_gmm, mean_means = gmm_unmixing(np.uint8(mean_img))
mean_ind = np.argsort(mean_means[:, 0])[::-1]
# Find the center
x_c, y_c, r = find_center(
    (mean_gmm[:, :, mean_ind[2]]+mean_gmm[:, :, mean_ind[1]]+mean_gmm[:, :, mean_ind[0]])*255)
x_c_int = int(x_c + 0.5)
y_c_int = int(y_c + 0.5)
r_int = int(r + 0.5)
print("Center Coordinate: row number->{}, column number->{}".format(y_c, x_c))
print("Estimated radius of transmission spot:{}".format(r))
np.savetxt(center_results, [[x_c, y_c, r]], delimiter='\t', fmt='%4e')
center_results.close()

#figure1 = cv.cvtColor(np.uint8(mean_img*255), cv.COLOR_GRAY2BGR)
# draw circle
#cv.circle(figure1, (x_c_int, y_c_int), r_int, (0, 255, 0), 1)
# draw the center of the circle
#cv.circle(figure1, (x_c_int, y_c_int), 2, (0, 0, 255), 3)
#cv.imwrite('figure1.bmp',figure1)
# The video feed is read in as a VideoCapture object
cap_ori = cv.VideoCapture(input_video)

fps = cap_ori.get(cv.CAP_PROP_FPS)
size = (int(cap_ori.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(cap_ori.get(cv.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv.VideoWriter_fourcc(*'XVID')
video_results = cv.VideoWriter(output_video, fourcc, fps, size)
#cap_seg = cv.VideoCapture(
#    "D:/GMS/Documents/Dong/data/unet_training/15_DP3/15_DP3_unet.avi")
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence

results = xlsxwriter.Workbook(save_path + 'tracking_results.xlsx')
sheet1 = results.add_worksheet('tracking')
index_pattern = results.add_format()
index_pattern.set_align('center')
index_pattern.set_bold()
cell_pattern = results.add_format()
cell_pattern.set_text_wrap()
frame_pattern = results.add_format()
frame_pattern.set_pattern(1)
frame_pattern.set_bg_color('yellow')
frame_pattern.set_align('center')
frame_pattern.set_bold()

ret_ori, first_frame_ori = cap_ori.read()
#ret_seg, first_frame_seg = cap_seg.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray_ori = cv.cvtColor(first_frame_ori, cv.COLOR_BGR2GRAY)
if len(regions) > 0:
    for region in regions:
        h1, w1, h2, w2 = region
        prev_gray_ori[h1:h2, w1:w2] = 0
        first_frame_ori[h1:h2, w1:w2, :] = 0
# Gaussian unmixing
#unmix, means = gmm_unmixing(first_frame_ori, components=5)
#ind = np.argsort(means[:,0])[::-1]

#prev_gray_seg = cv.cvtColor(first_frame_seg, cv.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame_ori)
# Sets image saturation to maximum
mask[..., 1] = 255

# Detect blobs.
#ind = np.argsort(means[:, 0])[::-1]
#blob_detection_image = unmix[:,:,ind[0]] + unmix[:,:,ind[1]]
#blob_detection_image = np.uint8(unmix[:, :, ind[0]]*255)
bboxes, _, _ = blob_detection(first_frame_ori, y_c, x_c, r)
# Counting the number of blobs
# Initialize locations with the first frame
# labels: matched->0, new_matched->1, disappeared->2, appeared->3
# labels are initialized as matched at first
count = 0

for i in range(len(bboxes)):
    #    config.locations.append(bboxes[i])
    #    config.colors.append(
    #        (randint(64, 255), randint(64, 255), randint(64, 255)))
    config.labels.append(0)
    # in bboxes_config, [0-3] is the location, [4-6] is the color, [7] is the label
    new_color = tuple(generate_new_color(config.colors, pastel_factor=0.5))
    config.colors.append(new_color)
    config.bboxes_config.append(
        bboxes[i] + tuple(256*x for x in new_color) + (0,) + (config.index,))
    config.index = config.index + 1

    # initialize the excel sheet

    sheet1.write(i + 1, 0, i, index_pattern)
    sheet1.write(i + 1, 1, tupletostring(bboxes[i]), cell_pattern)

    count = count + 1
config.current_number_blobs = count

# Visualize the detections of the first frame
#first_detection = draw_blobs(first_frame_ori)
#cv.imwrite('0_detection.bmp', first_detection)

pre_frame_ori = first_frame_ori

# Tracking part
num = 0
sheet1.write(0, num + 1, '0', frame_pattern)

#results.save('tracking_results.xls')

while(cap_ori.isOpened()):

    #config.bboxes_config.sort(key=lambda x: x[2]*x[3], reverse=True)
    config.bboxes_config.sort(key=lambda x: np.sum(prev_gray_ori[int(x[0]+0.5):int(x[0]+x[2]+0.5),int(x[1]+0.5):int(x[1]+x[3]+0.5)]), reverse=True)

    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret_ori, frame_ori = cap_ori.read()
    #ret_seg, frame_seg = cap_seg.read()
    # Opens a new window and displays the input frame
    if ret_ori is not True:
        break
    cv.imshow("input", frame_ori)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray_ori = cv.cvtColor(frame_ori, cv.COLOR_BGR2GRAY)
    if len(regions) > 0:
        for region in regions:
            h1, w1, h2, w2 = region
            gray_ori[h1:h2, w1:w2] = 0
            frame_ori[h1:h2, w1:w2, :] = 0
    #gray_seg = cv.cvtColor(frame_seg, cv.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method in opencv, get the magnitudes and angles
    mask[..., 1] = 255

    #pre_norm = normalize_image(prev_gray_seg)
    #norm = normalize_image(gray_seg)
    #unmix_cur, means_cur = gmm_unmixing(frame_ori, components=5)
    #ind_cur = np.argsort(means_cur[:, 0])[::-1]
    #blob_detection_image_cur = unmix_cur[:, :, ind_cur[0]] + unmix_cur[:, :, ind_cur[1]]
    #blob_detection_image_cur = np.uint8(unmix_cur[:, :, ind_cur[0]]*255)

    flow = optical_flow(prev_gray_ori,
                        gray_ori, mask=mask)

    # Calculate the dx, dy for each blob and update motions
    motions = calculate_blob_flow(prev_gray_ori, flow)

    # Find the center

    #x_c_cur, y_c_cur, _ = find_center(unmix_cur[:,:,ind_cur[1]]*255)
    #_,_,r_cur = find_center(unmix_cur[:, :, ind_cur[0]]*255)
    #x_c_int_cur = int(x_c_cur + 0.5)
    #y_c_int_cur = int(y_c_cur + 0.5)
    #r_int_cur = int(r_cur + 0.5)

    # Detect blobs of the current frame
    bboxes, _, _ = blob_detection(frame_ori, y_c, x_c, r)
   # print(len(bboxes))

    pre_bboxes = config.bboxes_config.copy()

    # Tracking
    # Calibrate the blobs between frames according to blob flow
    if len(bboxes) > 0:
        trustful_blobs, disappear_blobs, new_blobs, new_current_number, flags = update_locations(
            prev_gray_ori, gray_ori, bboxes, motions, sheet1, index_pattern)
    #update_locations_V2(bboxes)

    # draw detection and tracking results on the previous frame
    detection = draw_blobs(pre_frame_ori, pre_bboxes)


    # draw circle
    cv.circle(detection, (x_c_int, y_c_int), r_int, (0, 255, 0), 1)
    # draw the center of the circle
    cv.circle(detection, (x_c_int, y_c_int), 2, (0, 0, 255), 3)
    # draw the outer rectangular
    #cv2.drawContours(detection, [box], 0, (255, 0, 0), 1)
    cv.imshow('MultiTracker', detection)
    cv.waitKey(100)

    #cv.imwrite(save_path + '{}_detection.bmp'.format(num), detection)
    video_results.write(detection)
    num = num + 1
    update_excel_results(sheet1, num, cell_pattern, frame_pattern)
    
#    if num == 5:
#        results.close()
#        sys.exit()
 #   print(config.current_number_blobs)

 #   print(len(disappear_blobs))
 #   print(len(new_blobs))

    prev_gray_ori = gray_ori
    #lob_detection_image = blob_detection_image_cur
    pre_frame_ori = frame_ori
    #x_c_int = x_c_int_cur
    #y_c_int = y_c_int_cur
    #r_int = r_int_cur
    #x_c = x_c_cur
    #y_c = y_c_cur
    #r = r_cur
    #print("Center Coordinate: row number->{}, column number->{}".format(y_c, x_c))
    #print("Estimated radius of transmission spot:{}".format(r))

results.close()

# The following frees up resources and closes all windows
cap_ori.release()
video_results.release()
#cap_seg.release()

cv.destroyAllWindows()
