'''
Purpose: visualize data from the dataframe.
- Write predictions on frames.
- Generate video from annotated frames.
'''
import glob
import os
import argparse
import cv2
import numpy as np
import pandas as pd

import os
import os
from PIL import Image

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from helper import *
#
from helper import *
#
# w_img = cv2.imread('images/warning.jpeg',1)
# h, w = int(w_img.shape[0]*0.3), int(w_img.shape[1]*0.3)
# w_img = cv2.resize(w_img, (w,h),interpolation= cv2.INTER_AREA)
# w_img = Image.fromarray(w_img)

w_img = cv2.imread('D:/Irascience/rahisho/object_detection-PLUS-distance_estimation-v1/line_detection/warning.jpeg', 1)
h, w = int(w_img.shape[0] * 0.3), int(w_img.shape[1] * 0.3)
w_img = cv2.resize(w_img, (w, h), interpolation=cv2.INTER_AREA)
w_img = Image.fromarray(w_img)

def detect_lines(img, debug=False):
    global VERTS
    ysize, xsize = img.shape[0], img.shape[1]

    # hls_img = filter_img_hsv(img)
    # gray = grayscale(hls_img)
    # cv2.imshow(' HLS ', hls_img)

    blur_gray = gaussian_blur(grayscale(img), kernel_size=5)

    # cv2.imshow(' gray ', blur_gray)

    ht = 150  # First detect gradients above. Then keep between low and high if connected to high
    lt = ht // 3  # Leave out gradients below
    canny_edges = canny(blur_gray, low_threshold=lt, high_threshold=ht)
    if debug: save_img(canny_edges, 'canny_edges_{0}'.format(index))

    # cv2.imshow('canny', canny_edges)
    # Our region of interest will be dynamically decided on a per-image basis
    regioned_edges, region_lines = region_of_interest(canny_edges)

    # cv2.circle(frame,(p1,p2), 5, (0,255,0),-1)
    # print ()
    # print (line_info[0][0]*p1+ p2 + line_info[0][1])

    rho = 2
    theta = 3 * np.pi / 180
    min_line_length = xsize // 16
    max_line_gap = min_line_length // 2
    threshold = min_line_length // 4
    lines, VERTS = hough_lines(regioned_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # print (VERTS)

    # Let's combine the hough-lines with the canny_edges to see how we did

    overlayed_lines = weighted_img(img, lines)
    # overlayed_lines = weighted_img(weighted_img(img, region_lines, a=1), lines)
    if debug: save_img(overlayed_lines, 'overlayed_lines_{0}'.format(index))

    return overlayed_lines

#

import os
from PIL import Image

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from helper import *
#
# w_img = cv2.imread('images/warning.jpeg',1)
# h, w = int(w_img.shape[0]*0.3), int(w_img.shape[1]*0.3)
# w_img = cv2.resize(w_img, (w,h),interpolation= cv2.INTER_AREA)
# w_img = Image.fromarray(w_img)

def detect_lines(img, debug=False):
    global VERTS
    ysize, xsize = img.shape[0], img.shape[1]

    # hls_img = filter_img_hsv(img)
    # gray = grayscale(hls_img)
    # cv2.imshow(' HLS ', hls_img)

    blur_gray = gaussian_blur(grayscale(img), kernel_size=5)

    # cv2.imshow(' gray ', blur_gray)

    ht = 150  # First detect gradients above. Then keep between low and high if connected to high
    lt = ht // 3  # Leave out gradients below
    canny_edges = canny(blur_gray, low_threshold=lt, high_threshold=ht)
    if debug: save_img(canny_edges, 'canny_edges_{0}'.format(index))

    # cv2.imshow('canny', canny_edges)
    # Our region of interest will be dynamically decided on a per-image basis
    regioned_edges, region_lines = region_of_interest(canny_edges)

    # cv2.circle(frame,(p1,p2), 5, (0,255,0),-1)
    # print ()
    # print (line_info[0][0]*p1+ p2 + line_info[0][1])

    rho = 2
    theta = 3 * np.pi / 180
    min_line_length = xsize // 16
    max_line_gap = min_line_length // 2
    threshold = min_line_length // 4
    lines, VERTS = hough_lines(regioned_edges, rho, theta, threshold, min_line_length, max_line_gap)
    VER = VERTS
    # print (VERTS)

    # Let's combine the hough-lines with the canny_edges to see how we did

    overlayed_lines = weighted_img(img, lines)
    # overlayed_lines = weighted_img(weighted_img(img, region_lines, a=1), lines)
    if debug: save_img(overlayed_lines, 'overlayed_lines_{0}'.format(index))

    return overlayed_lines, VER
def insertWarning(img):
    s_h, s_w, _ = img.shape
    # print (s_w, s_h)
    img = Image.fromarray(img)
    img.paste(w_img, (s_w - w - 10, 10))
    return np.array(img)


def issue_warning(x1, y1, VER):
    global VERTS
    point = Point(x1, y1)
    polygon = Polygon(VER)
    return polygon.contains(point)

#
# def viz(frame, out):
#     img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
#
#     total_vehicle = 0
#     warning = False
#     for detection in np.array(out).reshape(-1, 7):
#         idx = int(detection[1])
#         confidence = float(detection[2])
#
#         xmin = int(detection[3] * frame.shape[1])
#         ymin = int(detection[4] * frame.shape[0])
#         xmax = int(detection[5] * frame.shape[1])
#         ymax = int(detection[6] * frame.shape[0])
#
#
#
#         if (warning == False):
#             warning = issue_warning((xmin + xmax) / 2, ymax)
#         # warning = issue_warning(600,700)
#
#             # print (warning)
#
#     frame = weighted_img(frame, img, a=1.0, b=.5, l=0.)
#
#     if (warning):
#         frame = insertWarning(frame)
#
#     return frame, total_vehicle


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w+2, y + text_h+2), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size, img


def write_predictions_on_frames(df):
    fn_old="0.png"
    for idx, row in df.iterrows():
        # warning = False
        fn = "{}.png".format(int(row['frame']))
        fp = os.path.join(os.getcwd(), fn)
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        if os.path.exists(fp):
            im = cv2.imread(fp)
            speed_car = 10.58
            string = "{} meters".format(row['distance'])
            string_ttc = "{} Seconds".format(row['distance']/speed_car)
            fontface = cv2.FONT_HERSHEY_COMPLEX_SMALL
            _,img_t = draw_text(im, string, fontface,  (x1+3, y1+3));
            # _,img_t = draw_text(im, "TTC", fontface,  (x2+3, x2+3));
            _,img_t = draw_text(im, string_ttc, fontface,  (x2+3, y2+3), text_color_bg = (150,0,0));

            if fn == fn_old:
                warning = issue_warning((x1 + x2) / 2, y2, VER)
                # warning = issue_warning(600,700)

                # print (warning)

                # im_w = im
                if (warning):
                    im_w = insertWarning(img_t)
                else:
                    im_w = img_t

                cv2.imwrite(
                    "D:/Irascience/rahisho/object_detection-PLUS-distance_estimation-v1/object-detector/results_test/frames/" + fn,
                    im_w)
            else:
                im_l, VER = detect_lines(img_t)

                warning = issue_warning((x1 + x2) / 2, y2, VER)
                    # warning = issue_warning(600,700)

                    # print (warning)

                # im_w = im
                if (warning):
                    im_w = insertWarning(im_l)
                else:
                    im_w = im_l


                # print(fn)
                cv2.imwrite("D:/Irascience/rahisho/object_detection-PLUS-distance_estimation-v1/object-detector/results_test/frames/" + fn,im_w)
                cv2.waitKey(0)
        else:
          print(fp)
        fn_old = fn


def generate_video_from_frames():
    img_array = []
    imgs = glob.glob(os.path.join(os.getcwd(), '*.png'))
    size = (360, 640)
    for filename in imgs:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append([int(filename.split("\\")[-1].split('.')[0]), img])
    img_array.sort()
    out = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), int(fps), size)
    for i in range(len(img_array)):
        out.write(img_array[i][1])
    out.release()


argparser = argparse.ArgumentParser(
    description='visualize data from the dataframe')
argparser.add_argument('-d', '--data', help='input data csv file path')
argparser.add_argument(
    '-f', '--frames', help='input annotated video frames path')
argparser.add_argument('-fps', help="video frames per second")
argparser.add_argument('-r', '--results', help="output directory path")
args = argparser.parse_args()

# parse arguments
csvfile_path = args.data
frames_dir = args.frames
fps = args.fps
results_dir = args.results

# write predictions on frames
df = pd.read_csv(csvfile_path)
os.chdir("D:/Irascience/rahisho/object_detection-PLUS-distance_estimation-v1/object-detector/results_test/frames/")
write_predictions_on_frames(df)

# generate video from annotated frames
os.chdir("D:/Irascience/rahisho/object_detection-PLUS-distance_estimation-v1/object-detector/results_test/frames/")
generate_video_from_frames()
