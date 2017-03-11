import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import sys

# -------------------------------------
# Command line argument processing
# -------------------------------------
if len(sys.argv) < 2:
    print("Missing image file.")
    print("python3 lane_detector.py <image_file>")

FILE = str(sys.argv[1])

clip = cv2.VideoCapture(FILE)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

frame_cnt = 0
frame_start = 680
frame_end = 750
#frame_end = 50

out=None

while True:
    flag, image = clip.read()
    if flag:
        frame_cnt += 1
        if frame_cnt < frame_start:
            continue
        elif frame_cnt > frame_end:
            break
        print('frame_cnt = ', frame_cnt)
        if out == None:
            out = cv2.VideoWriter('video_cut.avi', fourcc, 30.0, (image.shape[1], image.shape[0]))

        #cv2.imshow('video', image)
        cv2.imwrite('c_'+str(frame_cnt)+'.jpg', image)
        #out.write(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break




