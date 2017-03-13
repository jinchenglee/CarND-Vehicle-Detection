import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
import pickle
import glob
import sys
import img_filter 
import lane
import bbox



# Vehicle detection pipeline
def bbox_pipeline(bbox, img, bbox_list=[]):
    '''
    Processing vehicle detection and bounding box.
    '''
    img = np.copy(img)

    # Do multi-scale searching
    scale = 1.0
    bbox_list = bbox.find_cars(img, scale, bbox_list)
    scale = 1.5
    bbox_list = bbox.find_cars(img, scale, bbox_list)
    scale = 2.0
    bbox_list = bbox.find_cars(img, scale, bbox_list)
    
    ### Heatmap and labelledbounding box
    # Heat map
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = bbox.add_heat(heat,bbox_list)
    # Apply threshold to help remove false positives
    heat = bbox.apply_threshold(heat,5)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Label bounding box
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    img = bbox.draw_bboxes(img, bbox_list)
    draw_img = bbox.draw_labeled_bboxes(img, labels)
    # To view the heatmap boxes?
    #draw_img = np.array(np.dstack((heatmap, heatmap, heatmap))*255, dtype='uint8')
    # Alpha blending
    draw_img = cv2.addWeighted(draw_img, 0.9, img, 0.1, 0) 

    # Searching window (big and small)
    s_win = ((bbox.xstart_s,bbox.ystart_s), (bbox.xstop_s,bbox.ystop_s))
    b_win = ((bbox.xstart,bbox.ystart), (bbox.xstop,bbox.ystop))
    cv2.rectangle(draw_img, s_win[0], s_win[1], (0,0,255), 2)
    cv2.rectangle(draw_img, b_win[0], b_win[1], (0,0,255), 2)

    return draw_img




# -------------------------------------
# Command line argument processing
# -------------------------------------
if len(sys.argv) < 2:
    print("Missing image file.")
    print("python3 video.py <image_file>")

FILE = str(sys.argv[1])

VISUAL_ON = False
if len(sys.argv)>2:
    VISUAL_ON = True

clip = cv2.VideoCapture(FILE)
fourcc = cv2.VideoWriter_fourcc(*'X264')

frame_cnt = 0
frame_start = 0
frame_end = 0xffffffff
#frame_end = 50

out=None

# Vehicle detector
bbox = bbox.bbox()
bbox.get_param()
bbox_list = []

# Search as if from start of frame
detected = False

while True:
    flag, image = clip.read()
    bbox_list = []
    if flag:
        frame_cnt += 1
        if frame_cnt < frame_start:
            continue
        elif frame_cnt > frame_end:
            break
        print('frame_cnt = ', frame_cnt)
        if out == None:
            out = cv2.VideoWriter('output.avi', fourcc, 30.0, (image.shape[1]//2, image.shape[0]//2))

        # Vehicle detection pipeline
        res = bbox_pipeline(bbox, image, bbox_list)

        # Resize
        res = cv2.resize(res, (res.shape[1]//2, res.shape[0]//2))
        # Write video out
        cv2.imshow('video', res)
        out.write(res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


