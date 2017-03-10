import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import sys
import img_filter 
import lane


# Lane detection pipeline
def pipeline(lane, img, fresh_start=False, luma_th=30, sat_th=(170, 255), grad_th=(50, 150), sobel_ksize=5, visual_on=True):
    '''
    Processing pipeline
    
    1. Leverage HLS color space. 
    2. Gradients threshold.
    3. Area of interest mask.
    '''
    img = np.copy(img)

    # Get various parameters
    K,d,P,P_inv = lane.get_param()

    # Undistort
    img = img_filter.undistort(img, K, d)

    # Convert to HSV color space and separate the V channel
    hls = img_filter.conv_hls_halfmask(img)
    # Luma threshold
    luma_binary = np.zeros_like(hls[:,:,1])
    luma_binary = img_filter.filter_luma(hls[:,:,1], threshold=luma_th)

    # Sobel x on L channel
    grad_binary = np.zeros_like(luma_binary)
    grad_binary = img_filter.filter_gradient_threshold(image=hls[:,:,1],threshold=grad_th, ksize=sobel_ksize)

    # Threshold Saturation color channel
    sat_binary = np.zeros_like(luma_binary)
    sat_binary = img_filter.filter_sat(img_sat_ch=hls[:,:,2], threshold=sat_th)

    # Mentor feedback method
    mentor_binary = img_filter.filter_mentor_advise(img) 

    # Combine filter binaries
    binary = np.zeros_like(luma_binary)
    binary = img_filter.filter_fusion(luma_binary, sat_binary, grad_binary, mentor_binary)

    # Perspective transform
    img_size = (binary.shape[1], binary.shape[0])
    binary_warped = cv2.warpPerspective(binary, P, img_size, flags=cv2.INTER_NEAREST)

    # Curve fit for the 1st frame
    if fresh_start:
        _ = lane.curve_fit_1st(binary_warped)
    else:
        # Simulate the case to feed a "second" frame using curve_fit()
        lane.curve_fit(binary_warped)

    # Sanity check on curve fit parameters
    detected = False
    detected = lane.fit_sanity_check()

    # Draw detected lane onto the road
    res = lane.draw_lane_area(binary_warped, img, P_inv)

    # Return the binary image
    visualize_img = lane.visualize_fit(binary_warped)

    # Optional: blending with visualization image
    if visual_on:
        res = cv2.addWeighted(res, 1, visualize_img, 0.5, 0)

    return res, visualize_img, detected


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

# Generate x and y values for plotting
ploty = np.linspace(0, 719, 720)

lane = lane.lane()

# Search as if from start of frame
detected = False

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
            out = cv2.VideoWriter('output.avi', fourcc, 30.0, (image.shape[1], image.shape[0]//2))

        # Video pipeline
        res, vis_img, detected = pipeline(lane, image, (frame_cnt==1) or (not(detected)), visual_on=VISUAL_ON)

        # Resize
        res = cv2.resize(res, (res.shape[1]//2, res.shape[0]//2))
        vis_img = cv2.resize(vis_img, (vis_img.shape[1]//2, vis_img.shape[0]//2))
        new_res = np.concatenate([res,vis_img],axis=1)
        # Write video out
        cv2.imshow('video', new_res)
        out.write(new_res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


