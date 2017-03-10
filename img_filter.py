import numpy as np
import cv2

def undistort(image, K, d):
    """
    Camera distortion correction.
    K: camera intrinsic matrix
    d: distortion parameter
    """
    image = cv2.undistort(image, K, d, None, K)
    return image

def conv_hls_halfmask(image):
    """
    Convert input image to HLS colorspace and mask off upper half.
    Return converted image in three channels: H(0), L(1), S(2).
    """
    image = np.copy(image)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    # AOI (area of interest) mask - we only care about lower part of the image
    size_x, size_y, size_ch = hls.shape
    hls[0:size_x//2,:,:] = 0
    return hls

def filter_luma(image_luma, threshold = 30):
    """
    Return a image-sized binary file in which 1 represents 
    the pixel luminance greater than threshold.
    """
    assert image_luma.ndim==2
    luma_binary = np.zeros_like(image_luma)
    luma_binary[image_luma>threshold]=1
    return luma_binary

def filter_sat(img_sat_ch, threshold = (170,255)):
    """
    Return a image-sized binary file in which 1 represents 
    the saturation at the pixel location within threshold.
    Expect a S channel input from HLS colorspace converted image.
    """
    assert img_sat_ch.ndim==2
    sat_binary = np.zeros_like(img_sat_ch)
    scaled_s_ch = np.uint8(255*img_sat_ch/np.max(img_sat_ch))
    sat_binary[(scaled_s_ch >= threshold[0]) & (scaled_s_ch <= threshold[1])] = 1
    return sat_binary    

def filter_gradient_threshold(image, direction="x", threshold=(50,150),ksize=3):
    """
    Return a image-sized binary file in which 1 represents
    the gradient at specific pixel location is greater 
    than threshold. Taking "x" or "y" direction as input.
    """
    assert image.ndim==2
    # Sobel x on L channel
    if direction=="x":
        # Take the derivative in x dir
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize)
    else:
        # Take the derivative in y dir
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize)
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobel = np.absolute(sobel) 
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Threshold x gradient
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
    return sobel_binary

def filter_mentor_advise(image):
    """
    Implement what Udacity mentor feedback.
    """
    HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # For yellow
    yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))

    # For white
    sensitivity_1 = 68
    white = cv2.inRange(HSV, (0,0,255-sensitivity_1), (255,20,255))

    sensitivity_2 = 60
    HSL = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    white_2 = cv2.inRange(HSL, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
    white_3 = cv2.inRange(image, (200,200,200), (255,255,255))

    bit_layer = yellow | white | white_2 | white_3

    return bit_layer

def filter_fusion(luma_bin, sat_bin, grad_bin, mentor_bin):
    """
    Fuse binary filters result.
    """
    binary = np.zeros_like(luma_bin)
    binary[ (((grad_bin==1) | (sat_bin==1)) & (luma_bin==1)) | (mentor_bin==1) ] = 1

    # Erosion and dilation - Seems doesn't work. Mask-off
    #kernel = np.ones((5,5))
    #binary_dilation = cv2.dilate(binary, kernel, iterations=1)
    #binary_erosion = cv2.erode(binary_dilation, kernel, iterations=1)
    #binary = binary_erosion

    return binary

