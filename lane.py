import numpy as np
import cv2
import pickle

class lane():
    """
    Class to contain all lane features. 
    """

    def __init__(self):
        # Detection flag
        self.detected = False
        # x values of the last n fits of the line
        self.recent_l_fit = [] 
        self.recent_r_fit = [] 
        #polynomial coefficients for the most recent fit
        self.cur_l_fit = []
        self.cur_r_fit = []
        # Current base point in x direction, where search starts
        self.leftx_base = 320
        self.rightx_base = 960 
        # Number of history records
        self.num_history = 10
        # Number of consecutive no good detection, which should trigger reset
        self.num_no_update = 5
        # Max allowed percetage of deviation from avg in a single detection
        self.max_deviation_percentage = 0.5
        # No update counters
        self.l_cnt = 0
        self.r_cnt = 0
        # Search window margin
        self.margin = 100
        # Previous lane area points
        self.lane_pts = None

    def get_param(self):
        # Camera parameters
        f = open('camera_cal/wide_dist_pickle.p', 'rb')
        param = pickle.load(f)
        K = param["mtx"]        # Camera intrinsic matrix
        d = param["dist"]       # Distortion parameters
        f.close()
        
        # Perspective transform parameter
        warp_f = open('camera_cal/warp.p', 'rb')
        warp_param = pickle.load(warp_f)
        P = warp_param["warp"]  # Road to birdview transform
        warp_f.close()
        
        P_inv = np.linalg.inv(P)# Birdview to road transform

        return K,d,P,P_inv

    def curve_fit_1st(self, binary_warped, visualize=True):
        '''
        Curve fit for 1st frame, using searching windows.
        '''
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        if visualize:
            # Create an output image to draw on and  visualize the result
            out_img = np.array(np.dstack((binary_warped, binary_warped, binary_warped))*255, dtype='uint8')
        else:
            out_img = []
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        # Search within a certain window
        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        print("Rebase: self.leftx_base = ", self.leftx_base, "self.rightx_base = ", self.rightx_base)
        
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base
        # Set the width of the windows +/- margin
        # Set minimum number of pixels found to recenter window
        minpix = 100
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            if visualize:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        self.cur_l_fit = left_fit 
        self.cur_r_fit = right_fit
        
        # Set flag
        self.detected = True

        ## Visualization
        if visualize:
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return out_img  

    def curve_fit(self, binary_warped, visualize=True):
        '''
        Curve fit since 2nd frame
        '''
        left_fit = self.cur_l_fit
        right_fit = self.cur_r_fit

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - self.margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + self.margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - self.margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + self.margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        self.cur_l_fit = left_fit 
        self.cur_r_fit = right_fit

        # Set flag
        self.detected = True
        
    def visualize_fit(self, binary_warped):
        """
        Visualize the birds-eye viewpoint fit condition.
        """
        left_fit = self.cur_l_fit
        right_fit = self.cur_r_fit

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - self.margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + self.margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - self.margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + self.margin)))  
 
        #Visualize
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.array(np.dstack((binary_warped, binary_warped, binary_warped))*255, dtype='uint8')
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return out_img
   
    def fit_sanity_check(self):
        """
        - Current detected within n% of best average of past.
        - TODO: Distance between left/right lane lines.
        If yes to all above, update best records. 
        If no to any of above, keep using best records. 
        """
        l_fit = self.cur_l_fit
        r_fit = self.cur_r_fit

        # Start of detection, fill history
        if len(self.recent_l_fit)<self.num_history:
            self.recent_l_fit.append(self.cur_l_fit)
        else:
            l_avg = np.mean(np.array(self.recent_l_fit), axis=0)
            l_delta = np.abs((l_fit - l_avg)/l_avg) < self.max_deviation_percentage
            if l_delta.all():
                print("l updated.")
                self.recent_l_fit.pop(0)
                self.recent_l_fit.append(self.cur_l_fit)
                self.l_cnt = 0
            else:
                print("l uses avg.")
                l_fit = l_avg
                self.l_cnt += 1
                if self.l_cnt > self.num_no_update:
                    print("l reset.")
                    self.detected = False 
                    self.recent_l_fit = []
                    self.l_cnt = 0

        if len(self.recent_r_fit)<self.num_history:
            self.recent_r_fit.append(self.cur_r_fit)
        else:
            r_avg = np.mean(np.array(self.recent_r_fit), axis=0)
            r_delta = np.abs((r_fit - r_avg)/r_avg) < self.max_deviation_percentage
            if r_delta.all():
                print("r updated.")
                self.recent_r_fit.pop(0)
                self.recent_r_fit.append(self.cur_r_fit)
                self.r_cnt = 0
            else:
                print("r uses avg.")
                r_fit = r_avg
                self.r_cnt += 1
                if self.r_cnt > self.num_no_update:
                    print("r reset.")
                    self.detected = False 
                    self.recent_r_fit = []
                    self.r_cnt = 0

        # Update to use corrected l/r_fit
        self.cur_l_fit = l_fit
        self.cur_r_fit = r_fit
        return self.detected
 
    def match_points(self, pts):
        """
        Check whether current points largely match previous one.
        If not (above threshold), use previous saved points.
        """
        if (self.lane_pts == None):
            self.lane_pts = pts

        a = self.lane_pts[0]
        b = pts[0]
        ret = cv2.matchShapes(a,b,1,0.0)

        if (ret < 0.1):
        # Use the new polygon points to write the next frame due to similarites of last sucessfully written polygon area
            self.lane_pts = pts
        else:
        # Use the old polygon points to write the next frame due to irregularities
        # Then write the out the old polygon points
        # This will help only use your good detections
            pts = self.lane_pts
        return pts

    def draw_lane_area(self, binary_warped, image, P_inv):
        """
        Draw the detected lane area on the road surface.
        - binary_warped is the birds-eye view binary image.
        - image is the original image for alpha-blending.
        """
        # Convert back to map to road
        l_fit = self.cur_l_fit
        r_fit = self.cur_r_fit
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
        right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]
        # Create an image to draw the lines on
        color_warp = np.array(np.dstack((binary_warped, binary_warped, binary_warped))*0, dtype='uint8')
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # 2nd sanity check on shape
        pts = self.match_points(pts)
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        lane_shadow_on_road_img = cv2.warpPerspective(color_warp, P_inv, (image.shape[1], image.shape[0]))
        # Calculate curvature and car position
        curverad, off_center = self.cal_curvature(binary_warped)
        # Write onto the image
        cv2.putText(lane_shadow_on_road_img, 'Curverad = '+'%.1f'%curverad+'m', (50,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)
        cv2.putText(lane_shadow_on_road_img, 'Off center = '+'%.2f'%off_center+'m (negative means left)', (50,100), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)

        # Alpha blending with undistorted image.
        res = cv2.addWeighted(image, 1, lane_shadow_on_road_img, 0.3, 0)

        return res

    def cal_curvature(self, binary_warped):
        # Measure curvature

        left_fit = self.cur_l_fit
        right_fit = self.cur_r_fit

        ym_per_pix = 30/720 # Assuming 30 meters per pixel in y dimenstion
        xm_per_pix = 3.7/700 # 3.7 meters per pixel in x dimenstion
    
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        leftx= left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

        y_eval = np.max(ploty) # Select the bottom line position to evaluate
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        # Now our radius of curvature is in meters
        #print(left_curverad, 'm')

        # Find car position as to center of lane
        leftx= left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        rightx= right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

        off_x = 640 - (leftx + rightx)//2 
        off_center = off_x * xm_per_pix
        #print("off center = ", off_center)

        # Update left/right x base
        # Base x pixel position sanity check
        if leftx<160 or leftx> 480:
            self.leftx_base = 320
        else:
            self.leftx_base = leftx
        if rightx<800 or rightx> 1120:
            self.rightx_base = 960 
        else:
            self.rightx_base = rightx
        print("self.leftx_base = ", self.leftx_base, "self.rightx_base = ", self.rightx_base)

        return left_curverad, off_center
    


