import pickle
import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog
from scipy.ndimage.measurements import label
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

import img_filter 


class bbox():
    """
    Class to contain all vehicle detection classifier features. 
    """

    def __init__(self):
        # Parameters of image spatial and color histogram features 
        self.spatial_size = 16
        self.histbin = 10
        self.hist_range = (50,256)
        # Parameters of HOG features 
        self.colorspace = 'RGB2YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        # Classifier
        self.svc = None
        self.X_scaler = None
        # Only search the lower part of image
        self.xstart = 0
        self.xstop = 1280
        self.ystart = 400
        self.ystop = 656
        # Smaller window
        self.xstart_s = 150
        self.xstop_s = 1130
        self.ystart_s = 400
        self.ystop_s = 520
        # History heatmap
        self.heatmap_his = []
        self.len_heatmp_history = 3
        # Small scale threshold (to shrink search area with that scale)
        self.small_scale_threshold = 1.6
        # Prediction confidence threshold (To reject false positive at very beginning)
        self.pred_confidence_threshold = 0.9


    def save_param(self, dist_pickle={}):
        """
        # SVM classifier parameters
        # Save to pickle file
        """
        pickle_file = open("svc_pickle.p", 'wb')
        pickle.dump(dist_pickle, pickle_file)

    def get_param(self, pickle_file='svc_pickle.p'):
        """
        # Retrieve saved classifier parameters
        """
        dist_pickle = pickle.load(open(pickle_file, "rb" ) )
        self.svc = dist_pickle["svc"]
        self.X_scaler = dist_pickle["scaler"]

    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, 
                            vis=False, feature_vec=True):
        """
        # HoG feature extraction.
        # Call with two outputs if vis==True
        """
        if vis == True:
            features, hog_image = hog(img, orientations=orient, 
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), 
                                      transform_sqrt=False, 
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:      
            features = hog(img, orientations=orient, 
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), 
                           transform_sqrt=False, 
                           visualise=vis, feature_vector=feature_vec)
            return features
    
    def bin_spatial(self, img, size=(32, 32)):
        """
        # Define a function to compute binned color features  
        """
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features
    
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        """
        # Define a function to compute color histogram features  
        """
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features
    
    def extract_features(self, imgs):
        """
        # Define a function to extract features from a list of images
        # Have this function call bin_spatial(), color_hist() and get_hog_features().
        """
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = cv2.imread(file)
            # apply color conversion 
            feature_image = img_filter.convert_color(image, conv=self.colorspace)
            # Apply bin_spatial() to get spatial color features
            spatial_features = self.bin_spatial(feature_image, size=(self.spatial_size,self.spatial_size))
            #print(spatial_features.shape)
            # Apply color_hist() also with a color space option now
            hist_features = self.color_hist(feature_image, nbins=self.histbin, bins_range=self.hist_range)
            #print(hist_features.shape)
            # Compute individual channel HOG features for the entire image
            hog1 = self.get_hog_features(feature_image[:,:,0], self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            #hog2 = self.get_hog_features(feature_image[:,:,1], self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
            #hog3 = self.get_hog_features(feature_image[:,:,2], self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
    
            #hog_features = np.hstack((hog1, hog2, hog3)).reshape(-1,)
            hog_features = hog1.reshape(-1,)
            #print(hog_features.shape)

            # Append the new feature vector to the features list
            features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        # Return list of feature vectors
        return features
    

    def train_svm(self):
        """
        Training linear SVM classifier
        """
        # Read in car and non-car images
        image = []
        cars = []
        notcars = []
        
        #for d in ['cars1', 'cars2', 'cars3']:
        #    images = glob.glob('vehicles_smallset/'+d+'/*.jpeg')
        # If using below data, please remember to *255 to scale to right range
        for d in ['GTI_Far','GTI_Left','GTI_MiddleClose','GTI_Right','KITTI_extracted']:
            images = glob.glob('vehicles/'+d+'/*.png')
            for f in images:
                cars.append(f)
        
        #for d in ['notcars1', 'notcars2', 'notcars3']:
        #    images = glob.glob('non-vehicles_smallset/'+d+'/*.jpeg')
        # If using below data, please remember to *255 to scale to right range
        for d in ['Extras','GTI']:
            images = glob.glob('non-vehicles/'+d+'/*.png')
            for f in images:
                notcars.append(f)
        
        # Extract features from labeled dataset
        car_features = self.extract_features(cars)        
        notcar_features = self.extract_features(notcars)        

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
        
        print('Using spatial binning of:',self.spatial_size,'and', self.histbin,'histogram bins')
        print('Number of training data:',len(y_train), 'testing data:', len(y_test))
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
        
        # Save trained parameters to pickle file
        dist_pickle = {}
        dist_pickle["svc"] = svc
        dist_pickle["scaler"] = X_scaler
        self.save_param(dist_pickle)


    def find_cars(self, img, scale, bbox_list=[]):
        """
        # A single function that can extract features using hog sub-sampling and make predictions
        # using pre-trained SVM classifier.
        """
        
        if scale<self.small_scale_threshold:
            img_tosearch = img[self.ystart_s:self.ystop_s,self.xstart_s:self.xstop_s,:]
        else:
            img_tosearch = img[self.ystart:self.ystop,:,:]

        ctrans_tosearch = img_filter.convert_color(img_tosearch, conv=self.colorspace)
        #print(np.max(ctrans_tosearch))
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    
        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell)-1
        nyblocks = (ch1.shape[0] // self.pix_per_cell)-1 
        nfeat_per_block = self.orient*self.cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell)-1 
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        print(nxsteps)
        print(nysteps)
        
        # Compute individual channel HOG features for the entire image
        hog1 = self.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        #hog2 = self.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        #hog3 = self.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                #hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                #hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                #hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                hog_features = hog_feat1
    
                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell
    
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features
                spatial_features = self.bin_spatial(subimg, size=(self.spatial_size, self.spatial_size))
                hist_features = self.color_hist(subimg, nbins=self.histbin, bins_range=self.hist_range)
                #print(hog_features.shape, spatial_features.shape, hist_features.shape)
    
                # Scale features and make a prediction
                tmp = np.hstack((spatial_features, hist_features, hog_features)).astype(np.float64) 
                #print(tmp.shape)
                test_features = self.X_scaler.transform(tmp.reshape(1,-1))     
                test_prediction = self.svc.predict(test_features)
                test_confidence = self.svc.decision_function(test_features)
         
                # Safeguard using prediction confidence to eliminate false positive 
                # At very beginning
                if (test_prediction==1) and (test_confidence>self.pred_confidence_threshold):
                    #print(test_confidence)
                    if scale<self.small_scale_threshold:
                        xbox_left = self.xstart_s+np.int(xleft*scale)
                        ytop_draw = self.ystart_s+np.int(ytop*scale)
                    else:
                        xbox_left = self.xstart+np.int(xleft*scale)
                        ytop_draw = self.ystart+np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    top_left = (xbox_left, ytop_draw)
                    bottom_right = (xbox_left+win_draw,ytop_draw+win_draw)
                    bbox_list.append((top_left, bottom_right))
                    
        return bbox_list
    

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
        
    def apply_threshold(self, heatmap, threshold):
        # History list full
        len_heatmap_his = len(self.heatmap_his)
        #print(np.sum(heatmap))
        if len_heatmap_his>self.len_heatmp_history:
            # Remove oldest hist data
            self.heatmap_his.pop(0)
        # Add latest data into the list
        self.heatmap_his.append(heatmap)
        #print(np.sum(self.heatmap_his))

        # Accumulate over history
        acc_heatmap = np.zeros_like(heatmap)
        for cur_heatmap in self.heatmap_his:
            acc_heatmap += cur_heatmap
            print(np.max(cur_heatmap), np.max(acc_heatmap))

        # Zero out pixels below the threshold
        acc_heatmap[acc_heatmap <= threshold] = 0
        # Return thresholded map
        return acc_heatmap
    
    def draw_bboxes(self, img, bbox_list):
        # Iterate through all detected cars
        for bbox in bbox_list:
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 1)
        # Return the image
        return img

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # If box is too small, it must be a false positive
            THRESHOLD = 48 # pixels
            if ( (bbox[1][0]-bbox[0][0])<THRESHOLD or (bbox[1][1]-bbox[0][1])<THRESHOLD ):
                continue
            else:
                # Draw the box on the image
                cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img

