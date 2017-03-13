##Vehicle Detection Project##
![alt text][image0]
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run implementation pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image0]: ./output_images/lane_car_detection.gif
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in files bbox.py, as a class function call get_hog_feature().

```python
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

```

All the `vehicle` and `non-vehicle` images from the labelled dataset are read in train_svm() function.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

```python
    def train_svm(self):
        """
        Training linear SVM classifier
        """
        # Read in car and non-car images
        image = []
        cars = []
        notcars = []

        for d in ['GTI_Far','GTI_Left','GTI_MiddleClose','GTI_Right','KITTI_extracted']:
            images = glob.glob('vehicles/'+d+'/*.png')
            for f in images:
                cars.append(f)

        for d in ['Extras','GTI']:
            images = glob.glob('non-vehicles/'+d+'/*.png')
            for f in images:
                notcars.append(f)
        ...

```

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and it turns out colorspace 'YCrCb' with only luminance channel 'Y' fed into hog detection gives best result in the meantime smallest feature width, which saves training and running time. 

The 'CrCb' channels provide little extra differentiation in HOG detection. Instead, I used color information to extract color histogram and binning information to be attached to HOG features extracted from Y/luminance channel.

Below are functions defined in bbox.py class.

```python
    # Class functions
    ...
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

            # Apply color_hist() also with a color space option now
            hist_features = self.color_hist(feature_image, nbins=self.histbin, bins_range=self.hist_range)

            # Compute only Y/luminance channel HOG features for the entire image
            hog1 = self.get_hog_features(feature_image[:,:,0], self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

            #hog_features = np.hstack((hog1, hog2, hog3)).reshape(-1,)
            hog_features = hog1.reshape(-1,)
            #print(hog_features.shape)

            # Append the new feature vector to the features list
            features.append(np.concatenate((spatial_features, hist_features, hog_features)))
            
        # Return list of feature vectors
        return features
    ...

```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using skimage.svm library. 

Before feeding concatenated features into training, I've normalize the values per feature column by using sklearn.preprocessing.StandardScaler thus not a single feature can override others due to its absolute values. 

A following up step is to randomize the samples by using sklearn.model_selection.train_test_split() to avoid overfitting in trained model. 

```python
	...
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

        # Use a linear SVC 
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

