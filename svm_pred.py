import bbox
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label

bbox = bbox.bbox()
bbox.get_param()

# Bounding box list to save box (x,y) in sliding windowed searching
bbox_list=[]

img = mpimg.imread('test_images/test6.jpg')

scale = 1.8
out_img, bbox_list = bbox.find_cars(img, scale, bbox_list)
plt.imshow(out_img)

scale = 1.6
out_img, bbox_list = bbox.find_cars(img, scale, bbox_list)
plt.imshow(out_img)

scale = 1.3
out_img, bbox_list = bbox.find_cars(img, scale, bbox_list)
plt.imshow(out_img)

scale = 1.0
out_img, bbox_list = bbox.find_cars(img, scale, bbox_list)
plt.imshow(out_img)

### Heatmap and labelledbounding box

# Heat map
heat = np.zeros_like(img[:,:,0]).astype(np.float)

# Add heat to each box in box list
heat = bbox.add_heat(heat,bbox_list)

# Apply threshold to help remove false positives
heat = bbox.apply_threshold(heat,2)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Label bounding box
# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = bbox.draw_labeled_bboxes(np.copy(img), labels)

# Show heat map
plt.imshow(heatmap)
plt.show()

# Show labelled image
plt.imshow(draw_img)
plt.show()
    

