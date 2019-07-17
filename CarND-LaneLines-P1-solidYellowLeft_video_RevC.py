#!/usr/bin/env python
# coding: utf-8

# In[21]:


#importing packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
get_ipython().run_line_magic('matplotlib', 'inline')

import os
os.listdir("test_images/")

from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Read in and grayscale the image 
#image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
def grayscale_process(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return gray

# Define a kernel size and apply Gaussian smoothing
def gaussian_blur_process(gray):
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    return blur_gray

# Define our parameters for Canny and apply
def canny_process(blur_gray):
    low_threshold = 50
    high_threshold = 200
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    return edges

# Next we'll create a masked edges image using cv2.fillPoly()
def masked_edges_image(image, edges):
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   
    
    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(150,imshape[0]),(400, 322), (550, 322), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
def hough_lines_process(image):
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 40  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1#minimum number of pixels making up a line
    max_line_gap = 150# maximum gap in pixels between connectable line segments
    return rho, theta, threshold, min_line_length, max_line_gap
    
def separate_left_and_right_lines(lines, slope_boundaries):
    if lines is None:
        return

    left_points_x = []
    left_points_y = []
    right_points_x = []
    right_points_y = []
    # Separate left points from right ones
    for line in lines:        
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            # Left lines
            if slope < -slope_boundaries[0] and slope >= -slope_boundaries[1]:
                left_points_x.append(x1)
                left_points_x.append(x2)
                left_points_y.append(y1)
                left_points_y.append(y2)
            # Right lines
            if slope > slope_boundaries[0] and slope <= slope_boundaries[1]:
                right_points_x.append(x1)
                right_points_x.append(x2)
                right_points_y.append(y1)
                right_points_y.append(y2)
return [left_points_x, left_points_y, right_points_x, right_points_y]

def draw_lines(image):
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = hough_lines_process(image)
    slope_boundaries = [0.5, 1.0]
    seperate_points = separate_left_and_right_lines(lines, slope_boundaries) 
    lines_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) # Black image
    draw_line_polyfit(lines_img, separate_points[0], separate_points[1], []) # Left line on black image
    draw_line_polyfit(lines_img, separate_points[2], separate_points[3], []) # Right line on black image
    lines_edges = cv2.addWeighted(image, 1, lines_imgs, 1, 0)
    return lines_edges
    
def process_image(image):
        # Grayscale
        gray = grayscale_process(image)
        # Gaussian Smoothing
        blur_gray = gaussian_blur_process(gray)
        # Canny Edge
        edges = canny_process(blur_gray)
        # Region of Interest
        masked_edges = masked_edges_image(image, edges)
        # Hough Transform
        hough = hough_lines_process(image)
        # Draw lines 
        draw_lines(masked_edges) 
        return image    

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




