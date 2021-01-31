import numpy as np
import time
import cv2
from imutils.object_detection import non_max_suppression

img = "D:/DataScience/AppliedDS&AI/Session8/images/2018-Jaguar-XF-6-header.jpg"
east = "D:/DataScience/AppliedDS&AI/Session8/frozen_east_text_detection.pb"
height = 320  # in model doc, model works in any pic with width and height multiple of 32
width = 320
min_confidence = 0.5 # probability threshold 

# laoding image and graping image dimension
image = cv2.imread(img)
imageH, imageW = image.shape[:2]

cv2.imshow("Input Image", image)
cv2.waitKey(0)

# set the new width and height and then determine the ratio i change for both
# will be used to revet size to the original size
rH = imageH / float(height)
rW = imageW / float(width)

# resize the image and grap the new dimension
image_resized = cv2.resize(image, (width, height))

cv2.imshow("Resized Image", image_resized)
cv2.waitKey(1)

# load pre-trained EAST text detector NN Model
print("[INFO] loading EAST text detector Model")
net = cv2.dnn.readNet(east)
net.getLayerNames()

# Getting the two output layers that we are interested in
# The first layer is the ouput sigmoid activation which gives us the peobability
# of the text in a region
# The second layer is the output feature map that represents the "geometry" of
# the image that we will use drive the bounding box of the coordinates of the text

layerNames = [net.getLayerNames()[-1],  # 'feature_fusion/Conv_7/Sigmoid'
              net.getLayerNames()[-3]]  # 'feature_fusion/concat_3'

# constructing blob from the image (performing some operations to image)
# blob -> changing the image to suite the model

# [blobFromImage] creates 4-dim blob from image
# optionally resizes, crops image from center, subtract mean values, scales 
# value by scale factor, swap Blue and Red channles
# blobFromImage: performs **Mean subtraction, **Scaling, And optionally **channel swapping

# blob output dimensions "NCHW = (batch_size, channels, hieght, width)"
# Mean subtraction is used to help comabat illumination changes in the input image to aid CNN
# mean values for the ImageNet training set are R=103.93, G=116.77, and B=123.68
# Scaling factor, sigma, adds in a normalization = 1.0
# Channel swapping: OpenCV assumes images are in BGR channel order; however, the `mean` value assumes
# we are using RGB order. To resolve this discrepancy we can swap the R and B channels in image.

blob = cv2.dnn.blobFromImage(image_resized,
                         scalefactor = 1.0,
                         size = (width, height),
                         mean=(123.68, 116.78, 103.94), #mean pixel intensity across all training images
                         swapRB = True,
                         crop = False)

# pass blob through EAST network
# By suppling layerNames as a parameter to net.forward, we're instructing cv2 to return:
# 1. The output scores map provides the probability of a given region containing text
# 2. The output geometry map used to derive the bounding box coordinates of text in our input image
start = time.time()
net.setInput(blob)
scores, geometry = net.forward(layerNames)
end = time.time()

# show timing info on text detection
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab number of rows and columns from scores
# then, initialize our set of bounding box rectangles and the corresponding 
# confidence scores
numRows, numCols = scores.shape[2:4]
rects = []
confidences = []

# lop over number of rows
for y in range(0, numRows):
    # extract scores (probablities), followed by geometrical data used to drive 
    # potential bounding box rectangles and corresponding confidence score
    scoresData = scores[0, 0, y, :]
    xData0 = geometry[0, 0, y, :]
    xData1 = geometry[0, 1, y, :]
    xData2 = geometry[0, 2, y, :]
    xData3 = geometry[0, 3, y, :]
    anglesData = geometry[0, 4, y, :]
    
    # loop over number of columns
    for x in range(0, numCols):
        # if score less than confidence threshold ignore it
        if scoresData[x] < min_confidence:
            continue
        
        # compute the offset factor as our resulting feature maps will be 4x 
        # smaller than the input image "Input image (320, 320) & feature map (80,80)"
        # The EAST reduces volume size as the input image passes through the network
        # our volume size is 4x smaller than our input image
        # so we multiply by four to bring the coordinates back into respect of our 
        # orignal image
        offsetX, offsetY = (x * 4.0, y * 4.0)
        
        # use the geometry volume to drive the width and height of the bounding box
        # h_upper, w_right, h_lower, w_left, A = geometry[0,:,y,x]
        h = xData0[x] + xData2[x]       # upper/lower
        w = xData1[y] + xData3[y]       # left/right
        
        # extract predicted rotation angle and calculate sin and cos
        angle = anglesData[x]
        sin = np.sin(angle)
        cos = np.cos(angle)
        
        # compute the starting and ending (x, y) coordinates for the text
        # prediction bounding box
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)
        
        # add the bounding box coordinates and probability score to the respective lists
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])
        
        
# apply non-max suppression to suppress weak, overlapping bounding box
boxes = non_max_suppression(np.array(rects), confidences)

# loop over bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale bounding box coords based on the respective ratios
    # draw the bounding box on the original image before resizing
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    
    # draw the bounding box on the image
    # image, starting points, ending points, color, box thickness
    cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
    
cv2.imshow("Text Detection Image", image)
cv2.waitKey(0)
    
    
