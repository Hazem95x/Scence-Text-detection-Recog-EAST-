import numpy as np
import time
import pytesseract
import cv2


from imutils.object_detection import non_max_suppression

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
img = "G:\Master\Image analysis and pattern recognition\Project\textDetection-Recognition-master\Text-Detection-and-Recognition\images\download.jpg"
#img = "D:/DataScience/AppliedDS&AI/Session8/images/download.jpg"
east="G:\Master\Image analysis and pattern recognition\Project\textDetection-Recognition-master\Text-Detection-and-Recognition\frozen_east_text_detection.pb"
#east = "D:/DataScience/AppliedDS&AI/Session8/frozen_east_text_detection.pb"
height = 320  # in model doc, model works in any pic with width and height multiple of 32
width = 320
padding = 0.2
min_confidence = 0.5 # probability threshold 

# laoding image and graping image dimension
image = cv2.imread(img)
imageH, imageW = image.shape[:2]


cv2.waitKey(0)

# set the new width and height and then determine the ratio i change for both
# will be used to revet size to the original size
rH = imageH / float(height)
rW = imageW / float(width)

# resize the image and grap the new dimension
image_resized = cv2.resize(image, (width, height))

#cv2.imshow("Resized Image", image_resized)
#cv2.waitKey(0)

# load pre-trained EAST text detector NN Model
print("[INFO] loading EAST text detector Model")
net = cv2.dnn.readNet(east)
net.getLayerNames()

# Getting the two output layers that we are interested in
# The first layer is the ouput sigmoid activation which gives us the peobability
# of the text in a region
# The second layer is the output feature map that represents the "geometry" of
# the image that we will use drive the bounding box of the coordinates of the text
# By suppling layerNames as a parameter to net.forward, we're instructing cv2 to return:
# 1. The output scores map provides the probability of a given region containing text
# 2. The output geometry map used to derive the bounding box coordinates of text in our input image

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
                             mean=(123.68, 116.78, 103.94), # mean pixel intensity across all training images
                             swapRB = True,
                             crop = False)

# pass blob through EAST network

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
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]
    
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
        w = xData1[x] + xData3[x]       # left/right
        
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
boxes = non_max_suppression(np.array(rects), probs=confidences)


# intialize list of results
results = []

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    
    # scale bounding box coords based on the respective ratios
    # draw the bounding box on the original image before resizing
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    
    
    ## ENHANCEMENT
    # in order to obtain a better OCR of the text we can potentially apply a bit of padding
    # surrounding the boundingbox, -- here we compute the delta in both directions
    dx = int((endX - startX) * padding)
    dy = int((endY - startY) * padding)
    
    # apply padding to each side of the bounding box respectivly
    startX = max(0, startX - dx)
    startY = max(0, startY - dy)
    endX = min(imageW, endX + (dx * 2))
    endY = min(imageH, endY + (dy * 2))
    
    # extract the actual padded roi "Region of Interest"
    roi = image[startY:endY, startX:endX]
    
    # in order to apply tesseract v4 OCR to text we Must supply (1) a language,
    # (2) an OEM "OCR Engine Mode", indicating the algorithm we wish to use "1- for LSTM NN Model"
    # (3) an PSM "Page Segmentation Mode" , "7- implies that we treat ROI as a single line of text"
    
    # tesseract --help-oem: show available OCR Engine modes
    # 0 Legacy engine only, 1 Neural nets LSTM engine only, 2 Legacy + LSTM engines, 
    # 3 Default based on what is available.
    
    # tesseract --help-psm: show Page Segmentation Modes:
    # 0  Orientation and script detection (OSD) only.
    # 1  Automatic page segmentation with OSD.
    # 2  Automatic page segmentation, but no OSD, or OCR. (not implemented)
    # 3  Fully automatic page segmentation, but no OSD. (Default)
    # 4  Assume a single column of text of variable sizes.
    # 5  Assume a single uniform block of vertically aligned text.
    # 6  Assume a single uniform block of text.
    # 7  Treat the image as a single text line.
    # 8  Treat the image as a single word.
    # 9  Treat the image as a single word in a circle.
    # 10 Treat the image as a single character.
    # 11 Sparse text. Find as much text as possible in no particular order.
    # 12 Sparse text with OSD.
    # 13 Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
    
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, config=config)
    
    # add the bounding box coordinates and OCR'd text to the list of results
    results.append(((startX, startY, endX, endY), text))

    
# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][1])

# loop over the results
for ((startX, startY, endX, endY), text) in results:
    # display the text OCR'd by Tesseract
    print("OCR TEXT")
    print("========")
    print("{}\n".format(text))
    
    # strip out non-ASCII text so we can draw the text on the image using OpenCV,
    # then draw the text and a bounding box surrounding the text region of the input image
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = image.copy()
    cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(output, text, (startX, startY - 3), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    # show the output image
    cv2.imshow("Text Detection", output)
    cv2.waitKey(0)
    
    
    