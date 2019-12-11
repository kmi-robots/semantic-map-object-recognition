"""
Extracting bounding boxes predicted by YOLO,
regardless of image class/label

Derived from code at
https://github.com/meenavyas/Misc/blob/master/ObjectDetectionUsingYolo/ObjectDetectionUsingYolo.ipynb
"""
from __future__ import (division, absolute_import, print_function, unicode_literals)

import torch
import cv2
import numpy as np
import torchvision

from torchvision import transforms as T
from PIL import Image
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from collections import Counter


# 'path to yolo config file'
CONFIG='./data/yolo/yolov3.cfg'

# 'path to text file containing class names'
CLASSES='./data/yolo/yolov3.txt'

# 'path to yolo pre-trained weights'
# wget https://pjreddie.com/media/files/yolov3.weights
WEIGHTS='./data/yolo/yolov3.weights'


# read class names from text file
classes = None
with open(CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()] + ['N/A','saliency region']

scale = 0.00392         # 1/255.  factor
conf_threshold = 0.1 #0.01   #0.5
nms_threshold = 0.1     #0.4
overlapThresh = 0.4
low = 2000
high = 30000

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# function to get the output layer names
# in the architecture

yolonet = cv2.dnn.readNet(WEIGHTS, CONFIG)


#Mask RCNN
#segm_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_nms_thresh=0.0001)
# Faster RCNN
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_nms_thresh=0.0001)
#segm_model.eval()

#Loading Mask RCNN through OpenCV instead
textGraph = "./data/opencv-zoo/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt" #created through opencv-4.0.0/samples/dnn/tf_text_graph_mask_rcnn.py
modelWeights = "./data/opencv-zoo/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
segm_model = None #cv2.dnn.readNetFromTensorflow(modelWeights, textGraph)
#segm_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#segm_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

def get_output_layers(net):

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_obj_saliency_map(rgb_img):

    saliency = cv2.saliency.ObjectnessBING_create()
    saliency.setTrainingPath('./data/objectness_trained_model')
    success_flag, saliencyMap = saliency.computeSaliency(rgb_img)

    return saliencyMap

def get_static_saliency_map(rgb_img):

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()#StaticSaliencyFineGrained_create()
    success_flag, saliencyMap = saliency.computeSaliency(rgb_img)

    return (saliencyMap * 255).astype("uint8")



def checkRectangles(rects, overlapThresh, tol=100):

    rectsCC = rects.copy() #copy of rect

    for coords1,label1 in rects:

        # print(coords1)

        # top-left coordinates (origin point)
        x1 = coords1[0]
        y1 = coords1[1]
        # bottom right coordinates
        x_br_1 = coords1[2]
        y_br_1 = coords1[3]

        area1 = (x_br_1 - x1 + 1) * (y_br_1 - y1 + 1)
        pos = rects.index((coords1,label1))

        # Loops over all the other elements except the firstly considered one
        for coords2, label2 in [(r,l) for (r,l) in rects if rects.index((r,l)) != pos]:

            # print(coords2)

            # top-left coordinates (origin point)
            x2 = coords2[0]
            y2 = coords2[1]
            # bottom right coordinates
            x_br_2 = coords2[2]
            y_br_2 = coords2[3]

            pos2 = rects.index((coords2, label2))

            # If either rectangle 2 is included in rectangle 1
            if (x2 >= x1-tol) and (y2 >= y1-tol) and (x_br_2 <= x_br_1+tol) and (y_br_2 <= y_br_1+tol):

                # then rect2 is completely enclosed in rect1

                #compute IoU then
                xA = max(x1, x2)
                yA = max(y1, y2)
                xB = min(x_br_1, x_br_2)
                yB = min(y_br_1, y_br_2)

                # compute the area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                area2 = (x_br_2 - x2 + 1) * (y_br_2 - y2 + 1)

                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(area1 + area2 - interArea)

                #Filter based on that
                if iou >= overlapThresh:

                    try:

                        # But keep candidate label in other box
                        rectsCC[pos] = (coords1, rectsCC[pos][1] + " " + label2)

                        rectsCC.remove((coords2,label2)) #remove it from the rectangle list
                        rects.remove((coords2,label2))

                    except ValueError:

                        # could be duplicated, it was already removed
                        continue


            # or the other way around
            elif (x1 >= x2 - tol) and (y1 >= y2 -tol) and (x_br_1 <= x_br_2+tol) and (y_br_1 <= y_br_2+tol):

                xA = max(x1, x2)
                yA = max(y1, y2)
                xB = min(x_br_1, x_br_2)
                yB = min(y_br_1, y_br_2)

                # compute the area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                area2 = (x_br_2 - x2 + 1) * (y_br_2 - y2 + 1)

                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(area1 + area2 - interArea)

                # Filter based on that
                if iou >= overlapThresh:

                    try:
                        rectsCC[pos2] = (coords2, rectsCC[pos2][1] + " " + label1)

                        rectsCC.remove((coords1,label1))
                        rects.remove((coords1,label1))



                    except ValueError:

                        #could be duplicated, it was already removed
                        continue


    # returns the updated list after this check
    return rectsCC


def run_YOLO(net, mu=None, w=None, h=None):

    # run inference through the network
    # and gather predictions from output layers
    # print(get_output_layers(net))
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    # for each detection from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections

    for out in outs:

        for detection in out:

            scores = detection[5:]

            if np.count_nonzero(scores) > 0:

                class_id = np.argmax(scores)

            else:

                class_id = -2 #Cannot conclude on specific class, N/A

            #print(classes[class_id])
            segm_confidence = detection[4]
            #print(confidence)

            if segm_confidence >mu:

                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                """
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                w = int(detection[2] * w)
                h = int(detection[3] * h)
                x = center_x - w / 2
                y = center_y - h / 2
                """
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                class_ids.append(class_id)
                confidences.append(float(segm_confidence))
                boxes.append([x, y, int(width), int(height)])

    # apply non-max suppression

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    return boxes, confidences, indices, class_ids


def run_FRCNN_cv2(net,W, H, threshold=0.15):

    # Run the forward pass to get output from the output layers
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    numDetections = boxes.shape[2]

    pred_boxes=[]
    pred_class=[]
    pred_masks=[]

    for i in range(numDetections):

        box = boxes[0,0, i, :]
        mask = masks[i]
        score = box[2]
        id = int(box[1])

        if score > threshold:

            pred_class.append(COCO_INSTANCE_CATEGORY_NAMES[id])

            left = int(W * box[3])
            top = int(H * box[4])
            right = int(W * box[5])
            bottom = int(H * box[6])

            left = max(0, min(left, W - 1))
            top = max(0, min(top, H - 1))
            right = max(0, min(right, W - 1))
            bottom = max(0, min(bottom, H - 1))

            pred_boxes.append([(left,top),(right,bottom)])
            pred_masks.append(mask[id])


    return pred_boxes, pred_class, pred_masks



def run_FRCNN(img, threshold=0.15, nms=nms_threshold):

    #confidence used to be 0.2

    #print(np.transpose(img, (2, 0, 1)).shape)
    #img2 = Image.open('./temp.jpg')  # Load the image

    transform = T.Compose([T.ToPILImage(),T.ToTensor()])  # Defing PyTorch Transform
    img2= transform(img) # Apply the transform to the image

    #img_wrong = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    pred = segm_model([img2])#[torch.Tensor(np.transpose(img.copy(), (2, 0, 1)))])  # Pass the image to the model)

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in
                      list(pred[0]['labels'].numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())

    try:
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]  # Get list of index with score greater than threshold.

    except IndexError:

        return None, None, None

    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    masks = masks[:pred_t +1]

    return pred_boxes, pred_class, masks


def cluster_colours(image):

    d = image.copy()
    flat = d.reshape((d.shape[0] * d.shape[1], 3))
    clt = KMeans(n_clusters=5)
    clt.fit(flat)
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)

    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def white_balance(img):

    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)

    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)



# define color ranges in HSV

#primary & secondary colors
low_R = np.array([0,50,50])
up_R = np.array([29,255,255])

low_Y = np.array([30,50,50])
up_Y = np.array([59,255,255])

low_G = np.array([60, 50, 50])
up_G = np.array([89, 255, 255])

low_C = np.array([90,50,50])
up_C = np.array([119,255,255])

low_B = np.array([120,50,50])
up_B = np.array([149,255,255])

low_M = np.array([149,50,50])
up_M = np.array([179,255,255])


def get_HSV(img):

    #img = white_balance(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def segment_by_color(obj):

    # import matplotlib.pyplot as plt
    #plt.imshow(obj)
    #plt.show()

    obj_hsv = get_HSV(obj)
    pixels = Counter()

    # Threshold the HSV image to get only blue colors
    G_mask = cv2.inRange(obj_hsv, low_G, up_G)
    Y_mask = cv2.inRange(obj_hsv, low_Y, up_Y)
    R_mask = cv2.inRange(obj_hsv, low_R, up_R)
    C_mask = cv2.inRange(obj_hsv, low_C, up_C)
    B_mask = cv2.inRange(obj_hsv, low_B, up_B)
    M_mask = cv2.inRange(obj_hsv, low_M, up_M)

    #no of pixels for each color

    pixels["green"]= np.argwhere(G_mask!=0).shape[0]
    pixels["yellow"] = np.argwhere(Y_mask != 0).shape[0]
    pixels["red"] = np.argwhere(R_mask!=0).shape[0]
    pixels["cyan"] = np.argwhere(C_mask != 0).shape[0]
    pixels["blue"] = np.argwhere(B_mask!=0).shape[0]
    pixels["magenta"] = np.argwhere(M_mask != 0).shape[0]



    """
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(obj, obj.copy(), mask=G_mask)


    res = cv2.bitwise_and(obj, obj.copy(), mask=R_mask)


    res = cv2.bitwise_and(obj, obj.copy(), mask=B_mask)

    #plt.imshow(res)
    #plt.show()
    """

    return pixels

def segment(img, YOLO=True, w_saliency=False, static=False, masks=None, depth_image= None):

    Width = img.shape[1]
    Height = img.shape[0]
    tot_pix = Width*Height

    temp = img.copy()

    # Denoise image
    denoised = img # Already done in upper method
    # denoised = cv2.fastNlMeansDenoisingColored(temp, None, 10, 10, 7, 15)
    # denoised = white_balance(denoised)

    #cv2.imshow("Whiter Image", denoised)
    #cv2.waitKey(1000)


    #Cluster pixels
    # from data_loaders import BGRtoRGB
    #cluster_colours(BGRtoRGB(denoised))

    #for_mask = denoised.copy()

    if w_saliency and static:

        #Take bottom-up saliency also into account
        saliency_map = get_static_saliency_map(denoised)
        output = cv2.threshold(saliency_map.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    elif w_saliency and not static:

        saliency_map = get_obj_saliency_map(denoised)[5:,:]
        output = img.copy()

        # first dim of saliency map gives the no. of detections
        for i in range(min(saliency_map.shape[0], 5)):
            # for each candidate salient region
            startX, startY, endX, endY = saliency_map[i].flatten()

            color = np.random.randint(0, 255, size=(3,))
            color = [int(c) for c in color]
            cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

    """
    if depth_image is not None:

        #binarise based on depth value (within 50 cm)
        #print(len(depth_image[(depth_image> 0.) & (depth_image <800.) ]))

        mask = depth_image.copy()
        mask[depth_image >= 3000.] = 0. #& (depth_image < 2000.)] = 0.
        # mask[(depth_image >= 2000.) | (depth_image == 0.)] = 1.
        mask[depth_image < 3000.] = 1.

        #trying to reduce noisy edges due to invalid depth measure
        #kernel = np.ones((5, 5), np.float32) / 25
        smask = cv2.blur(mask,(5,5)) #cv2.filter2D(mask, -1, kernel)
        smask = smask.astype(np.uint8)

        #mask the RGB image based on extracted foreground
        res = cv2.bitwise_and(for_mask, for_mask, mask=smask)

        # cv2.imshow("masked Image", denoised)
        # cv2.waitKey(8000)
        # cv2.destroyAllWindows()

        
        # d_saliency_map = get_obj_saliency_map(depth_image)

        #for i in range(min(d_saliency_map.shape[0], 10)):
            # for each candidate salient region
        #    startX, startY, endX, endY = saliency_map[i].flatten()

        #    color = np.random.randint(0, 255, size=(3,))
        #    color = [int(c) for c in color]
        #    cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
    """

    # show the output image


    if YOLO:

        blob = cv2.dnn.blobFromImage(denoised, scale, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        # set input blob for the network
        yolonet.setInput(blob)

        yolo_boxes, confidences, indices, class_ids = run_YOLO(yolonet, mu=conf_threshold, w=Width,h=Height)

        #yolo_map = np.zeros_like(saliency_map)
        #predictions = []
        all_boxes=[]

        if len(list(indices))>0:
        #if len(list(boxes))>0:

            for i in indices.flatten():#for i,box in enumerate(boxes): #

                box = yolo_boxes[i]
                x = round(box[0])
                y = round(box[1])
                w = round(box[2])
                h = round(box[3])

                # in case negative coordinates are returned
                if x < 0:

                    x = 0

                elif y < 0:

                    y = 0

                area = (int(x) + int(w) - int(x) + 1) * (int(y) + int(h) - int(y) + 1)

                #assign conf value of box to all pixels in that box
                #saliency_map[x:x+w,y:y+h] = confidences[i]*100
                #if area > int(tot_pix/30):

                #draw_bounding_box(temp, class_ids[i], confidences[i], x, y, x + w, y + h)
                all_boxes.append(([x,y,x+w,y+h], str(classes[class_ids[i]])))

                #tmp = img.copy()
                #predictions.append((tmp[y:y+h, x:x+w],str(classes[class_ids[i]])))

    else:

        blob = cv2.dnn.blobFromImage(denoised, swapRB=True, crop=False)

        segm_model.setInput(blob)

        fcnn_boxes, fcnn_labels, masks = run_FRCNN_cv2(segm_model, Width, Height)  #run_FCRNN(denoised)

        all_boxes = []

        if fcnn_boxes is not None:

            for coords, label in zip(fcnn_boxes, fcnn_labels):

                x, y = coords[0]
                x2, y2 = coords[1]

                #if area > 5000:

                all_boxes.append(([int(round(x)), int(round(y)), int(round(x2)), int(round(y2))], str(label)))

    """
    if w_saliency and static:

        bin2 = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        contours, hierarchy = cv2.findContours(bin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:

           area = round(cv2.contourArea(cnt))
           x, y, w, h = cv2.boundingRect(cnt)
           area = (int(x)+int(w) - int(x) + 1) * (int(y)+int(h) - int(y) + 1)


           #Filtering the smallest ones
           if area> 5*low and area <high:

                x, y, w, h = cv2.boundingRect(cnt)
                #print(saliency_map.shape)
                #print(temp.shape)


                #Add also boxes from saliency
                all_boxes.append(([x,y,x+w,y+h],classes[-1]))
                #draw_bounding_box(temp, -1, None, x, y, x + w, y + h)

        #print(np.asarray(all_boxes).shape)
    """

    #Removing overlapping boxes

    if len(all_boxes)>1:
        
        all_boxes = checkRectangles(all_boxes, overlapThresh)


    return all_boxes, masks

def display_mask(img, mask, color, alpha=0.5):

    #Called when Mask RCNN is used and the image is semantically segmented
    #Not used when just bboxes are available
    r = np.zeros_like(mask).astype('uint8')
    g = np.zeros_like(mask).astype('uint8')
    b = np.zeros_like(mask).astype('uint8')

    r[mask==1], g[mask==1], b[mask==1] = color
    coloured_mask = np.stack([r,g,b], axis=2)

    return cv2.addWeighted(img, 1, coloured_mask, 0.3, 0)

