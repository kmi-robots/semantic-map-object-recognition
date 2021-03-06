"""
Extracting bounding boxes predicted by YOLO,
regardless of image class/label

Derived from code at
https://github.com/meenavyas/Misc/blob/master/ObjectDetectionUsingYolo/ObjectDetectionUsingYolo.ipynb
"""

import cv2
import numpy as np
import math


w_saliency = True
static= True

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

scale = 0.00392 # 1/255.  factor
conf_threshold = 0.01 #0.5
nms_threshold = 0.1 #0.4
overlapThresh = 0.1

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# function to get the output layer names
# in the architecture

net = cv2.dnn.readNet(WEIGHTS, CONFIG)

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

                    rectsCC.remove((coords2,label2)) #remove it from the rectangle list
                    rects.remove((coords2,label2))


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

                    rectsCC.remove((coords1,label1))
                    rects.remove((coords1,label1))


    # returns the updated list after this check
    return rectsCC


def run_YOLO(blob, net, mu=None, w=None, h=None):


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


def segment(temp_path, img):

    Width = img.shape[1]
    Height = img.shape[0]

    img = cv2.imread(temp_path)

    temp = img.copy()
    temp2 = img.copy()

    # Denoise image
    denoised = cv2.fastNlMeansDenoisingColored(temp, None, 10, 10, 7, 15)

    if w_saliency and static:

        #Take bottom-up saliency also into account
        saliency_map = get_static_saliency_map(denoised)


    elif w_saliency and not static:

        saliency_map = get_obj_saliency_map(img)

        for i in range(min(saliency_map.shape[0], 10)):
            # for each candidate salient region
            startX, startY, endX, endY = saliency_map[i].flatten()

            output = img.copy()

            #draw_bounding_box(output, -1, None, startX, startY, endX, endY)

    #Visualise binarised saliency regions
    #cv2.imshow('Saliency',bin)
    #cv2.waitKey(5000)
    #cv2.destroyAllWindows()


    #Visualise bboxes for saliency regions only
    #cv2.imshow('Saliency', img)
    #cv2.waitKey(5000)
    #cv2.destroyAllWindows()

    # set input blob for the network

    blob = cv2.dnn.blobFromImage(denoised, scale, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    yolo_boxes, confidences, indices, class_ids = run_YOLO(blob, net, mu=conf_threshold, w=Width,h=Height)

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

            #assign conf value of box to all pixels in that box
            #saliency_map[x:x+w,y:y+h] = confidences[i]*100

            #draw_bounding_box(temp, class_ids[i], confidences[i], x, y, x + w, y + h)
            all_boxes.append(([x,y,x+w,y+h], str(classes[class_ids[i]])))

            tmp = img.copy()
            #predictions.append((tmp[y:y+h, x:x+w],str(classes[class_ids[i]])))

    bin2 = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    contours, hierarchy = cv2.findContours(bin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

       area = round(cv2.contourArea(cnt))
       x, y, w, h = cv2.boundingRect(cnt)
       area = (int(x)+int(w) - int(x) + 1) * (int(y)+int(h) - int(y) + 1)

       #Filtering the smallest ones
       if area> 2000:

            x, y, w, h = cv2.boundingRect(cnt)
            #print(saliency_map.shape)
            #print(temp.shape)

            #Add also boxes from saliency
            all_boxes.append(([x,y,x+w,y+h],classes[-1]))
            #draw_bounding_box(temp, -1, None, x, y, x + w, y + h)

    #print(np.asarray(all_boxes).shape)
    filtered_boxes = checkRectangles(all_boxes, overlapThresh)
    #print(len(filtered_boxes))

    """
    for box,l in filtered_boxes:

        x = box[0]
        y = box[1]
        x2 = box[2]
        y2 = box[3]


        color = COLORS[classes.index(l)]
        cv2.rectangle(temp2, (x, y), (x2, y2), color, 2)
        cv2.putText(temp2, l, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Visualise saliency+yoloconf regions
    #cv2.imshow('Saliency', temp2)
    #cv2.waitKey(5000)
    #cv2.destroyAllWindows()


    cv2.imshow('union', temp)
    cv2.waitKey(6000)
    cv2.destroyAllWindows()

    cv2.imshow('union',temp2)
    cv2.waitKey(6000)
    cv2.destroyAllWindows()

    """
    # Visualise saliency+yoloconf regions
    cv2.imshow('Saliency', img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return filtered_boxes

