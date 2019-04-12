"""
Extracting bounding boxes predicted by YOLO,
regardless of image class/label

Derived from code at
https://github.com/meenavyas/Misc/blob/master/ObjectDetectionUsingYolo/ObjectDetectionUsingYolo.ipynb
"""

import cv2
import numpy as np

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
    classes = [line.strip() for line in f.readlines()]

scale = 0.00392 # 1/255.  factor
conf_threshold = 0.0 #0.5
nms_threshold = 0.0 #0.4

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# function to get the output layer names
# in the architecture

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


def segment(temp_path, img):

    Width = img.shape[1]
    Height = img.shape[0]

    img = cv2.imread(temp_path)

    # read pre-trained model and config file
    net = cv2.dnn.readNet(WEIGHTS, CONFIG)

    # create input blob
    blob = cv2.dnn.blobFromImage(img, scale, (416, 416), (0, 0, 0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    #print(get_output_layers(net))
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box

    predictions = []

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = round(box[0])
        y = round(box[1])
        w = round(box[2])
        h = round(box[3])

        #draw_bounding_box(img, class_ids[i], confidences[i], x, y, x + w, y + h)
        predictions.append((img[y:y+h, x:x+w],str(classes[class_ids[i]])))

    # display output image

    #cv2.imshow('prediction',img)
    #cv2.waitKey(5000)
    #cv2.destroyAllWindows()

    return predictions

