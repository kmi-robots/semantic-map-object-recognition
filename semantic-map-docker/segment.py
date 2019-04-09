"""
Extracting bounding boxes predicted by YOLO,
regardless of image class/label


Derived from code at
https://github.com/meenavyas/Misc/blob/master/ObjectDetectionUsingYolo/ObjectDetectionUsingYolo.ipynb
"""

import cv2
import numpy as np

# 'path to yolo config file'
# download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.cfg
CONFIG='./data/yolo/yolov3.cfg'

# 'path to text file containing class names'
# download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
CLASSES='./data/yolo/yolov3.txt'

# 'path to yolo pre-trained weights'
# wget https://pjreddie.com/media/files/yolov3.weights
WEIGHTS='./data/yolo/yolov3.weights'

# read class names from text file
classes = None
with open(CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

scale = 0.00392
conf_threshold = 0.0
nms_threshold = 0.1 #0.4

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
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    # display output image
    #out_image_name = "object detection" + str(index)
    cv2.imshow('prediction',img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()



"""
import lightnet


model = lightnet.load('yolo')


def find_bboxes(out, rgb_image=None, thr=None):

    img = lightnet.Image.from_bytes(open(out, 'rb').read())

    # Object detection threshold defaults to 0.1 here
    #img = lightnet.Image(rgb_image) #Reading from NP array is documented on the lightnet page but not reliable

    boxes = model(img, thresh=thr)

    #print(boxes)
    # Coordinates in YOLO are relative to center coordinates
    boxs_coord = [(int(x), int(y), int(w), int(h)) for cat, name, conf, (x, y, w, h) in boxes]

    #print(boxs_coord)
    return boxs_coord


def convert_bboxes(box_list, shape, resolution=(416, 312)):

    ow, oh, _ = shape
    print(shape)
    tgt_w, tgt_h = resolution

    new_bx = []

    for x, y, w, h in box_list:

        print("Original: (%s, %s, %s, %s)" % (x, y, w, h))

        # Make them absolute from relative
        x_ = x  # *tgt_w
        y_ = y  # *tgt_h
        w_ = w  # *tgt_w
        h_ = h  # *tgt_h

        #print("Scaled: (%s, %s, %s, %s)" % (x_, y_, w_, h_))
        # And change coord system for later cropping
        x1 = int(x_ - w_ / 2)  # /ow
        y1 = int(y_ - h_ / 2)  # /oh
        x2 = int(x_ + w_ / 2)  # /ow
        y2 = int(y_ + h_ / 2)  # /oh

        # Add check taken from draw_detections method in Darknet's image.c
        if x1 < 0:
            x1 = 0
        if x2 > ow - 1:
            x2 = ow - 1

        if y1 < 0:
            y1 = 0

        if y2 > oh - 1:
            y2 = oh - 1

        print("For ROI: (%s, %s, %s, %s)" % (x1, y1, x2, y2))
        new_bx.append((x1, y1, x2, y2))

        #if len(new_bx)>1:
        #    print("More than one bbox found!")
    return new_bx


def crop_img(rgb_image, boxs_coord):

    return [rgb_image[y1:y2, x1:x2] for (x1,y1,x2,y2) in boxs_coord]

"""