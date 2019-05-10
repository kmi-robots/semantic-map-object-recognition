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


def filter_boxes(img, S, confY, indY, idsY, confY_, indY_, idsY_):


    #Rs = find_maxsqall1(S)

    pass

def backproject(source, target, levels=2, scale=1):
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    # calculating object histogram
    roihist = cv2.calcHist([hsv], [0, 1], None, \
                           [levels, levels], [0, 180, 0, 256])

    # normalize histogram and apply backprojection
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], scale)
    return dst

def segment(temp_path, img):

    Width = img.shape[1]
    Height = img.shape[0]

    img = cv2.imread(temp_path)

    temp = img.copy()
    # Denoise image
    denoised = cv2.fastNlMeansDenoisingColored(temp, None, 10, 10, 7, 15)

    if w_saliency and static:

        #Take bottom-up saliency also into account
        saliency_map = get_static_saliency_map(denoised)

        bin = cv2.threshold(saliency_map.copy(), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:

            area = cv2.contourArea(cnt)
            """
            #Filtering the smallest ones
            if area > 1000:

                x, y, w, h = cv2.boundingRect(cnt)
                #print(saliency_map.shape)
                #print(temp.shape)
                avg_saliency = saliency_map[x:x+w, y:y+h].mean()
                #print(avg_saliency)

                #skip nans
                if math.isnan(avg_saliency):
                   continue

                color = COLORS[-1]
                cv2.rectangle(temp, (x, y), (x+w, y+h), color, 2)
                cv2.putText(temp, str(round(avg_saliency,2)), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                #draw_bounding_box(temp, -1, None, x, y, x + w, y + h)
            """

    elif w_saliency and not static:

        saliency_map = get_obj_saliency_map(img)

        for i in range(min(saliency_map.shape[0], 10)):
            # for each candidate salient region
            startX, startY, endX, endY = saliency_map[i].flatten()

            output = img.copy()

            draw_bounding_box(output, -1, None, startX, startY, endX, endY)

    """
    start = img.copy()

    #Mean-shift filter
    d = denoised.copy()
    mshift = cv2.pyrMeanShiftFiltering(d, 2, 10, d, 4)

    #backproject hue hist values on img itself
    backproj = np.uint8(backproject(mshift, mshift, levels=2))

    saliency_map = get_static_saliency_map(mshift) #denoised)


    #saliency_map = get_static_saliency_map()
    #binarize result

    #Find fixation points as done in lit, e.g., Karaoguz and Jensfelt 2018
    fix_points=[]

    for cnt in contours:

        #Compute centroid of each contour
        # i.e., convert from shape moments

        area = cv2.contourArea(cnt)

        if area > 2000:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            fix_points.append((cX, cY))

            x, y, w, h = cv2.boundingRect(cnt)

            draw_bounding_box(temp, -1, None, x, y, x+w, y+h)


    """

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

    boxes, confidences, indices, class_ids = run_YOLO(blob, net, mu=conf_threshold, w=Width,h=Height)

    #Algorithm to combine saliency regions with YOLO boxes
    #union = filter_boxes(temp, bin, confidences, indices, class_ids, hi_confs, hi_indices, hi_cids)

    #yolo_map = np.zeros_like(saliency_map)
    predictions = []

    if len(list(indices))>0:
    #if len(list(boxes))>0:

        for i in indices.flatten():#for i,box in enumerate(boxes): #

            box = boxes[i]
            x = round(box[0])
            y = round(box[1])
            w = round(box[2])
            h = round(box[3])

            #assign conf value of box to all pixels in that box

            saliency_map[x:x+w,y:y+h]= confidences[i]*100

            draw_bounding_box(temp, class_ids[i], confidences[i], x, y, x + w, y + h)
            tmp = img.copy()
            predictions.append((tmp[y:y+h, x:x+w],str(classes[class_ids[i]])))

    bin2 = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    temp2 = img.copy()
    contours, hierarchy = cv2.findContours(bin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        #Filtering the smallest ones
        if area > 1000:

            x, y, w, h = cv2.boundingRect(cnt)
            #print(saliency_map.shape)
            #print(temp.shape)
            """
            avg_saliency = saliency_map[x:x+w, y:y+h].mean()
            #print(avg_saliency)

            skip nans
            if math.isnan(avg_saliency):
               continue

            color = COLORS[-1]
            cv2.rectangle(temp, (x, y), (x+w, y+h), color, 2)
            cv2.putText(temp, str(round(avg_saliency,2)), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            """
            draw_bounding_box(temp, -1, None, x, y, x + w, y + h)



    # Visualise saliency+yoloconf regions
    #cv2.imshow('Saliency', temp2)
    #cv2.waitKey(5000)
    #cv2.destroyAllWindows()

    # display yolo-only image

    cv2.imshow('union',temp)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    return predictions

