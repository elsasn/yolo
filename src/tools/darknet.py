import cv2
import os
import numpy as np
from src.config.Path import *
import glob

def bb_intersection_over_union(boxA, boxB):
    print(boxA)
    print(boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = abs((xB - xA) * (yB - yA))
    print(interArea)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2]) * (boxA[3])
    boxBArea = (boxB[2]) * (boxB[3])
    print(boxAArea)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def yolo(raw_input, input_from_mog, out_yolo, out_end, mog_bbox):

    image = cv2.imread(raw_input)
    mog_image = cv2.imread(input_from_mog)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    classes = None

    with open(Path.classes_names, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    net = cv2.dnn.readNet(Path.model_yolov3, Path.model_yolov3_config)
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, label):
        color = COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, str(label), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        if (class_ids[i] == 0):
            draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),'')
            draw_bounding_box(mog_image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), bb_intersection_over_union(box, mog_bbox))

    cv2.imwrite(out_yolo, image)
    cv2.imwrite(out_end, mog_image)

def process():

    raw_file = []
    boxed_by_mog_file = []

    for filename in glob.iglob(f'{Path.raw}/*'):
        raw_file.append(filename)
    
    for filename in glob.iglob(f'{Path.MOG_out}/*'):
        boxed_by_mog_file.append(filename)
    
    cnt = 0
    for filename in glob.iglob(f'{Path.MOG_out}/*'):
        file = open(Path.MOG_bbox)
        content = file.readlines()[cnt]
        content = content.split(",")
        data = content.pop(0)
        data = data[39:len(data) - 9]
        print(Path.raw + data)
        content = np.asarray(content)
        content = content.astype(float)
        out_yolo = os.path.join(Path.YOLO_out ,"%d_boxed.jpg") % cnt
        out_end = os.path.join(Path.END_out ,"%d_end.jpg") % cnt
        yolo(Path.raw + data  + 'nobox.jpg', Path.MOG_out + data  + 'boxed.jpg', out_yolo, out_end, content)
        cnt = cnt + 1
        print(cnt)
