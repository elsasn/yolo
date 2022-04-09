import os

class Path():
    root_dir = os.getcwd()
    vid_input = r'input_files/vid.mp4'
    MOG_out = root_dir + '/outputs/from_MOG/'
    YOLO_out = root_dir + '/outputs/from_YOLO/'
    END_out = root_dir + '/outputs/end_result_bbox/'
    MOG_bbox = root_dir + '/outputs/bboxes/MOG.csv'
    YOLO_bbox = root_dir + '/outputs/bboxes/YOLO.csv'
    raw = root_dir + '/outputs/raw_frames/'
    model_yolov3 = root_dir + '/models/yolov3.weights'
    model_yolov3_config = root_dir + '/models/yolov3.cfg'
    classes_names = root_dir + '/models/coco.names'