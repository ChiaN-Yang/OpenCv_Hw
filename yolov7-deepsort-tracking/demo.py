from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

detector = Detector(iou_thresh = 0.25, classes = [0,32]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('./weights/yolov7x.pt',) # pass the path to the trained weight file

# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

# output = None will not save the output video
tracker.track_video("test.mp4", output="fifa.mp4", show_live = False, skip_frames = 0, count_objects = True, verbose=1)