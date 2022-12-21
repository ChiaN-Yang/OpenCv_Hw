## Installation
Before run this program, you need to install package below:
- matplotlib
- numpy
- opencv-python
- Pillow
- PyYAML
- requests
- scipy
- torch
- torchvision
- tqdm
- protobuf

It can be installed with pip:

    pip install -r requirements.txt

## Training
    python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 4 --device 0,1 --sync-bn --batch-size 64 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

## Download a video from Youtube
    jupyter nbconvert --execute Demo.ipynb

## Inference
    python detect.py --weights ./weights/yolov7.pt --img-size 640 --source test.mp4 --device 0,1 --classes 0 32 --iou 0.25

## Hint
- You can find the classes names in [here](https://github.com/ChiaN-Yang/OpenCv_Hw/blob/master/yolov7-deepsort-tracking/tracking_helpers.py#L247)
- The inferred videos will be saved in a folder called RUNS