from ultralytics import YOLO

model = YOLO('yolov8s-pose.pt')
results = model.predict(source='../dataset/mocap_8.v1i.coco/train', save=True)