from ultralytics import YOLO

# Load the pretrained YOLOv8 pose estimation model (small variant)
model = YOLO('yolov8s-pose.pt')

# Run pose estimation on all images in the specified folder,
# saving annotated output images to the default runs/ directory
results = model.predict(
    source='../dataset/mocap_8.v1i.coco/train',  # input directory or file pattern
    save=True                                   # save the visualized predictions
)
