import cv2
import torch
import torchvision
from torchvision.transforms import functional as F

# Загрузка предобученной модели детекции из torchvision (второй фреймворк!)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()


COCO_INSTANCE_CATEGORY_NAMES = [
	'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
	'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign'
]

video_path = "../samples/test_video.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
	ret, frame = cap.read()
	if not ret or frame_count > 30:
		break

	img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	img_tensor = F.to_tensor(img_rgb).unsqueeze(0)

	with torch.no_grad():
		predictions = model(img_tensor)

	labels = predictions[0]['labels'].numpy()
	scores = predictions[0]['scores'].numpy()

	detected = [COCO_INSTANCE_CATEGORY_NAMES[i] for i, s in zip(labels, scores) if
				s > 0.8 and i < len(COCO_INSTANCE_CATEGORY_NAMES)]
	print(f"Frame {frame_count}: objects defined -> {set(detected)}")
	frame_count += 1

cap.release()