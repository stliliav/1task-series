import torchvision

# Скачивание модели Faster R-CNN из библиотеки torchvision (PyTorch)
print("Загрузка модели детекции для видео...")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
print("Модель для видео успешно загружена!")