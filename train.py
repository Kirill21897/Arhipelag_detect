from ultralytics import YOLO
import os
from pathlib import Path

# Убедимся, что пути существуют
data_dir = Path('data')
assert (data_dir / 'images').exists(), "Папка images не найдена"
assert (data_dir / 'labels').exists(), "Папка labels не найдена"
assert (data_dir / 'data.yaml').exists(), "Файл data.yaml не найден"

# Инициализация модели
model = YOLO('yolov8n.pt')  # или ваш custom pretrained weights

# Обучение с абсолютными путями
results = model.train(
    data=str(data_dir / 'data.yaml'),  # абсолютный путь
    epochs=50,
    imgsz=640,
    batch=8,
    device='cpu',
    workers=4  # количество ядер CPU
)

print("Обучение завершено! Модель сохранена в runs/detect/train/weights/best.pt")