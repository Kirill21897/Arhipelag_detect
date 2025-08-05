import numpy as np
from typing import List, Union
import torch
from ultralytics import YOLO
import os

# === ВАЖНО: Укажите путь к вашей модели ===
MODEL_PATH = "best.pt" # Измените, если файл имеет другое имя или путь

# Инициализация модели (загрузка происходит один раз при импорте модуля)
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"Модель {MODEL_PATH} загружена успешно")
    else:
        raise FileNotFoundError(f"Файл модели {MODEL_PATH} не найден!")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    model = None


def infer_image_bbox(image: np.ndarray) -> List[dict]:
    """Функция для получения ограничивающих рамок объектов на изображении."""
    res_list = []
    
    if model is None:
        print("Модель не загружена. Возвращаем пустой результат.")
        return res_list

    try:
        # Выполняем предсказание
        # Убедитесь, что размер изображения (imgsz) соответствует тому, на котором обучалась модель
        # device=0 для использования GPU, device='cpu' для CPU
        results = model.predict(
            source=image,
            imgsz=640, # Или другой размер, если модель обучалась на другом
            conf=0.25, # Порог уверенности (можно настроить)
            iou=0.45,  # Порог для NMS (можно настроить)
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=False # Отключаем вывод логов
        )
        
        # Обрабатываем результаты
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Получаем нормализованные координаты (в формате YOLO)
                    xc = float(box.xywhn[0][0])
                    yc = float(box.xywhn[0][1])
                    w = float(box.xywhn[0][2])
                    h = float(box.xywhn[0][3])
                    conf = float(box.conf[0])
                    # Для этой задачи label всегда 0
                    # Если ваша модель обучена на нескольких классах, уточните логику
                    label = 0 # int(box.cls[0]) если нужно брать из модели
                    
                    # Добавляем проверку на корректность значений
                    # Это критично для прохождения валидации метрики
                    if (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 
                        0.0 < w <= 1.0 and 0.0 < h <= 1.0 and 
                        0.0 <= conf <= 1.0):
                        formatted = {
                            'xc': round(xc, 6),
                            'yc': round(yc, 6),
                            'w': round(w, 6),
                            'h': round(h, 6),
                            'label': label,
                            'score': round(conf, 6)
                        }
                        res_list.append(formatted)
                    # else:
                        # print(f"Пропущено предсказание с некорректными координатами: xc={xc}, yc={yc}, w={w}, h={h}, conf={conf}")
    except Exception as e:
        print(f"Ошибка при инференсе модели: {e}")

    return res_list


def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    """Функция производит инференс модели на одном или нескольких изображениях."""
    results = []
    if isinstance(images, np.ndarray):
        images = [images]

    # Обрабатываем каждое изображение из полученного списка
    for image in images:        
        image_results = infer_image_bbox(image)
        results.append(image_results)
    
    return results