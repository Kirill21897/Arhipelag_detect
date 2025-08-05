import os
import sys
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
# Импортируем вашу функцию predict
from solution import predict


# [ Параметры ]
EXTENSION = '.jpg' # Убедитесь, что это правильное расширение ваших тестовых изображений
COLUMNS = ['image_id', 'xc', 'yc', 'w', 'h', 'label', 'score', 'time_spent', 'w_img', 'h_img']

# Фиксируем сиды для воспроизводимости (необязательно для проверки)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def process_images(images_path: str, result_csv_path: str = None) -> pd.DataFrame:
    """Функция обработки папки с изображениями."""
    # Тестовая папка должна содержать подпапку images
    images_dir = os.path.join(images_path, 'images')
    
    if not os.path.exists(images_dir):
        raise Exception(f"Директория {images_dir} не найдена!")
    
    image_paths = list(Path(images_dir).glob(f'*{EXTENSION}'))
    
    if not image_paths:
        raise Exception(f'Отсутствуют изображения в папке {images_dir}')

    # Сортируем пути для воспроизводимости
    image_paths = sorted(image_paths)
    print(f"Найдено {len(image_paths)} изображений для обработки.")

    results = []
    # Обрабатываем изображения по одному
    for i, image_path in enumerate(image_paths):
        image_id = os.path.basename(image_path).split(EXTENSION)[0]
        print(f"Обработка изображения {i+1}/{len(image_paths)}: {image_id}")
        
        # Открываем изображение в RGB формате
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Ошибка при загрузке изображения {image_path}")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img = image.shape[:2]

        # Засекаем время выполнения функции predict
        start_time = time.time()
        # Вызываем функцию predict для одного изображения
        image_results = predict([image])
        # Останавливаем таймер
        elapsed_time = time.time() - start_time
        time_per_image = round(elapsed_time, 6)
        
        # Дополняем результаты ID изображения и затраченным временем
        if image_results and image_results[0]:
            # Только если есть результаты (найдены объекты)
            for res in image_results[0]:
                res['image_id'] = image_id
                res['time_spent'] = time_per_image
                res['w_img'] = int(w_img)
                res['h_img'] = int(h_img)
                results.append(res)
        # Если объектов не найдено, НЕ добавляем никаких строк
        # Это важно для избежания ошибки '<' not supported between instances of 'NoneType' and 'float'

    result_df = pd.DataFrame()
    if result_csv_path:
        if results:
            # Создаем DataFrame только из записей с предсказаниями
            result_df = pd.DataFrame(results)
            
            # Убедимся, что все необходимые колонки присутствуют
            for col in COLUMNS:
                if col not in result_df.columns:
                    # Для новых колонок заполняем значениями по умолчанию или NaN
                    if col in ['w_img', 'h_img']:
                        result_df[col] = np.nan 
                    else:
                        result_df[col] = np.nan
            
            # Переупорядочим колонки в нужном порядке
            result_df = result_df[COLUMNS]
            
            # Убедимся, что все числовые поля имеют правильный тип
            numeric_columns = ['xc', 'yc', 'w', 'h', 'label', 'score', 'time_spent', 'w_img', 'h_img']
            for col in numeric_columns:
                if col in result_df.columns:
                    # Преобразуем в числовой тип, ошибки становятся NaN
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            
            # === КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ ===
            # Удаляем все строки с NaN в ключевых полях ДО сохранения
            # Это предотвращает ошибку в метрике
            result_df = result_df.dropna(subset=['xc', 'yc', 'w', 'h', 'score'])
            # === КОНЕЦ КРИТИЧЕСКОГО ИЗМЕНЕНИЯ ===
            
            # Сортируем по image_id для воспроизводимости
            result_df = result_df.sort_values('image_id').reset_index(drop=True)
            
            result_df.to_csv(result_csv_path, index=False)
            print(f'Результаты сохранены в {result_csv_path}')
            print(f'Всего обнаружено объектов: {len(result_df)}')
        else:
            # Если вообще не найдено объектов, создаем пустой файл с заголовками
            # Метрика ожидает файл с заголовками, даже если он пустой
            empty_df = pd.DataFrame(columns=COLUMNS)
            empty_df.to_csv(result_csv_path, index=False)
            print(f'Создан пустой файл результатов: {result_csv_path}')
    
    print('Обработка выборки выполнена успешно!')
    return result_df # Возвращаем пустой df, если results пуст


if __name__ == '__main__':
    # Проверяем количество аргументов командной строки
    # Для локальной проверки можно задать пути напрямую
    if len(sys.argv) == 3:
        # Стандартный режим запуска через платформу
        DATASET_PATH = sys.argv[1]
        CSV_PATH = sys.argv[2]
    elif len(sys.argv) == 1:
        # Режим для локальной проверки (без аргументов)
        # === ВАЖНО: УКАЖИТЕ ВАШИ ПУТИ ДЛЯ ЛОКАЛЬНОЙ ПРОВЕРКИ ===
        DATASET_PATH = "./test_data" # Путь к папке с тестовыми данными (содержит подпапку images)
        CSV_PATH = "./solution.csv"  # Путь, куда будет сохранен результат
        print("=== ЛОКАЛЬНЫЙ РЕЖИМ ПРОВЕРКИ ===")
        print(f"Путь к данным: {DATASET_PATH}")
        print(f"Путь к результату: {CSV_PATH}")
    else:
        print("Использование:")
        print("  Для платформы: python main.py <путь_к_директории_с_изображениями> <путь_к_выходному_csv>")
        print("  Для локальной проверки: python main.py (укажите пути внутри скрипта)")
        sys.exit(1)

    print(f'Инференс модели на выборке из {DATASET_PATH}')
    predicted_df = process_images(DATASET_PATH, CSV_PATH)