from ultralytics import YOLO
import os

def main():
    # Загружаем предобученную модель YOLOv8 (версия nano для скорости)
    model = YOLO('yolov8n.pt') 
    
    # Путь к нашему конфигурационному файлу датасета
    dataset_path = os.path.join('dataset', 'dataset.yaml')

    print("Начинаем обучение модели...")
    
    # Обучение нейросети
    # epochs - количество эпох (проходов по всему датасету)
    # imgsz - размер изображения, к которому оно будет сжато
    results = model.train(
        data=dataset_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name='plant_recognition_model'
    )
    
    print("Обучение завершено. Результаты сохранены в папке runs/")

if __name__ == '__main__':
    main()