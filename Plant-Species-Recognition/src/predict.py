from ultralytics import YOLO
import cv2

def predict_image(image_path):
    # Загружаем наши обученные веса (появятся после запуска train.py)
    # Путь может немного отличаться в зависимости от папки runs/
    model = YOLO('runs/detect/plant_recognition_model/weights/best.pt')

    # Делаем предсказание
    results = model(image_path)

    # Показываем результат
    for r in results:
        # Получаем изображение с нарисованными рамками
        im_array = r.plot()  
        cv2.imshow('Prediction', im_array)
        cv2.waitKey(0)  # Ждем нажатия любой клавиши
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Тестовый запуск: замените 'test_image.jpg' на реальный путь к фото
    # predict_image('test_image.jpg')
    print("Запустите функцию predict_image с путем к вашей картинке.")