import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os

# Функция для загрузки списка классов
def load_class_names(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден!")
    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Функция для загрузки модели
def load_model():
    # Создаем модель VGG13
    model = models.vgg13_bn(pretrained=False)  # Используем untrained модель
    model.classifier[-4] = nn.Linear(in_features=4096, out_features=1024)
    model.classifier[-1] = nn.Linear(in_features=1024, out_features=210)
    
    model_path = './models/vgg13.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError('VGG13 model is not available. Please download it first.')
    
    # Загружаем веса модели
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Используем CPU по умолчанию
    model.eval()  # Переводим модель в режим инференса
    return model

# Функция для предсказания содержания изображения
def predict_image(image_path, model):
    # Загрузка списка классов
    class_names = load_class_names('./data/classes.txt')

    # Преобразование изображения с помощью Albumentations
    tf = A.Compose([
        A.Resize(224, 224),  # Изменяем размер до 224x224, как обучалась модель
    ])
    
    # Чтение и преобразование изображения
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразование из BGR в RGB
    transformed = tf(image=image)
    
    # Нормализация пикселей изображения до диапазона [0, 1]
    image = transformed['image'] / 255.0
    
    # Преобразование изображения в tensor и приведение к типу float32
    tensor = ToTensorV2()(image=image)['image'].unsqueeze(0).float()  # Добавляем размер батча (1) и приводим к float32

    # Передача изображения через модель
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()  # Получаем индекс предсказания
        class_label = class_names[prediction]  # Преобразуем индекс в метку

    return class_label
