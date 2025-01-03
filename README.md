# Landmark-Recognition-Bot
## Проект: Чат-бот для определения достопримечательностей по фото
## Описание
Этот проект представляет собой чат-бот, основанный на модели машинного обучения, который может определять достопримечательности на основе предоставленных фотографий. Целью проекта является создание удобного инструмента, который позволяет пользователям отправлять фотографии и получать информацию о достопримечательностях.

## Описание концепции продукта

Возможные области применения:
- **Туристические приложения**: Пользователи могут узнать о достопримечательностях, отправив фотографию через чат-бот.
- **Образовательные платформы**: Интеграция чат-бота в образовательные системы для обучения истории и культуры с помощью фотографий.
- **Геолокационные сервисы**: Улучшение точности определения местоположения с помощью анализа фотографий.
- **Архивирование и цифровизация**: Использование чат-бота для маркировки и организации больших архивов фотографий.

### Информация по проекту:
- presentation.pptx

## Структура проекта
- classes.txt (блокнот, который содержит классы объектов)
- vgg13.pth (модель, которая обработывает фотографии), на гит не помещается, ссылка на гугл диск: https://drive.google.com/file/d/19OEn47zfS5qaoa94AjDpntH5SMsyQ6QP/view?usp=sharing
- bot.py (Инициализация и насторйка Telegram-бота)
- model_loader.py (Моделирование и предсказание изображений)

### Использование телеграм-бота:
Разработан телеграм-бот для использования полученной модели при определении достопримечательностей по фото. Файл запуска и ссылка:
- bot.py
- Ссылка на телеграм-бот для определения достопримечательностей: @Hackaton_Facts_bot

## Полученные результаты
Был разработан чат-бот, который определяет достопримечательность по отправленной фотографии.

## Перспективы и идеи для дальнейшего развития:
- **Интеграция с OpenAI**: Добавление связи с OpenAi, которая будет выдавать пользователю факты о определенной достопримечательности.
- **Увеличение количества классов**: Добавление новых классов достопримечательностей для расширения охвата модели.
- **Использование ансамблей моделей**: Комбинирование нескольких моделей для улучшения точности предсказаний и уменьшения вероятности ошибок.

### Краткое резюме:
Разработанный телеграм-бот обеспечивает удобный интерфейс для взаимодействия пользователей с моделью, позволяя легко получать информацию о достопримечательностях. 
