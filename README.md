# **Настройка бота**
## Добавление токена бота .env файла
Создайте файл '.env' в директории telegram_bot

![img.png](img_to_read_me/telegram_bot.png)

Добавьте свой токен telegram в переменную "TOKEN"

![img.png](img_to_read_me/TOKEN.png)

## Добавление модели, чтобы не тренировать её

Скачать модель нужно тут - https://disk.yandex.ru/d/xiv2M3o7rVKpZQ

Скаченную папку sentiment_model нужно закинуть в директорию model_training

##### **Архитектура модели -** 

![img.png](img_to_read_me/sentiment_model.png)


## Установка библиотек

pip install -r requirements.txt

## Итоговая архитектура проекта:

![img.png](img.png)