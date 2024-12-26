from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import logging

# Импорт функций обработки команд и изображений
from model_loader import load_model, predict_image

# Настроим логирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Включаем токен для вашего бота
TELEGRAM_TOKEN = '7742974007:AAE_cGKcVhQegxktTmCKjx_cXFvQHfUA8OE'

async def start(update: Update, context: CallbackContext) -> None:
    """Отправляет сообщение при начале работы с ботом."""
    await update.message.reply_text('Привет! Отправьте мне изображение.')

async def handle_image(update: Update, context: CallbackContext) -> None:
    """Обрабатывает изображение, которое прислал пользователь."""
    try:
        # Получаем фотографию, отправленную пользователем
        file = await update.message.photo[-1].get_file()
        file_path = await file.download_to_drive('image.jpg')

        # Загружаем модель и предсказываем результат
        model = load_model()
        prediction = predict_image('image.jpg', model)
        
        # Отправляем результат пользователю
        await update.message.reply_text(f"На изображении: {prediction}")
    except Exception as e:
        # Обрабатываем ошибки и отправляем сообщение пользователю
        await update.message.reply_text(f"Произошла ошибка: {str(e)}")

def main():
    # Создаем приложение с токеном вашего бота
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Добавляем обработчики команд и сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Запускаем бота
    application.run_polling()

if __name__ == '__main__':
    main()
