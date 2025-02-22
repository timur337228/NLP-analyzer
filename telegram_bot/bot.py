from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart, Command
import telegram_bot.keyboards as kb
from telegram_bot.config import settings
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import asyncio
from model_training.api_model import get_output_to_model

bot = Bot(token=settings.TOKEN)
dp = Dispatcher()
types = []

language = ''


def check_lang():
    async def custom_filter(message: Message):
        return language != ''

    return custom_filter


@dp.message(CommandStart())
async def send_welcome(message: Message):
    global types
    await message.answer('Hello, choose language.', reply_markup=kb.language)


@dp.message(Command('set_language'))
async def send_welcome(message: Message):
    await message.answer('Please, choose language', reply_markup=kb.language)


@dp.callback_query(F.data.in_(['ru', 'eu']))
async def lang_set(data: CallbackQuery):
    global language
    language = data.data
    txt = {'ru': 'Язык успешно выбран, теперь напиши сообщения, а я пришлю его характеристику.',
           'eu': 'The language has been successfully selected,'
                 ' now write a message and I will send its characteristics.'}
    await data.message.answer(txt[language])
    return


@dp.message(check_lang())
async def get_sentiment(message: Message):
    global language
    api_txt = get_output_to_model(message.text, language)
    txt = {'ru': f"Предсказанное настроение: {api_txt['sentiment']}\n"
                 f"Вероятности для каждого класса:\n{'\n'.join(api_txt['all_sentiment'])}",
           'eu': f'Predicted mood: {api_txt['sentiment']}\n'
                 f"Probabilities for each class:\n{'\n'.join(api_txt['all_sentiment'])}"}

    await message.answer(txt[language])


@dp.message()
async def get_sentiment(message: Message):
    await message.answer('Please select language, command - /set_language')


if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))
