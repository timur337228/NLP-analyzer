from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

language = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text='Русский🇷🇺', callback_data='ru'),
            InlineKeyboardButton(text='Английский🇬🇧', callback_data='eu'),
        ]
    ],
    resize_keyboard=True,
    one_time_keyboard=False,
)
