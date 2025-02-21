import asyncio
from googletrans import Translator

# emotion_dict = {
#     "love": "любовь",
#     "sadness": "печаль",
#     "joy": "радость",
#     "fear": "страх",
#     "anger": "гнев",
#     "surprise": "удивление"
# }


async def translate_text(text, translator):
    translation = await translator.translate(text, src='en', dest='ru')
    return translation.text


def translate_emotion(emotion, emotion_dict):
    return emotion_dict.get(emotion, emotion)  # Если эмоция не найдена, возвращаем оригинал


async def main():
    translator = Translator()

    with open('eu_train.txt', 'r', encoding='utf-8') as infile, open('ru_train.txt', 'w', encoding='utf-8') as outfile:
        for line in infile:
            text, emotion = line.strip().split(';')

            translated_text = await translate_text(text, translator)

            outfile.write(f"{translated_text};{emotion}\n")
    print('Все гуд!')


if __name__ == "__main__":
    asyncio.run(main())