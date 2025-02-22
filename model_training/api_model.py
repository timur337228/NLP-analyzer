import torch
import joblib
from model_training.func_for_model import path_to_model, load_model, predict_sentiment


emotion_dict = {
    "love": "любовь",
    "sadness": "печаль",
    "joy": "радость",
    "fear": "страх",
    "anger": "гнев",
    "surprise": "удивление"
}


def get_output_to_model(text, lang):
    message = {}
    model, tokenizer = load_model(path_to_model)

    label_encoder = joblib.load(f'{path_to_model}/label_encoder.pkl')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    sentiment, probabilities = predict_sentiment(text, model, tokenizer, label_encoder, device)
    message['all_sentiment'] = []
    for emotion, prob in zip(label_encoder.classes_, probabilities):
        message['all_sentiment'].append(f"{emotion_dict[emotion] if lang == 'ru' else emotion}: {prob:.4f}")
    if lang == 'ru':
        sentiment = emotion_dict[sentiment]
    message['sentiment'] = sentiment
    return message

