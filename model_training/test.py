import torch
import joblib
from model_training.func_for_model import path_to_model, load_model, predict_sentiment


def test_model():
    model, tokenizer = load_model(path_to_model)

    label_encoder = joblib.load(f'{path_to_model}/label_encoder.pkl')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    text = input("Введите текст для анализа настроения: ")

    sentiment, probabilities = predict_sentiment(text, model, tokenizer, label_encoder, device)

    print(f"Предсказанное настроение: {sentiment}")
    print("Вероятности для каждого класса:")
    for emotion, prob in zip(label_encoder.classes_, probabilities):
        print(f"{emotion}: {prob:.4f}")


if __name__ == "__main__":
    test_model()
