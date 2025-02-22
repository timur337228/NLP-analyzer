import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from torch.cuda.amp import (GradScaler,
                            autocast)
from model_training.func_for_model import load_data, main_path, TextDataset, path_to_model, train_epoch, eval_model


def learn_model():
    texts_eu, labels_eu = load_data(f'{main_path}/eu_train.txt')
    texts_ru, labels_ru = load_data(f'{main_path}/ru_train.txt')
    texts = texts_eu + texts_ru
    labels = labels_ru + labels_eu

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    test_texts_eu, test_labels_eu = load_data(f'{main_path}/eu_test.txt')
    test_texts_ru, test_labels_ru = load_data(f'{main_path}/ru_test.txt')
    test_texts = test_texts_eu + test_texts_ru
    test_labels = label_encoder.transform(test_labels_ru + test_labels_eu)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    max_len = 128
    train_dataset = TextDataset(texts, labels, tokenizer, max_len)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss().to('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f'Train loss {train_loss} accuracy {train_acc}')

    test_acc, test_loss = eval_model(model, test_loader, loss_fn, device)
    print(f'Test loss {test_loss} accuracy {test_acc}')

    model.save_pretrained(path_to_model)
    tokenizer.save_pretrained(path_to_model)
    joblib.dump(label_encoder, f'{path_to_model}/label_encoder.pkl')


if __name__ == '__main__':
    learn_model()
