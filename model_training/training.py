import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from model_training.func_for_model import load_data, main_path, TextDataset, path_to_model, train_epoch, eval_model


def learn_model():

    texts, labels = load_data(f'{main_path}/train.txt')

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    max_len = 128
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss().to('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, val_loader, loss_fn, device)
        print(f'Val loss {val_loss} accuracy {val_acc}')
    model.save_pretrained(path_to_model)
    tokenizer.save_pretrained(path_to_model)

    joblib.dump(label_encoder, f'{path_to_model}/label_encoder.pkl')