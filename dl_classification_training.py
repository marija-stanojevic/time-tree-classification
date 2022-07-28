from datasets import *
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pickle
import os


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions)
    acc = accuracy_score(labels, predictions)
    return {'f1': f1, 'acc': acc}


def modify_datasets(filename, split=False):
    df = pd.read_csv(filename, header=None)
    df = df.loc[:, 2:3]
    df.columns = ["text", "label"]
    if split:
        X_train, X_cv, y_train, y_cv = train_test_split(df["text"], df["label"], test_size=1000, random_state=42)
        df1 = pd.DataFrame({"text": X_train, "label": y_train})
        df2 = pd.DataFrame({"text": X_cv, "label": y_cv})
        df1.to_csv("data_" + filename, index=False)
        df2.to_csv("data_" + filename.replace("train", "cv"), index=False)
    df.to_csv("data_" + filename, index=False)


def train_model(dataset, model, tokenizer, dataset_name, model_name):
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset = dataset.with_format("torch")
    logging_steps = 307
    epochs = 2
    lr = 2e-5
    batch_size = 8

    data_collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments("classifier", per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      num_train_epochs=epochs, learning_rate=lr, weight_decay=0.01,
                                      logging_steps=logging_steps, evaluation_strategy="epoch")

    trainer = Trainer(model, training_args, train_dataset=dataset["train"],
                      eval_dataset=dataset["validation"], data_collator=data_collator, tokenizer=tokenizer,
                      compute_metrics=compute_metrics)
    trainer.train()
    for i in range(epochs):
        chck = logging_steps * i
        model = AutoModelForSequenceClassification.from_pretrained(f"classifier/checkpoint-{chck}")
        predictions = model.predict(dataset["test"])
        with open(f"predictions_{dataset_name}_{model_name}_{chck}.pc", "wb") as output_file:
            pickle.dump(predictions, output_file)


os.chdir('data')

# modify_datasets("train.csv", split=True)
# modify_datasets("test.csv")
# modify_datasets("fig_train.csv", split=True)
# modify_datasets("fig_test.csv")
# modify_datasets("small_fig_train.csv", split=True)
# modify_datasets("small_fig_test.csv")

# dataset1 = load_dataset("csv", data_files={
#     "train": "data_train.csv",
#     "validation": "data_cv.csv",
#     "test": "data_test.csv"}, sep=",")
#
# dataset2 = load_dataset("csv", data_files={
#     "train": "data_fig_train.csv",
#     "validation": "data_fig_cv.csv",
#     "test": "data_fig_test.csv"
# }, sep=",")
#
dataset3 = load_dataset("csv", data_files={
    "train": "data_small_fig_train.csv",
    "validation": "data_small_fig_cv.csv",
    "test": "data_small_fig_test.csv"
}, sep=",")


# tokenizer1 = AutoTokenizer.from_pretrained("bert-base-uncased")
# model1 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# train_model(dataset1, model1, tokenizer1, "full", "bert")
# train_model(dataset2, model1, tokenizer1, "figures", "bert")
# train_model(dataset3, model1, tokenizer1, "small_figures", "bert")
#
# tokenizer2 = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
# model2 = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased")
# train_model(dataset1, model2, tokenizer2, "full", "scibert")
# train_model(dataset2, model2, tokenizer2, "figures", "scibert")
# train_model(dataset3, model2, tokenizer2, "small_figures", "scibert")
#
# tokenizer3 = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
# model3 = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
# train_model(dataset1, model3, tokenizer3, "full", "biobert")
# train_model(dataset2, model3, tokenizer3, "figures", "biobert")
# train_model(dataset3, model3, tokenizer3, "small_figures", "biobert")
