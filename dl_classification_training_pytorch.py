from datasets import *
import pandas as pd
# from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, DataCollatorWithPadding, \
    get_scheduler
from accelerate import Accelerator
import numpy as np
# from sklearn.metrics import f1_score, accuracy_score
import pickle
import os
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import glob
RESULTS_FILENAME = "bert_predictions_doc_level.txt"


# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     f1 = f1_score(labels, predictions)
#     acc = accuracy_score(labels, predictions)
#     return {'f1': f1, 'acc': acc}


# def modify_datasets(filename, split=False):
#     df = pd.read_csv(filename, header=None)
#     df = df.loc[:, 2:3]
#     df.columns = ["text", "label"]
#     if split:
#         X_train, X_cv, y_train, y_cv = train_test_split(df["text"], df["label"], test_size=1000, random_state=42)
#         df1 = pd.DataFrame({"text": X_train, "label": y_train})
#         df2 = pd.DataFrame({"text": X_cv, "label": y_cv})
#         df1.to_csv("data_" + filename, index=False)
#         df2.to_csv("data_" + filename.replace("train", "cv"), index=False)
#     df.to_csv("data_" + filename, index=False)


def train_model(dataset, model, tokenizer, dataset_name, model_name):
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset = dataset.with_format("torch")
    epochs = 2
    lr = 2e-5
    batch_size = 8

    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(dataset["validation"], batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(dataset["test"], batch_size=batch_size, collate_fn=data_collator)
    logging_steps = len(train_dataloader)

    training_steps = epochs * logging_steps
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=training_steps)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    accelerator = Accelerator()
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    progress_bar = tqdm(range(training_steps))
    metric = load_metric("glue", "mrpc")

    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predicitons=accelerator.gather(predictions),
                             references=accelerator.gather(batch["labels"]))
        print(f"validation predictions {dataset_name} {model_name} {epoch} ", metric.compute())

    model.eval()
    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        with open(f"predictions_{dataset_name}_{model_name}.pc", "ab") as output_file:
            pickle.dump(predictions, output_file)
        metric.add_batch(predicitons=accelerator.gather(predictions),
                         references=accelerator.gather(batch["labels"]))
    print(f"test predictions {dataset_name} {model_name} ", metric.compute())


def predict_unlabeled(model, tokenizer, search_str, prefix):
    def tokenize_function(example):
        return tokenizer(example['1'], padding="max_length", truncation=True, max_length=512)

    for file_name in glob.glob(search_str):
        print(file_name)
        dataset = load_dataset("csv", data_files={
            "test": file_name}, sep=",", header=None)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset = dataset.remove_columns(['0', '1', '2'])
        dataset = dataset.with_format("torch")
        batch_size = 500

        data_collator = DataCollatorWithPadding(tokenizer)
        test_dataloader = DataLoader(dataset["test"], batch_size=batch_size, collate_fn=data_collator)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                print(outputs.loss, outputs.logits.shape)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1)
                probabilities = torch.softmax(logits, dim=-1).cpu().detach().numpy()
                prob_y = [y for [_, y] in probabilities]
                results = pd.DataFrame({"pred_y": prediction, "prob_y": prob_y})
                results.to_csv(prefix + file_name.replace('figure_corpus/', ''), mode='a', index=False, header=False)


def predicted_positive_by_year_from_fig(search_str, prefix):
    for file_name in glob.glob(search_str):
        print(file_name)
        dataset = pd.read_csv(file_name, header=None)
        results = pd.read_csv(prefix + file_name.replace('figure_corpus/', ''), header=None)
        dataset['pred_y'] = results[0]
        dataset['prob_y'] = results[1]
        data = dataset.groupby([0, 2], as_index=False).agg({'pred_y': 'mean', 'prob_y': 'mean', 1: 'count'})
        data['fig_pred'] = (data['pred_y'] > 0.5) & (data['prob_y'] > 0.5)
        data['take'] = (data['pred_y'] > 0.5) & (data[1] == 1)
        data_doc = data.groupby([0], as_index=False).agg({'fig_pred': 'max', 'prob_y': 'mean', 'take': 'max'})
        data_doc['doc_pred'] = (data_doc['fig_pred'] is True) | (data_doc['prob_y'] > 0.5) | (
                data_doc['take'] is True)
        data_doc['take'] = data_doc['take'].astype(int)
        data_doc['doc_pred'] = data_doc['doc_pred'].astype(int)
        data_doc = data_doc.loc[:, [0, 'prob_y', 'doc_pred']]
        # data_doc.to_csv(prefix + file_name.replace('figure_corpus/', '').replace('figures', 'text'), header=None,
        #                 index=None)
        data_doc_positive = data_doc[data_doc['doc_pred'] == 1]
        data_doc_positive.to_csv("positive_" + prefix + file_name.replace('figure_corpus/', '').replace('figures', 'text'),
                        header=None, index=None)
        print(f'Percent of positive labels group by research paper in BERT is:' +
              f'{sum(data_doc["doc_pred"]) / data_doc.shape[0]} and number of documents is {data_doc.shape[0]}', )


os.chdir('data')

# modify_datasets("train.csv", split=True)
# modify_datasets("test.csv")
# modify_datasets("fig_train.csv", split=True)
# modify_datasets("fig_test.csv")
# modify_datasets("small_fig_train.csv", split=True)
# modify_datasets("small_fig_test.csv")
#
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
# dataset3 = load_dataset("csv", data_files={
#     "train": "data_small_fig_train.csv",
#     "validation": "data_small_fig_cv.csv",
#     "test": "data_small_fig_test.csv"
# }, sep=",")


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

# tokenizer = AutoTokenizer.from_pretrained("../BERT_phylogenetics")
# model = AutoModelForSequenceClassification.from_pretrained("../BERT_phylogenetics")
# predict_unlabeled(model, tokenizer, "figure_corpus/figuresiGEMsPDFs.csv", "labeled_")
predicted_positive_by_year_from_fig("figure_corpus/*", "labeled_")
