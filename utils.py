import re
import pandas as pd
from gensim.models.callbacks import CallbackAny2Vec


def clean_ids(df, columns=True):
    if not columns:
        df.columns = ['ID', 'text', 'downloaded']
    for index, row in df.iterrows():
        df.at[index, 'ID'] = re.sub(r'[^\s\w]+', ' ', row['ID']).replace('  ', ' ').lower()
    if not columns:
        df.columns = [0, 1, 2]
    return df


def normalize_text_in_column(data, col):
    data[col] = ["\"" + re.sub(r'[^\w\s.(),:;{}_><=\"\'\[\]\-/]+', '', str(x)).replace('  ', ' ').lower() + "\"" for x
                 in data[col]]
    return data


def normalize_text(data):
    data = ["\"" + re.sub(r'[^\w\s.(),:;{}_><=\"\'\[\]\-/]+', '', str(x)).replace('  ', ' ').lower() + "\"" for x
            in data]
    return data


def preprocess_text(filename, col=2):
    data = pd.read_csv(filename, header=None)
    data = data.iloc[:, col].values.tolist()
    data = normalize_text(data)
    return data


# species that are found in less than 2 studies
def species_load_1(filename):
    species = pd.read_csv(filename)
    return species[species['count'] <= 1].dropna()


# species that are found in less than 11 studies
def species_load_10(filename):
    species = pd.read_csv(filename)
    return species[species['count'] <= 10].dropna()


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1
