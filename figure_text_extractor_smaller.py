import pandas as pd
import re
import glob
from utils import clean_ids, normalize_text_in_column
import os


# Covers supplement as well as long as it contains time tree figures that are mentioned in main text
def extract_figures_text():
    for file in glob.glob('fulltext_corpus/text*.csv'):
        print(file)
        texts = pd.read_csv(file, header=None)
        for index, row in texts.iterrows():
            figures = []
            figures_no = re.findall(
                r"((\n.*){0,2}(([Ff][Ii][Gg](|\.|[Uu][Rr][Ee]|[Ss]\.)|[Ff] [Ii] [Gg] [Uu] [Rr] [Ee])"
                + r"(\s|\\n)*((\s|\\n)\w|\w?\d+[-\\.]*\d*\w?)(\(|\)|,|\\'|\\.|\s|\\n))(.*\n){0,2})", row[1],
                re.MULTILINE)
            for figure in figures_no:
                fig_id = re.findall(r"(([Ff][Ii][Gg](|\.|[Uu][Rr][Ee]|[Ss]\.)|[Ff] [Ii] [Gg] [Uu] [Rr] [Ee])(\s|\\n)*" +
                                    r"((\s|\\n)\w|\w?\d+[-\\.]*\d*\w?)(\(|\)|,|\\'|\\.|\s|\\n))", figure[0],
                                    re.MULTILINE)
                for id in fig_id:
                    id = re.sub(r"([Ff] [Ii] [Gg] [Uu] [Rr] [Ee]|[Ff][Ii][Gg](\.|[Uu][Rr][Ee]|[Ss]\.))(\s|\\n)*", "",
                                id[0])
                    id = re.sub(r"[Ff][Ii][Gg]|\(|\)|\\'|,|\s|\\n", "", id)
                    if id[-1] == '.' or id[-1] == '-':
                        id = id[:-1]
                    if id != '':
                        k = len(figure[0])
                        if k > 400:
                            k = k - 400
                            fig = figure[0][k // 2:-k // 2]
                        else:
                            fig = figure[0]
                        if len(fig) > 0:
                            figures.append([row[0], id, fig])
            if len(figures) == 0:
                print(index, row[0])
            else:
                figures = pd.DataFrame(figures)[[0, 2, 1]]
                figures = figures.drop_duplicates()
                figures.to_csv(file.replace('fulltext_corpus/text', 'small_figure_corpus/figures'), mode='a', index=False,
                               header=False)


def training_test_figures_texts(labeled_file, crawler_file, igem_file):
    df = clean_ids(pd.read_csv(labeled_file))
    data = clean_ids(pd.read_csv(igem_file, header=None)[[0, 1, 2]], False)
    data2 = pd.read_csv(crawler_file, header=None)
    data = pd.concat([data, data2])
    data = data.merge(df, how='right', left_on=0, right_on='ID')
    data_pos = data[data['Time_F'] != '0']
    data_pos = data_pos[data_pos['Time_F'] == data_pos[2]]
    data_neg = data[data['Time_F'] == '0']
    data_neg = data_neg[data_neg['Binary'] == 0].sample(frac=1)
    data = pd.concat([data_pos, data_neg[0:data_pos.shape[0]]])[['ID', 'PY', 1, 'Binary', 2]]
    data = data.sample(frac=1)
    data = data.loc[data[1].notnull()]
    data = normalize_text_in_column(data, 1)
    df_test = data[0:int(0.2 * data.shape[0])]
    df_training = data[int(0.2 * data.shape[0]):]
    df_test.to_csv('small_fig_test.csv', index=False, header=False)
    df_training.to_csv('small_fig_train.csv', index=False, header=False)


os.chdir('data')
# extract_figures_text()
# training_test_figures_texts('figure_labeled_examples.csv', 'small_figure_corpus/figurescrawledData.csv',
#                             'small_figure_corpus/figuresiGEMsPDFs.csv')
