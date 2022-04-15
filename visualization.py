import glob
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import re
import scattertext as st
from scattertext import produce_scattertext_explorer
import spacy
from nltk import word_tokenize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import matplotlib.cm as cm
import os
from matplotlib import rcParams as rc
# import nltk
# nltk.download('punkt')

nlp = spacy.load('en_core_web_sm')
swords = set(STOPWORDS)
swords.update(['fig', 'figure', 'et', 'al', 'two', 'within', 'figs', 'three', 'table', 'found', 'among', 'using',
               'shown', 'may', 'one', 'see', 'used', 'supported', 'new', 'well', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
               'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'dr',
               'mr', 'org', 'biorxiv', 'preprint', 'peer', 'doi', 'review', 'copyright', 'version', 'http', 'https',
               'author', 'license', 'international', 'certified', 'granted', 'reserved', 'posted', 'funder', 'holder',
               'available', 'perpetuity'])


def generate_cloud(text, fig_file):
    print('Started generation')
    wc = WordCloud(background_color='white', max_words=150, stopwords=swords, collocations=False,
                   normalize_plurals=False, width=1000, height=600)
    wc.generate(text)
    wc.to_file(fig_file)


def cloud_viz(search_str, fig_file):
    text = ''
    for file_name in glob.glob(search_str):
        print(file_name)
        data = pd.read_csv(file_name, header=None)
        data = data.iloc[:, 1].values.tolist()
        for i in range(len(data)):
            if type(data[i]) == str:
                text += re.sub(r'[^\s\w]+', ' ', data[i]).replace('  ', ' ').lower()
    generate_cloud(text, fig_file)


def cloud_viz_pos_neg(files, fig_file, positive):
    text = ''
    for file_name in files:
        print(file_name)
        data = pd.read_csv(file_name, header=None)
        if positive:
            data = data[data[3] == 1]
        else:
            data = data[data[3] == 0]
        data = data.iloc[:, 2].values.tolist()
        for i in range(len(data)):
            if type(data[i]) == str:
                text += re.sub(r'[^\s\w]+', ' ', data[i]).replace('  ', ' ').lower()
    generate_cloud(text, fig_file)


def scatertext_viz_pos_neg(files, file_name, scaler, min_freq):
    data = pd.read_csv(files[0], header=None)
    if len(files) > 1:
        data = pd.concat([data, pd.read_csv(files[1], header=None)])
    texts = data[2].tolist()
    text = ''
    for i in range(len(texts)):
        if type(texts[i]) == str:
            text += re.sub(r'[^\s\w]+', ' ', texts[i]).replace('  ', ' ').lower() + '#'
    text = word_tokenize(text)
    words = []
    for word in text:
        if word not in swords and not word.isnumeric():
            words.append(word)
    text = ' '.join(words)

    data['cat'] = data[3].astype(str)
    data['parsed'] = pd.Series(text.split('#')).apply(nlp)

    print('Generation started')
    corpus = st.CorpusFromParsedDocuments(data, category_col='cat', parsed_col='parsed').build()
    html = produce_scattertext_explorer(corpus,
                                        category='1',
                                        category_name='Positive',
                                        not_category_name='Negative',
                                        width_in_pixels=1400,
                                        minimum_term_frequency=min_freq,
                                        transform=scaler)
    open(file_name, 'w').write(html)


def years_pie(files, fig_name, positive, start, end, title, fs=9):
    data = pd.concat([pd.read_csv(files[0], header=None), pd.read_csv(files[1], header=None)])
    if positive:
        data = data[data[3] == 1]
    else:
        data = data[data[3] == 0]
    data[1] = data[1].astype(str)
    data = data[data[1].str.len() <= 5]
    data[1] = data[1].astype(int)
    data = data.groupby([1]).agg({0: 'count'})
    num_examples = sum(data[0].tolist())
    data.sort_index(inplace=True)
    data['labels'] = data.index.astype(str)
    data.reset_index(drop=True, inplace=True)
    earlier = sum(data.iloc[:start, 0].tolist())
    data = data.iloc[start:end, :]
    data.loc[end] = [earlier, 'earlier']
    plt.pie(data[0], labels=data['labels'], autopct='%0.f%%', startangle=90, textprops={'fontsize': fs})
    plt.title(title + f'(total {num_examples} examples)')
    plt.savefig(fig_name, dpi=300)
    plt.clf()


def tsne(model_file, fig_file, title):
    model = Doc2Vec.load(model_file)
    keys = ['tree', 'species', 'beast', 'mega', 'mrbayes', 'r8s']

    for word in keys:
        arr = np.empty((0, 100), dtype='f')
        word_labels = [word]

        # get close words
        close_words = model.wv.most_similar(word, 20)

        # add the vector for each of the closest words to the array
        arr = np.append(arr, np.array([model[word]]), axis=0)
        for wrd_score in close_words:
            wrd_vector = model[wrd_score[0]]
            word_labels.append(wrd_score[0])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)

        # find tsne coords for 2 dimensions
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)

        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        # display scatter plot
        plt.scatter(x_coords, y_coords)

        rc.update({'font.size': 22})
        for label, x, y in zip(word_labels, x_coords, y_coords):
            plt.annotate(label, xy=(x, y), xytext=(-60, 5), weight='bold', textcoords='offset points')
        plt.xlim(x_coords.min() + 0.00005, x_coords.max() - 0.00005)
        plt.ylim(y_coords.min() + 0.00005, y_coords.max() - 0.00005)
        # plt.title(title)
        plt.savefig(f'tsne_{word}{fig_file}.png', dpi=300)
        plt.clf()


def tsne_clusters(model_file, title, fig, keys):
    model = Doc2Vec.load(model_file)
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(keys)))

    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word, _ in model.wv.most_similar(word, topn=50):
            words.append(similar_word)
            embeddings.append(model[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    for i in range(len(keys)):
        x = embeddings_en_2d[i, :, 0]
        y = embeddings_en_2d[i, :, 1]
        plt.scatter(x, y, c=colors[i], alpha=0.75, label=keys[i])
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    keys = '_'.join(keys)
    plt.savefig(f'tsne_{fig}{keys}.png', dpi=300)


os.chdir('data')
# cloud_viz('fulltext_corpus/textcrawledData.csv', 'cloud_crawled.png')
# cloud_viz('figure_corpus/figurescrawledData.csv', 'cloud_fig_crawled.png')
# cloud_viz('fulltext_corpus/textcrawledDataRound2.csv', 'cloud_crawledRound2.png')
# cloud_viz('figure_corpus/figurescrawledDataRound2.csv', 'cloud_fig_crawledRound2.png')
# cloud_viz('fulltext_corpus/textiGEMsPDFs.csv', 'cloud_igem.png')
# cloud_viz('figure_corpus/figuresiGEMsPDFs.csv', 'cloud_fig_igem.png')
# cloud_viz('fulltext_corpus/textPMC.csv', 'cloud_PMC.png')
# cloud_viz('figure_corpus/figuresPMC.csv', 'cloud_fig_PMC.png')
# cloud_viz('fulltext_corpus/textPMCHistorical.csv', 'cloud_PMCHistorical.png')
# cloud_viz('figure_corpus/figuresPMCHistorical.csv', 'cloud_fig_PMCHistorical.png')
# cloud_viz('fulltext_corpus/textBiorxivTDM.csv', 'cloud_biorxivTDM.png')
# cloud_viz('figure_corpus/figuresBiorxivTDM.csv', 'cloud_fig_biorxivTDM.png')

# cloud_viz_pos_neg(['train.csv', 'test.csv'], 'cloud_pos.png', True)
# cloud_viz_pos_neg(['train.csv', 'test.csv'], 'cloud_neg.png', False)
# cloud_viz_pos_neg(['fig_train.csv', 'fig_test.csv'], 'cloud_fig_pos.png', True)
# cloud_viz_pos_neg(['fig_train.csv', 'fig_test.csv'], 'cloud_fig_neg.png', False)
# scatertext_viz_pos_neg(['test.csv'], 'scatter.html', st.Scalers.dense_rank, 200)
# scatertext_viz_pos_neg(['test.csv'], 'scatter_stand.html', st.Scalers.log_scale_standardize, 200)
# scatertext_viz_pos_neg(['test.csv'], 'scatter_perc.html', st.Scalers.percentile, 200)
# scatertext_viz_pos_neg(['fig_train.csv', 'fig_test.csv'], 'scatter_fig.html', st.Scalers.dense_rank, 20)
# scatertext_viz_pos_neg(['fig_train.csv', 'fig_test.csv'], 'scatter_stand_fig.html', st.Scalers.log_scale_standardize, 20)
# scatertext_viz_pos_neg(['fig_train.csv', 'fig_test.csv'], 'scatter_perc_fig.html', st.Scalers.percentile, 20)
# years_pie(['train.csv', 'test.csv'], 'pie_pos.png', True, 14, 26, 'Distribution of labeled papers with time-tree\n')
# years_pie(['train.csv', 'test.csv'], 'pie_neg.png', False, 32, 44, 'Distribution of labeled papers without time-tree\n')
# years_pie(['fig_train.csv', 'fig_test.csv'], 'pie_fig_pos.png', True, 11, 23, 'Distribution of labeled figures with time-tree\n')
# years_pie(['fig_train.csv', 'fig_test.csv'], 'pie_fig_neg.png', False, 19, 31, 'Distribution of labeled figures without time-tree\n', 7)
tsne('doc2vec.pc', '', 'Similar words from whole papers(doc2vec)')
tsne('fig_doc2vec.pc', '_fig', 'Similar words from figures descriptions (doc2vec)')
# keys = ['beast', 'mega', 'mrbayes', 'r8s']
# tsne_clusters('doc2vec.pc', 'Doc2vec word clusters (papers)', '', keys)
# tsne_clusters('fig_doc2vec.pc', 'Doc2vec word clusters (figure descriptions)', 'fig_', keys)
