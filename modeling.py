from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle as pc
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils import preproces_text, EpochLogger
import glob
import os
import random


# To download stopwords in first usage
# import nltk
# nltk.download('stopwords')


def prepare_tfidf_features(filename, vectorizer, fig='', max_range=2):
    data = preproces_text(filename)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(min_df=0.05, max_df=0.9, ngram_range=(1, max_range))
        X = vectorizer.fit_transform(data)
        with open(fig + 'vectorizer.csv', 'wb') as file:
            pc.dump(vectorizer, file)
        print('Features number: ', str(len(vectorizer.get_feature_names())))
    else:
        X = vectorizer.transform(data)
    with open(fig + 'features_' + filename.replace('csv', 'pc'), 'wb') as file:
        pc.dump(X, file)
    return vectorizer


def prepare_doc2vec_features(filename, vectorizer, fig=''):
    epoch_logger = EpochLogger()
    data = preproces_text(filename)
    final_data = []
    if vectorizer is None:
        ind = 0
        for row in data:
            toks = tokenizer.tokenize(row)
            final_data.append(TaggedDocument(toks, [ind]))
            ind += 1
        print('Training starts')
        vectorizer = Doc2Vec(workers=8, vector_size=100, callbacks=[epoch_logger])
        vectorizer.build_vocab(final_data)
        vectorizer.train(final_data, total_examples=vectorizer.corpus_count, epochs=20)
        vectorizer.save(fig + 'doc2vec.pc')
        print('Training finishes')
    X = []
    for i in range(len(data)):
        X.append(vectorizer.infer_vector(tokenizer.tokenize((data[i]))))
    with open(fig + 'doc2vec_features_' + filename.replace('csv', 'pc'), 'wb') as file:
        pc.dump(X, file)
    return vectorizer


def create_corpus(search_str, corpus_file):
    for file_name in glob.glob(search_str):
        print(file_name)
        data = pd.Series(preproces_text(file_name, 1))
        data.to_csv(corpus_file, header=False, index=False, mode='a')
        print('Finished corpus creation')


def preprocessed_tfidf_features(filename, vectorizer, fig='', max_range=2, corpus_file='final_data.csv'):
    data = preproces_text(filename)
    if vectorizer is None:
        all_data = pd.read_csv(corpus_file, header=None)[0].values.tolist()
        vectorizer = TfidfVectorizer(min_df=0.05, max_df=0.9, ngram_range=(1, max_range))
        vectorizer.fit(all_data)
        with open(fig + 'preprocessed_vectorizer.csv', 'wb') as file:
            pc.dump(vectorizer, file)
        print('Features number: ', str(len(vectorizer.get_feature_names())))
    X = vectorizer.transform(data)
    with open(fig + 'preprocessed_features_' + filename.replace('csv', 'pc'), 'wb') as file:
        pc.dump(X, file)
    return vectorizer


def read_corpus(corpus_file):
    ind = 0
    for chunk in pd.read_csv(corpus_file, header=None, chunksize=1):
        toks = tokenizer.tokenize(chunk.loc[ind, 0])
        el = TaggedDocument(toks, [ind])
        ind += 1
        yield el


def preprocessed_doc2vec_features(filename, vectorizer, fig='', corpus_file='final_data.csv'):
    epoch_logger = EpochLogger()
    if vectorizer is None:
        print('Training starts')
        vectorizer = Doc2Vec(workers=8, vector_size=100, callbacks=[epoch_logger])
        vectorizer.build_vocab(list(read_corpus(corpus_file)))
        vectorizer.train(list(read_corpus(corpus_file)), total_examples=vectorizer.corpus_count, epochs=20)
        vectorizer.save(fig + 'preprocessed_doc2vec.pc')
        print('Training finishes')
    X = []
    data = preproces_text(filename)
    for i in range(len(data)):
        X.append(vectorizer.infer_vector(tokenizer.tokenize(data[i])))
    with open(fig + 'preprocessed_doc2vec_features_' + filename.replace('csv', 'pc'), 'wb') as file:
        pc.dump(X, file)
    return vectorizer


def split_training_cv(train_features_file, train_file):
    with open(train_features_file, 'rb') as file:
        X = pc.load(file)
    y = pd.read_csv(train_file, header=None).iloc[:, 3].values.tolist()
    cv_inds = random.sample(range(0, len(y)), 1000)
    train_inds = list(set(range(0, len(y))) - set(cv_inds))
    if 'doc2vec' in train_features_file:  # array of arrays
        X_cv = list(map(X.__getitem__, cv_inds))
        X = list(map(X.__getitem__, train_inds))
    else:  # sparse matrix
        X_cv = X[cv_inds]
        X = X[train_inds]
    y_cv = list(map(y.__getitem__, cv_inds))
    y = list(map(y.__getitem__, train_inds))
    return X, y, X_cv, y_cv


def train_lr(train_features_file, train_file):
    X, y, X_cv, y_cv = split_training_cv(train_features_file, train_file)
    model = LogisticRegression()
    model.fit(X, y)
    pred_y = model.predict(X_cv)
    f1 = f1_score(y_cv, pred_y)
    return model, f1


def train_l1(train_features_file, train_file, alpha):
    X, y, X_cv, y_cv = split_training_cv(train_features_file, train_file)
    model = SGDClassifier(loss='log', penalty='l1', alpha=alpha, l1_ratio=1, class_weight='balanced', max_iter=500)
    model.fit(X, y)
    weights = model.coef_[0]
    counter = 0
    good_weights = []
    for w in weights:
        if w != 0:
            good_weights.append(counter)
        counter += 1
    print('Non-zero weights:', str(len(good_weights)))
    pred_y = model.predict(X_cv)
    f1 = f1_score(y_cv, pred_y)
    return model, f1


def train_decision_tree(train_features_file, train_file, impurity_decrease):
    X, y, X_cv, y_cv = split_training_cv(train_features_file, train_file)
    model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=0,
                                   min_impurity_decrease=impurity_decrease)
    model.fit(X, y)
    print("Model depth: ", model.get_depth())
    pred_y = model.predict(X_cv)
    f1 = f1_score(y_cv, pred_y)
    return model, f1


def train_random_forest(train_features_file, train_file, impurity_decrease):
    X, y, X_cv, y_cv = split_training_cv(train_features_file, train_file)
    model = RandomForestClassifier(criterion='entropy', class_weight='balanced_subsample', random_state=0,
                                   min_impurity_decrease=impurity_decrease)
    model.fit(X, y)
    pred_y = model.predict(X_cv)
    f1 = f1_score(y_cv, pred_y)
    return model, f1


def train_knn(train_features_file, train_file, nn):
    X, y, X_cv, y_cv = split_training_cv(train_features_file, train_file)
    model = KNeighborsClassifier(n_neighbors=nn)
    model.fit(X, y)
    pred_y = model.predict(X_cv)
    f1 = f1_score(y_cv, pred_y)
    return model, f1


def train_svm(train_features_file, train_file, C, kernel):
    X, y, X_cv, y_cv = split_training_cv(train_features_file, train_file)
    model = SVC(class_weight='balanced', random_state=0, C=C, kernel=kernel, probability=True)
    model.fit(X, y)
    pred_y = model.predict(X_cv)
    f1 = f1_score(y_cv, pred_y)
    return model, f1


def train_adaboost(train_features_file, train_file, n_estimators):
    X, y, X_cv, y_cv = split_training_cv(train_features_file, train_file)
    model = AdaBoostClassifier(random_state=0, n_estimators=n_estimators)
    model.fit(X, y)
    pred_y = model.predict(X_cv)
    f1 = f1_score(y_cv, pred_y)
    return model, f1


def train_gb(train_features_file, train_file, n_estimators):
    X, y, X_cv, y_cv = split_training_cv(train_features_file, train_file)
    model = GradientBoostingClassifier(random_state=0, n_estimators=n_estimators)
    model.fit(X, y)
    pred_y = model.predict(X_cv)
    f1 = f1_score(y_cv, pred_y)
    return model, f1


def train_bagging(train_features_file, train_file, n_estimators):
    X, y, X_cv, y_cv = split_training_cv(train_features_file, train_file)
    model = BaggingClassifier(random_state=0, n_estimators=n_estimators)
    model.fit(X, y)
    pred_y = model.predict(X_cv)
    f1 = f1_score(y_cv, pred_y)
    return model, f1


def predict_prob(model, test_features_file, filename, model_name):
    df = pd.read_csv(filename, header=None)
    with open(test_features_file, 'rb') as file:
        X = pc.load(file)
    y = df[3].values.tolist()
    pred_y = model.predict(X)
    df['pred_y'] = pred_y
    df['prob_y'] = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred_y).ravel()
    print("Accuracy: ", str((tp + tn) / (tp + tn + fp + fn)))
    print("Sensitivity: ", str(tp / (tp + fn)))
    print('Specificity: ', str(tn / (tn + fp)))
    print('F1-score: ', str(f1_score(y, pred_y)))
    df.to_csv(model_name + test_features_file.replace('.pc', '') + '_predicted_proba.csv', index=False, header=False)
    with open(model_name + test_features_file.replace('test', 'model'), 'wb') as file:
        pc.dump(model, file)


# using figures level prediction figure out if document has a figure with a time tree or not
def doc_level_pred(model_file, test_features_file, filename):
    print(model_file)
    with open(model_file, 'rb') as file:
        model = pc.load(file)
    df = pd.read_csv(filename, header=None)
    with open(test_features_file, 'rb') as file:
        X = pc.load(file)
    pred_y = model.predict(X)
    df['pred_y'] = pred_y
    df['prob_y'] = model.predict_proba(X)[:, 1]
    data = df.groupby([0, 4]).agg({'pred_y': 'mean', 'prob_y': 'mean', 3: 'mean', 2: 'count'})
    data['fig_pred'] = (data['pred_y'] > 0.5) & (data['prob_y'] > 0.5)
    data['take'] = (data['pred_y'] > 0.5) & (data[2] == 1)
    tn, fp, fn, tp = confusion_matrix(data[3], data['fig_pred']).ravel()
    print("Fig Accuracy: ", str((tp + tn) / (tp + tn + fp + fn)))
    print("Fig Sensitivity: ", str(tp / (tp + fn)))
    print('Fig Specificity: ', str(tn / (tn + fp)))
    print('Fig F1-score: ', str(f1_score(data[3], data['fig_pred'])))

    data_doc = data.groupby([0]).agg({'fig_pred': 'max', 3: 'mean', 'prob_y': 'mean', 'take': 'max'})
    data_doc['doc_pred'] = (data_doc['fig_pred'] is True) | (data_doc['prob_y'] > 0.5) | (data_doc['take'] is True)
    tn, fp, fn, tp = confusion_matrix(data_doc[3], data_doc['doc_pred']).ravel()
    print("Doc Accuracy: ", str((tp + tn) / (tp + tn + fp + fn)))
    print("Doc Sensitivity: ", str(tp / (tp + fn)))
    print('Doc Specificity: ', str(tn / (tn + fp)))
    print('Doc F1-score: ', str(f1_score(data_doc[3], data_doc['doc_pred'])))


def predict_unlabeled(vectorizer_file, model_file, tfidf, search_str, prefix, fig=False):
    with open(model_file, 'rb') as file:
        model = pc.load(file)
    for file_name in glob.glob(search_str):
        if 'PMC.' not in file_name:
            continue
        print(file_name)
        dataset = pd.read_csv(file_name, header=None).drop_duplicates().sample(frac=0.01)
        data = preproces_text(file_name, 1)
        if tfidf:
            with open(vectorizer_file, 'rb') as tfidf_file:
                vectorizer = pc.load(tfidf_file)
            X = vectorizer.transform(data)
        else:
            vectorizer = Doc2Vec.load(vectorizer_file)
            X = []
            for i in range(len(data)):
                X.append(list(vectorizer.infer_vector(tokenizer.tokenize((data[i])))))
        pred_y = model.predict(X)
        dataset['pred_y'] = pred_y
        dataset['prob_y'] = model.predict_proba(X)[:, 1]
        print(f'Percent of positive labels for {model_file} is: {sum(pred_y/len(pred_y))} and size is {len(pred_y)}',)
        dataset.to_csv(prefix + file_name, index=False, header=False)
        # if fig:
        #     data = dataset.groupby([0, 2], as_index=False).agg({'pred_y': 'mean', 'prob_y': 'mean', 1: 'count'})
        #     data['fig_pred'] = (data['pred_y'] > 0.5) & (data['prob_y'] > 0.5)
        #     data['take'] = (data['pred_y'] > 0.5) & (data[1] == 1)
        #     data_doc = data.groupby([0], as_index=False).agg({'fig_pred': 'max', 'prob_y': 'mean', 'take': 'max'})
        #     data_doc['doc_pred'] = (data_doc['fig_pred'] is True) | (data_doc['prob_y'] > 0.5) | (data_doc['take'] is True)
        #     data_doc['take'] = data_doc['take'].astype(int)
        #     data_doc['doc_pred'] = data_doc['doc_pred'].astype(int)
        #     print(
        #         f'Percent of positive labels group by research paper in {model_file} is:' +
        #         f'{sum(data_doc["doc_pred"]) / data_doc.shape[0]} and number of documents is {data_doc.shape[0]}', )
        #
        #     data_doc.to_csv(prefix + file_name.replace('figures', 'doc_agg_figures'), index=False, header=False)


os.chdir('data')

tokenizer = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')
lemma = WordNetLemmatizer()

# create_corpus(search_str='fulltext_corpus/text*', corpus_file='final_data.csv')
# create_corpus(search_str='figure_corpus/figures*.csv', corpus_file='figures_final_data.csv')

# vectorizer = prepare_tfidf_features('train.csv', None)
# vectorizer = prepare_tfidf_features('test.csv', vectorizer)
# vectorizer = prepare_tfidf_features('fig_train.csv', None, 'fig_', max_range=3)
# vectorizer = prepare_tfidf_features('fig_test.csv', vectorizer, 'fig_')
# vectorizer = prepare_doc2vec_features('train.csv', None)
# vectorizer = prepare_doc2vec_features('test.csv', vectorizer)
# vectorizer = prepare_doc2vec_features('fig_train.csv', None, 'fig_')
# vectorizer = prepare_doc2vec_features('fig_test.csv', vectorizer, 'fig_')

# vectorizer = preprocessed_tfidf_features('train.csv', None, corpus_file='final_data.csv')
# vectorizer = preprocessed_tfidf_features('test.csv', vectorizer)
# vectorizer = preprocessed_tfidf_features('fig_train.csv', None, 'fig_', corpus_file='figures_final_data.csv')
# vectorizer = preprocessed_tfidf_features('fig_test.csv', vectorizer, 'fig_')
# vectorizer = preprocessed_doc2vec_features('train.csv', None, corpus_file='final_data.csv')
# vectorizer = preprocessed_doc2vec_features('test.csv', vectorizer)
# vectorizer = preprocessed_doc2vec_features('fig_train.csv', None, 'fig_', corpus_file='figures_final_data.csv')
# vectorizer = preprocessed_doc2vec_features('fig_test.csv', vectorizer, 'fig_')

train_features = ['features_train.pc', 'fig_features_fig_train.pc', 'doc2vec_features_train.pc',
                  'fig_doc2vec_features_fig_train.pc', 'preprocessed_features_train.pc',
                  'fig_preprocessed_features_fig_train.pc', 'preprocessed_doc2vec_features_train.pc',
                  'fig_preprocessed_doc2vec_features_fig_train.pc']
train_data = ['train.csv', 'fig_train.csv']
test_features = ['features_test.pc', 'fig_features_fig_test.pc', 'doc2vec_features_test.pc',
                 'fig_doc2vec_features_fig_test.pc', 'preprocessed_features_test.pc',
                 'fig_preprocessed_features_fig_test.pc', 'preprocessed_doc2vec_features_test.pc',
                 'fig_preprocessed_doc2vec_features_fig_test.pc']
test_data = ['test.csv', 'fig_test.csv']

# baseline
model, f1 = train_lr('doc2vec_features_train.pc', 'train.csv')
predict_prob(model, 'doc2vec_features_test.pc', 'test.csv', 'lr_')

for i in range(4, 5):
    print(train_features[i])
    best_f1 = 0
    best_alpha = 0.0001
    best_model = None
    for alpha in [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]:
        model, f1 = train_l1(train_features[i], train_data[i % 2], alpha)
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_model = model
    print(f'\nL1: Best alpha is {best_alpha} and it\'s cv f1 is {best_f1}\n')
    predict_prob(best_model, test_features[i], test_data[i % 2], 'l1_')

    best_f1 = 0
    best_imp_dec = 0.0001
    best_model = None
    for impurity_decrease in [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
        model, f1 = train_decision_tree(train_features[i], train_data[i % 2], impurity_decrease)
        if f1 > best_f1:
            best_f1 = f1
            best_imp_dec = impurity_decrease
            best_model = model
    print(f'\nDT: Best impurity decrease is {best_imp_dec} and it\'s cv f1 is {best_f1}\n')
    predict_prob(best_model, test_features[i], test_data[i % 2], 'dt_')

    best_f1 = 0
    best_imp_dec = 0.0001
    best_model = None
    for impurity_decrease in [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
        model, f1 = train_random_forest(train_features[i], train_data[i % 2], impurity_decrease)
        if f1 > best_f1:
            best_f1 = f1
            best_imp_dec = impurity_decrease
            best_model = model
    print(f'\nRF: Best impurity decrase is {best_imp_dec} and it\'s cv f1 is {best_f1}\n')
    predict_prob(best_model, test_features[i], test_data[i % 2], 'rf_')

    best_f1 = 0
    best_nn = 1
    best_model = None
    for nn in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]:
        model, f1 = train_knn(train_features[i], train_data[i % 2], nn)
        if f1 > best_f1:
            best_f1 = f1
            best_nn = nn
            best_model = model
    print(f'\nKNN: Best nn is {best_nn} and it\'s cv f1 is {best_f1}\n')
    predict_prob(best_model, test_features[i], test_data[i % 2], 'knn_')

    best_f1 = 0
    best_C = 1
    best_kernel = 'linear'
    best_model = None
    for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            model, f1 = train_svm(train_features[i], train_data[i % 2], C, kernel)
            if f1 > best_f1:
                best_f1 = f1
                best_C = C
                best_kernel = kernel
                best_model = model
            print(f'\nSVM: Best C is {best_C} and best kernel is {best_kernel} and it\'s cv f1 is {best_f1}\n')
            predict_prob(best_model, test_features[i], test_data[i % 2], 'svm_')

    best_f1 = 0
    best_n_est = 5
    best_model = None
    for n_est in [5, 10, 20, 40, 80]:
        model, f1 = train_adaboost(train_features[i], train_data[i % 2], n_est)
        if f1 > best_f1:
            best_f1 = f1
            best_n_est = n_est
            best_model = model
    print(f'\nAdaboost: Best number of estimators is {best_n_est} and it\'s cv f1 is {best_f1}\n')
    predict_prob(best_model, test_features[i], test_data[i % 2], 'ada_')

    best_f1 = 0
    best_n_est = 5
    best_model = None
    for n_est in [5, 10, 20, 40, 80]:
        model, f1 = train_gb(train_features[i], train_data[i % 2], n_est)
        if f1 > best_f1:
            best_f1 = f1
            best_n_est = n_est
            best_model = model
    print(f'\nGB: Best number of estimators is {best_n_est} and it\'s cv f1 is {best_f1}\n')
    predict_prob(best_model, test_features[i], test_data[i % 2], 'gb_')

    best_f1 = 0
    best_n_est = 5
    best_model = None
    for n_est in [5, 10, 20, 40, 80]:
        model, f1 = train_bagging(train_features[i], train_data[i % 2], n_est)
        if f1 > best_f1:
            best_f1 = f1
            best_n_est = n_est
            best_model = model
    print(f'\nBag: Best number of estimators is {best_n_est} and it\'s cv f1 is {best_f1}\n')
    predict_prob(best_model, test_features[i], test_data[i % 2], 'bag_')

    for model in ['l1', 'dt', 'rf', 'knn', 'svm', 'ada', 'gb', 'bag']:
        if i == 1:
            doc_level_pred(f'{model}_fig_features_fig_model.pc', 'fig_features_fig_test.pc', 'fig_test.csv')
        if i == 3:
            doc_level_pred(f'{model}_fig_doc2vec_features_fig_model.pc', 'fig_doc2vec_features_fig_test.pc', 'fig_test.csv')
        if i == 5:
            doc_level_pred(f'{model}_fig_preprocessed_features_fig_model.pc', 'fig_preprocessed_features_fig_test.pc',
                       'fig_test.csv')
        if i == 7:
            doc_level_pred(f'{model}_fig_preprocessed_doc2vec_features_fig_model.pc',
                       'fig_preprocessed_doc2vec_features_fig_test.pc', 'fig_test.csv')

# fig best
predict_unlabeled('fig_vectorizer.csv', 'rf_fig_features_fig_model.pc', True, 'figure_corpus/figures*', 'labeled_', fig=True)
# fulltext best
predict_unlabeled('../data_new/vectorizer.csv', '../data_new/rf_features_model.pc', True, 'fulltext_corpus/text*', 'labeled_', fig=False)