import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from text_utils import decide_on_word
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score


def filter_word_list(word_list):
    # decide_on_word removes punctuation and replaces numbers with specific token.
    return [w for w in map(decide_on_word, word_list) if len(w) > 0]
    # return word_list


def filter_specific_words(word_list, word_to_counts):
    words_to_remove = ['man', 'men', "men's", 'msm']
    prefixes = ['male', 'female', 'woman', 'women', "women's", 'boys', 'girls', 'mother', 'maternal', 'lbw',
                'fem-prep', 'vaginosis', 'mammogram', 'prenatal',
                'ovar', #'ovarian', 'ovary',
                'pregnan',  # 'pregnancy', 'pregnant',
                'postpartum', 'uterus', 'abortion', 'perinatal', 'antenatal', 'newborn', 'birth', 'obstetrics',
                'menopausal', 'menopause',
                'breast', 'postmenopausal', 'contraceptive',
                'cervi', # 'cervix', 'cervical',
                'vulvovaginal', 'vulvar', 'vagina',  # 'vaginal
                'prostate', 'castration', 'erectile', 'androgen-deprivation', 'transrectal', 'anal', 'prostatic', 'crpc',
                'ipss', 'mcrpc',
                'pcos', 'polycystic', 'fallopian',
                'gestation', 'hysterectomy', 'mbc', 'cesarean', 'preeclampsia', 'menstrual', 'nonpregnant', 'embryo',
                'her2', 'cleopatra', 'vasomotor', 'exemestane', 'premenopausal',
                # noise:
                'filipino', 'spite', 'alternate-day', 'on-demand', 'botswana']

    def test_word(w):
        if len(re.findall('nct[0-9]+', w)) > 0:
            return False
        for word in prefixes:
            if w.startswith(word):
                return False
            if w in words_to_remove:
                return False
        if word_to_counts[word]['female'] == 0 or word_to_counts[word]['male'] == 0:
            return False
        return True

    # words_to_remove = []
    return [w for w in word_list if test_word(w)]


def read_abstracts():
    df = pd.read_csv('abstracts_and_population_tokenized_for_cui2vec_copyrightfix_sent_sep.csv', index_col=0)
    # df = pd.read_csv('pubmed2019_abstracts_with_participants.csv', index_col=0)
    #df = pd.read_csv("abstracts_and_population.csv", index_col=0)
    df['all_participants'] = df['male'] + df['female']
    df['percent_female'] = df['female'] / df['all_participants']
    df = df.dropna(subset=['abstract'])

    word_to_counts = pd.read_csv(os.path.join('outputs_regression', 'words_and_counts.csv'), index_col=0).to_dict(orient='index')

    def process_row(row):
        title = row['title']
        if type(title) == str:
            title = title + " "
        else:
            title = ""
        abstract = row['abstract'].replace(';', ' ')
        clean = filter_word_list((title + abstract).lower().split())
        return filter_specific_words(clean, word_to_counts)

    df['tokenized'] = df.apply(process_row, axis=1)
    return df


def get_vocab(list_of_word_lists):
    vocab_set = set()
    for lst in list_of_word_lists:
        for w in lst:
            vocab_set.add(w)
    vocab = sorted(list(vocab_set))
    word_to_index = {w: i for i, w in enumerate(vocab)}
    print("vocab size: {}".format(len(vocab)))
    return word_to_index, vocab
# get_vocab(abstracts['tokenized'])


def texts_to_BOW(texts_list, vocab):
    """embed abstracts using BOW. (set representation).
    :param texts_list: list of abstracts, each is a list of words.
    :param vocab: dictionary of word to index
    :return:
    """
    X = lil_matrix((len(texts_list), len(vocab)))
    for i, abstract in tqdm(enumerate(texts_list), total=len(texts_list)):
        word_indices = [vocab[w] for w in sorted(set(abstract))]
        X[i, word_indices] = 1
    return X.tocsr()


def shuffle_csr(mat):
    # utility function to shuffle sparse csr_matrix rows
    index = np.arange(mat.shape[0])
    np.random.shuffle(index)
    return mat[index, :]


def MSE_score(X, y, model):
    pred = model.predict(X)
    return mean_squared_error(y, pred)


def regression_for_percent_female(df, out=None):
    """regression for the percent of female participants."""
    vocab, index_to_word = get_vocab(df['tokenized'])
    X = texts_to_BOW(df['tokenized'], vocab)
    y = df['percent_female']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

    # shuffle the data for a random baseline.
    shuff_Xtrain = shuffle_csr(Xtrain)
    shuff_Xtest = shuffle_csr(Xtest)

    models = [#("Vanilla Linear Regression", LinearRegression),
              ("Ridge Regression", Ridge),
              #("Lasso Regression", Lasso)
    ]
    for model_desc, model_class in models:
        model = model_class()
        print("\n\n{}:".format(model_desc))
        model.fit(Xtrain, ytrain)
        print("score on train: {}/1".format(model.score(Xtrain, ytrain)))
        print("MSE on train: {}".format(MSE_score(Xtrain, ytrain, model)))

        print("score on test: {}/1".format(model.score(Xtest, ytest)))
        print("MSE on test: {}".format(MSE_score(Xtest, ytest, model)))

        c = model.coef_
        words_and_weights = zip(index_to_word, c)
        print("coefficients: min={}, max={}, mean:{}".format(max(c), min(c), np.mean(c)))
        prominent = sorted(words_and_weights, key=lambda x: np.abs(x[1]), reverse=True)
        print("top 20 prominent features: {}".format(prominent[:20]))
        if out is not None:
            f = open(out, 'w')
            f.write('word, weight\n')
            for word, weight in prominent:
                f.write(f'{word},{weight}\n')
            f.close()
        model = model_class()
        model.fit(shuff_Xtrain, ytrain)
        print("score on random train: {}/1".format(model.score(shuff_Xtrain, ytrain)))
        print("MSE on random train: {}".format(MSE_score(shuff_Xtrain, ytrain, model)))
        print("score on random test: {}/1".format(model.score(shuff_Xtest, ytest)))
        print("MSE on random test: {}".format(MSE_score(shuff_Xtest, ytest, model)))

    return vocab, Xtrain, Xtest, ytrain, ytest


def year_to_binary_label(year):
    if 2010 <= year <= 2013:
        return 0
    if 2016 <= year <= 2018:
        return 1


def classification_for_year(df, binary):
    if binary:
        df['label'] = df['year'].apply(year_to_binary_label)
        df = df.dropna(subset=['label'])
        y = df['label']
    else:
        y = df['year']
    vocab, index_to_word = get_vocab(df['tokenized'])
    X = texts_to_BOW(df['tokenized'], vocab)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

    # shuffle the data for a random baseline.
    shuff_Xtrain = shuffle_csr(Xtrain)
    shuff_Xtest = shuffle_csr(Xtest)

    model = LogisticRegression()
    model.fit(Xtrain, ytrain)
    ypred_train = model.predict_proba(Xtrain)[:, 1]
    ypred_test = model.predict_proba(Xtest)[:, 1]
    if binary:
        print(f"Accuracy on train: {accuracy_score(ytrain, model.predict(Xtrain))}")
        print(f"Accuracy on test: {accuracy_score(ytest, model.predict(Xtest))}")
        print(f"AUC on train: {roc_auc_score(ytrain, ypred_train)}")
        print(f"AUC on test: {roc_auc_score(ytest, ypred_test)}")
    else:
        print(f"Logloss on train: {log_loss(ytrain, model.predict_proba(Xtrain))}")
        print(f"Logloss on test: {log_loss(ytest, model.predict_proba(Xtest))}")

    model = LogisticRegression()
    model.fit(shuff_Xtrain, ytrain)
    ypred_train = model.predict_proba(shuff_Xtrain)[:, 1]
    ypred_test = model.predict_proba(shuff_Xtest)[:, 1]
    if binary:
        print(f"Accuracy on random train: {accuracy_score(ytrain, model.predict(Xtrain))}")
        print(f"Accuracy on random test: {accuracy_score(ytest, model.predict(Xtest))}")
        print(f"AUC on random train: {roc_auc_score(ytrain, ypred_train)}")
        print(f"AUC on random test: {roc_auc_score(ytest, ypred_test)}")
    else:
        print(f"Logloss on train: {log_loss(ytrain, model.predict_proba(shuff_Xtrain))}")
        print(f"Logloss on test: {log_loss(ytest, model.predict_proba(shuff_Xtest))}")


if __name__ == "__main__":
    df = read_abstracts()
    vocab, Xtrain, Xtest, ytrain, ytest = regression_for_percent_female(
        df, os.path.join('outputs_regression', 'word_and_weights_blacklisted4.csv'))
