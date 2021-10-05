import os
import pandas as pd
import numpy as np
import random
from ast import literal_eval
from scipy.stats import chi2_contingency, norm
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
import xgboost as xgb
import time
from tqdm import tqdm, tqdm_gui
tqdm.pandas()
import pickle
from collections import defaultdict
#from translation import AutoEncoder
from matplotlib import pyplot as plt

from delong.compare_auc_delong_xu import delong_roc_test
from read_utilities import DIAGNOSIS_FILES, DIAGNOSIS_FIELD_NAMES, ICD9_CUI_FILE,\
 DISEASE_PAIRS_FILE, DISEASE_PAIRS_V2, DISEASE_PAIRS_DOUBLE_LABELS
from read_utilities import read_customers, Translator, read_diagnosis,\
 read_patients_with_measurements, make_array, read_embedding_file, get_IDF_dict, union, make_embedding_matrix
from read_utilities import FOR_COMORBIDITY, FOR_TRANSLATION
from AUC_confidence_interval import AUROC_confidence_interval
 
# TODO: change to 100
EPOCHS_FOR_NN_CLASSIFIER = 10

   

################ comorbidity (disease pair) classifier #######################


def encode_from_df(df, field_to_encode, autoenc, encoded_field=None):
    if encoded_field is not None:
        return np.stack(df[encoded_field])
    to_encode = np.stack(df[field_to_encode])
    return autoenc.encode_matrix(to_encode)
    

def dataframe_to_x_and_y(df, label_name, encoded_x,
                         mode='enc_emb', combine_mode='concat'):
    # mode in: {'enc', 'emb', 'enc_emb', 'emb_random', 'emb_emb', 'emb_emb_enc'}
    # combine_mode in: {'concat', 'hadamard'}
    X = []
    y = df[label_name].values.astype('bool')
    if mode == 'enc':
        return encoded_x, y
    if mode in ('emb', 'emb_random', 'enc_emb'):
        for row_index, row in df.iterrows():
            if combine_mode == 'concat':
                vec = np.concatenate((row['source_emb'], row['target_emb'])) 
            elif combine_mode == 'hadamard':
                vec = row['source_emb'] * row['target_emb']
            X.append(vec)
        X = np.stack(X)
    if mode in ('emb_emb', 'emb_emb_enc'):
        for row_index, row in df.iterrows():
            if combine_mode == 'concat':
                vec = np.concatenate((row['source_emb'], row['source_emb2'],
                                      row['target_emb'], row['target_emb2'])) 
            elif combine_mode == 'hadamard':
                vec = np.concatenate((row['source_emb'] * row['target_emb'],
                                      row['source_emb2'] * row['target_emb2']))
            X.append(vec)
        X = np.stack(X)
    if mode in ('enc_emb', 'emb_random', 'emb_emb_enc'):
        X = np.concatenate((X, encoded_x), axis=1)
    print("*******************shape of X:{}".format(X.shape))
    return X, y


def get_NN_model(input_size):
    model = Sequential()
    model.add(Dense(50, input_dim=input_size, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def filter_disease_pairs_by_embeddings(emb_files, output_file_no_ext, pairs=None, cui_list_file=None,
                                       allow_disease_overlap=True):
    if pairs is None:
        pairs = pd.read_csv(DISEASE_PAIRS_DOUBLE_LABELS, index_col=0)
    if cui_list_file is not None:
        cui_list = set(pd.read_csv(cui_list_file)['cui'].values)
        before_filter = len(pairs)
        pairs = pairs[pairs.apply(lambda row: row['source_cui'] in cui_list or row['target_cui'] in cui_list, axis=1)]
        print(f"After filtering by cui list {cui_list_file}, {len(pairs)}/{before_filter} pairs remaining.")
    embs = []
    for emb_file in emb_files:
        print("reading {}".format(emb_file))
        embs.append(read_embedding_file(emb_file))
    removed_diseases = []
    
    def process_row(row):
        d1 = row['source_cui']
        d2 = row['target_cui']
        name1 = row['source_name']
        name2 = row['target_name']
        ret = True
        for emb in embs:
            for d, name in [(d1, name1), (d2, name2)]:
                if d not in emb:
                    ret = False
                    removed_diseases.append(name)
        return ret
    pairs = pairs[pairs.apply(process_row, axis=1)]
    if allow_disease_overlap:
        CV_indices = list(ShuffleSplit(n_splits=5, test_size=0.2, random_state=0).split(pairs))
        CV_indices = [(pairs.iloc[train].index.values, pairs.iloc[test].index.values) for train, test in CV_indices]
        pickle.dump(CV_indices, open(output_file_no_ext+"_CV.pickle", "wb"))
    else:
        all_diseases = list(pd.concat([pairs['source_cui'], pairs['target_cui']]).drop_duplicates().values)
        CV_indices = list(ShuffleSplit(n_splits=5, test_size=0.2, random_state=0).split(all_diseases))
        def index_list_to_cui_list(index_list, diseases):
            return set([diseases[i] for i in index_list])
        cui_lists = [(index_list_to_cui_list(train, all_diseases), index_list_to_cui_list(test, all_diseases)) for
                     train, test in CV_indices]
        CV_indices = []
        for train_cuis, test_cuis in cui_lists:
            train = pairs[pairs.apply(
                lambda row: row['source_cui'] in train_cuis and row['target_cui'] in train_cuis,
                axis=1)].index.values
            test = pairs[pairs.apply(
                lambda row: row['source_cui'] in test_cuis and row['target_cui'] in test_cuis,
                axis=1)].index.values
            CV_indices.append((train, test))
            pickle.dump(CV_indices, open(output_file_no_ext + "_CV.pickle", "wb"))
    removed_diseases = list(set(removed_diseases))
    f = open(output_file_no_ext+"_removed_diseases.txt", "w")
    f.write("\n".join(list(set(removed_diseases))))
    return removed_diseases


class ComorbidityClassifierBase:
    def __init__(self, model_name, 
                 custom_test_file=None,
                 double_labels=False,
                 custom_cv_index=None):
        self.model_name = model_name # one of: NN or logreg
        self.double_labels = double_labels
        if self.double_labels:
            self.pairs = pd.read_csv(DISEASE_PAIRS_DOUBLE_LABELS, index_col=0)
            #read_cv_index = 'big_dataset_cui2vec_style_w2v_double_labels_CV.pickle'
            read_cv_index = 'data/pairs_cv_index_CV.pickle'
        else:
            self.pairs = pd.read_csv(DISEASE_PAIRS_V2, index_col=0)
            read_cv_index = 'big_dataset_cui2vec_style_w2v_no_min_count_shuffle_CV.pickle'
        if custom_cv_index:
            read_cv_index = custom_cv_index
        self.pairs = self.pairs.rename({'source_name': 'source', 'target_name': 'target'}, axis=1)
        self.test = None
        if custom_test_file is not None:
            self.test = pd.read_csv(custom_test_file, index_col=0)
        self.CV_indices = pickle.load(open(read_cv_index, "rb"))
        self.accs = []
        self.aucs = []
        
    def set_label_prefix_and_desc(self, model_desc, label_prefix, test_label_prefix=None):
        self.model_desc = model_desc # example: male_emb_logreg
        self.label_prefix = label_prefix # example: 'W_'
        #label_suffix = 'pos_dep_double' if self.double_labels else 'pos_dep'
        label_suffix = 'pos_dep'
        self.label_name = label_prefix + label_suffix
        
        if self.double_labels:
            self.pairs[self.label_name+'_double'] = self.pairs[self.label_name].values.astype('bool') & self.pairs[self.label_name+'21'].values.astype('bool')
            label_suffix = 'pos_dep_double'
            self.label_name = label_prefix + label_suffix
        
        if test_label_prefix is None:
            self.test_label = self.label_name
            self.test_label_prefix = label_prefix
        else:
            self.test_label = test_label_prefix + label_suffix
            
        
        self.fields_to_keep = ['source', 'target','source_cui', 'target_cui',
                              self.test_label, self.label_prefix+"1",
                              self.label_prefix+"2", self.label_prefix+"both", 
                              self.label_prefix+"none", self.label_prefix+"pval",
                              '{}_pred_prob'.format(self.model_desc)]
    
    def set_embedding(self, emb_file):
        self.emb = read_embedding_file(emb_file)
        self.pairs = self.embed_pairs(self.emb, self.pairs, '{}_cui', '{}_emb')
        if self.test is not None:
            self.test = self.embed_pairs(self.emb, self.test, '{}_cui', '{}_emb')
        
        
    def embed_pairs(self, emb, df, input_field_template, output_field_template):
        for name in ['source', 'target']:
            df[output_field_template.format(name)] = df[input_field_template.format(name)].apply(lambda x: emb[x])
        return df
    
    def get_y_labels(self, df):
        return df[self.label_name].values.astype('bool')
        
    def df_to_xy(self, df):
        X = []
        y = self.get_y_labels(df)
        for _, row in df.iterrows():
            vec = np.concatenate((row['source_emb'], row['target_emb']))
            X.append(vec)
        X = np.stack(X)
        return X, y
    
    def prepare_data(self, train_index, test_index):
        train = self.pairs.loc[train_index]
        test = self.test
        if test is None:
            test = self.pairs.loc[test_index]
        print("train index: {} train: {} test_index: {} test: {}".format(len(train_index), len(train), len(test_index), len(test)))
        train = train[(train[self.label_prefix+'1'] >= 30) & (train[self.label_prefix+'2'] >=30)]
        test = test[(test[self.test_label_prefix+'1'] >= 30) & (test[self.test_label_prefix+'2'] >=30)]
        print("after filter by number of patients with each disease:\ntrain index: {} train: {} test_index: {} test: {}".format(len(train_index), len(train), len(test_index), len(test)))
#        encX_train = self.encode_from_df(train, pair_emb_field_name, autoenc, encoded_field=encoded_field_name)
#        encX_test = self.encode_from_df(test, pair_emb_field_name, autoenc, encoded_field=encoded_field_name)
        Xtrain, ytrain = self.df_to_xy(train)
        Xtest, ytest = self.df_to_xy(test)
        return train, test, Xtrain, ytrain, Xtest, ytest
    
    def train_and_predictNN(self, Xtrain, ytrain, Xtest, ytest):
        input_size = Xtrain.shape[1]
        model = get_NN_model(input_size)
        model.fit(Xtrain, ytrain, epochs=EPOCHS_FOR_NN_CLASSIFIER, batch_size=32, verbose=1)
        pred = model.predict(Xtest)
        loss, acc = model.evaluate(Xtest, ytest)
        #pos_pred = (sum(pred)/len(pred))[0]
        return pred, acc
    
    def train_and_predict_logreg(self, Xtrain, ytrain, Xtest, ytest):
        model = LogisticRegression(random_state=0)
        model.fit(Xtrain, ytrain)
        pred = model.predict_proba(Xtest)[:, 1]
        acc = model.score(Xtest, ytest)
        return pred, acc
    
    def train_and_predict(self, *args):
        if self.model_name == 'NN':
            return self.train_and_predictNN(*args)
        if self.model_name == 'logreg':
            return self.train_and_predict_logreg(*args)
        print(f"Unsupported model name: {self.model_name}")
        
    def evaluate(self):
        accs = []
        aucs = []
        all_tests = None
        for train_index, test_index in self.CV_indices:
            train, test, Xtrain, ytrain, Xtest, ytest = self.prepare_data(train_index, test_index)
            ypred, acc = self.train_and_predict(Xtrain, ytrain, Xtest, ytest)
            test['{}_pred_prob'.format(self.model_desc)] = ypred
            test = test[self.fields_to_keep]
            if all_tests is None:
                all_tests = test
            else:
                all_tests = pd.concat([all_tests, test])
            accs.append(acc)
            aucs.append(roc_auc_score(ytest, ypred))
        print("avg acc: {}, avg auc: {}, desc: {}".format(
            np.mean(accs),
            np.mean(aucs),
            '{}_pred_prob'.format(self.model_desc)))
        return accs, aucs, all_tests


class ComorbidityClassifierConcatEmbs(ComorbidityClassifierBase):
    def __init__(self, **kwargs):
        super(ComorbidityClassifierConcatEmbs, self).__init__(**kwargs)

    def set_embedding(self, emb_file, secondary_emb_file):
        super(ComorbidityClassifierConcatEmbs, self).set_embedding(emb_file)
        self.secondary_emb = read_embedding_file(secondary_emb_file)
        self.pairs = self.embed_pairs(self.secondary_emb, self.pairs, '{}_cui', '{}_emb2')
        if self.test is not None:
            self.test = self.embed_pairs(self.secondary_emb, self.test, '{}_cui', '{}_emb2')
    
    def df_to_xy(self, df):
        X = []
        y = self.get_y_labels(df)
        for _, row in df.iterrows():
            vec = np.concatenate((row['source_emb'], row['source_emb2'],
                                  row['target_emb'], row['target_emb2']))
            X.append(vec)
        X = np.stack(X)
        return X, y


class ComorbidityClassifierEnc(ComorbidityClassifierBase):
    def __init__(self, use_idf_weights=False, **kwargs):
        super(ComorbidityClassifierConcatEnc, self).__init__(**kwargs)
        self.use_idf_weights = use_idf_weights
        
    def embed_for_encoding(self, df):
        if self.use_idf_weights:
            w1 = df['source_cui'].apply(lambda x: self.idfs[x])
            w2 = df['target_cui'].apply(lambda x: self.idfs[x])
            df['pair_emb'] = (
                    df['source_emb2'].multiply(w1)
                    + df['target_emb2'].multiply(w2)).divide(w1+w2)
        else:
            df['pair_emb'] = (df['source_emb2']+df['target_emb2'])/2
        return df
    
    def set_embedding(self, autoenc, emb_file, emb_file_for_encoding=None):
        super(ComorbidityClassifierConcatEmbs, self).set_embedding(emb_file)
        self.autoenc = autoenc
        if emb_file_for_encoding is not None:
            self.secondary_emb = read_embedding_file(emb_file_for_encoding)
            self.pairs = self.embed_pairs(self.secondary_emb, self.pairs, '{}_cui', '{}_emb2')
            if self.test is not None:
                self.test = self.embed_pairs(self.secondary_emb, self.test, '{}_cui', '{}_emb2')
        else: # use the primary embedding as input to the autoencoder
            for name in ('source', 'target'):
                self.pairs[f'{name}_emb2'] = self.pairs[f'{name}_emb']
            if self.test is not None:
                for name in ('source', 'target'):
                    self.test[f'{name}_emb2'] = self.test[f'{name}_emb']
        if self.use_idf_weights:
            self.idfs =  defaultdict(int, get_IDF_dict())
        self.pairs = self.embed_for_encoding(self.pairs)
        if self.test is not None:
            self.test = self.embed_for_encoding(self.test)
    
    def df_to_xy(self, df):
        # encode everything
        encodedX = self.autoenc.encode_matrix(np.stack(df['pair_emb']))
        y = self.get_y_labels(df)
        return encodedX, y
 

class ComorbidityClassifierConcatEnc(ComorbidityClassifierEnc):
    def __init__(self, **kwargs):
        super(ComorbidityClassifierConcatEnc, self).__init__(**kwargs)
    
    def df_to_xy(self, df):
        # encode everything
        encodedX = self.autoenc.encode_matrix(np.stack(df['pair_emb']))
        X = []
        y = self.get_y_labels(df)   
        for _, row in df.iterrows():
            vec = np.concatenate((row['source_emb'], row['target_emb']))
            X.append(vec)
        X = np.stack(X)
        X = np.concatenate((X, encodedX), axis=1)
        return X, y


class ComorbidityClassifierConcatEmbsAndEnc(ComorbidityClassifierEnc):
    def __init__(self, **kwargs):
        super(ComorbidityClassifierConcatEmbsAndEnc, self).__init__(**kwargs)
    
    def df_to_xy(self, df):
        # encode everything
        encodedX = self.autoenc.encode_matrix(np.stack(df['pair_emb']))
        X = []
        y = self.get_y_labels(df)
        for _, row in df.iterrows():
            vec = np.concatenate((row['source_emb'], row['source_emb2'],
                                  row['target_emb'], row['target_emb2']))
            X.append(vec)
        X = np.stack(X)
        X = np.concatenate((X, encodedX), axis=1)
        return X, y

   

def batch_runs(emb_dir, df=None, model="NN", label="W_", output_file=None):
    # model = "NN" or "LogisticRegression"
    model_to_str = {"NN":"NN", "LogisticRegression": "logreg"}
    #sizes = (8, 10, 20, 30, 40, 50)
    #sizes = (10, 20, 30, 40, 50)
    sizes = (40,)
    res = {}
    com_class = ComorbidityClassifierBase(model_to_str[model])
    for size in sizes:
        for version in ('neutral', 'female', 'male'):
            desc = "{}{}_{}".format(version, size, model_to_str[model])
            
            com_class.set_label_prefix_and_desc(desc, label)
            com_class.set_embedding(emb_dir + f"{version}_cui2vec_style_w2v_copyrightfix_{size}_emb_filtered.tsv")
            accs, aucs, all_test = com_class.evaluate()
            if df is None:
                df = all_test
            else:
                df[desc+"_pred_prob"] = all_test[desc+"_pred_prob"]
            res[desc] = (np.mean(accs), np.mean(aucs))
            print("finished {} {}".format(version, size))
    # one more time for random emb
    for size in sizes:
        desc = "random{}_{}".format(size, model_to_str[model])
        com_class.set_label_prefix_and_desc(desc, label)
        com_class.set_embedding(emb_dir + f"random{size}.tsv")
        accs, aucs, all_test = com_class.evaluate()
        if df is None:
            df = all_test
        else:
            df[desc+"_pred_prob"] = all_test[desc+"_pred_prob"]
        res[desc] = (np.mean(accs), np.mean(aucs))
        print("finished random {}".format(size))
    if output_file is not None:
        df.to_csv(output_file)
    return res, df
        

def make_random_emb(size, emb_dir, pairs):
    cuis = list(set(pairs['source_cui'].values.tolist()+pairs['target_cui'].values.tolist()))
    rand_emb = np.random.rand(len(cuis), size).tolist()
    emb_df = pd.DataFrame({'cui': cuis, 'vector':rand_emb})
    emb_df['vec_as_str'] = emb_df['vector'].apply(lambda x: ",".join([str(i) for i in x]))
    emb_df[['cui', 'vec_as_str']].to_csv(emb_dir+"random{}.tsv".format(size),
          sep="\t", index=False, header=False)
    


def read_categories(pairs_df):
    cat_df = pd.read_csv('E:\Shunit\cuis_filtered_glove_single1_nodes_with_categories_plia.csv')
    cat_df = cat_df.dropna(axis=0, subset=['category'])
    cat_df['category'] = cat_df['category'].apply(lambda x: x.split("/"))
    categories = list(union(cat_df['category'].values))
    categories_dict = {categories[i]: i for i in range(len(categories))} 
    res = pairs_df.merge(cat_df[['cui', 'category', 'can be chronic?']], 
                    left_on='source_cui', right_on='cui', how='inner')
    res = res.rename({'category': 'source_category', 
                      'can be chronic?': 'source_chronic'}, axis=1)
    res = res.drop('cui', axis=1)
    
    res = res.merge(cat_df[['cui', 'category', 'can be chronic?']], 
                    left_on='target_cui', right_on='cui', how='inner')
    res = res.rename({'category': 'target_category', 
                      'can be chronic?': 'target_chronic'}, axis=1)
    res = res.drop('cui', axis=1)
    return res, categories_dict

# class ClassificationResultsAnalyzer():
#     def __init__(self, classifier_res_file, by_category=False):
#         self.res = pd.read_csv(classifier_res_file, index_col=0)
#         print(f"reading res file: {classifier_res_file}, {len(self.res)} rows.")
#         if by_category:
#             self.res, _ = read_categories(self.res)
#             self.res = self.res.dropna(subset=['source_category', 'target_category'])[
#                     (self.res['target_chronic'] == 'TRUE') &
#                     (self.res['source_chronic'] == 'TRUE')]
#             print("working with categories")
#         else:
#             print("working without categories")
            
            


def analyze_classifier_results(
        classifier_res_file, label_name, output_file, show_precision_recall=True,
        model1=None, model2=None, threshold=None, show_categories=False, 
        show_diseases=True, divide_by_age=True, exclude=None, include=None):
    # classifier_res_file = "classify_res_female.csv"
    # label_name = 'W_pos_dep'
    res = pd.read_csv(classifier_res_file, index_col=0)
    print(f"reading res file: {classifier_res_file}, {len(res)} rows.")
    if show_categories:
        res, _ = read_categories(res)
        only_chronic = res.dropna(subset=['source_category', 'target_category'])[
                (res['target_chronic'] == 'TRUE') &
                (res['source_chronic'] == 'TRUE')]
        print("working with categories")
    else:
        print("working without categories")
        only_chronic = res
        
    # draw roc curve
    predictions = [c for c in only_chronic.columns if c.endswith('pred_prob')]
    
    plt.rcParams['figure.figsize'] = 10,10
    label = only_chronic[label_name]
    
    if show_precision_recall:
        if model1 is not None and model2 is not None:
            predictions = [model1, model2]
        print("showing precision-recall curve for {}".format(predictions))
        print("all test size {}".format(len(only_chronic)))
        for pred_name in predictions:
            if exclude is not None and pred_name in exclude:
                continue
            if include is not None and pred_name not in include:
                continue
            precision, recall, thresh = precision_recall_curve(
                    label, only_chronic[pred_name])
            if threshold is not None and pred_name in [model1, model2]:
                prec = np.mean(precision[:-1][(thresh > threshold-0.01) & (thresh < threshold+0.01)])
                print("precision for {} in thresh={}: {:.2f}".format(pred_name, threshold, prec))
            plt.plot(recall, precision, label=pred_name)
            #fpr, tpr, thresh = roc_curve(label, only_chronic[pred_name])
            #plt.plot(fpr, tpr, label=pred_name)
            
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend(loc=0)
        plt.grid(which='both')
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("precision-recall curve")
        plt.show()
        return
        
    if model1 is not None and model2 is not None and threshold is None:
        for pred_name in [model1, model2]:
            precision, recall, thresh = precision_recall_curve(
                    label, only_chronic[pred_name])
            # recall and precision are longer by one than thresholds
            plt.plot(thresh, recall[:-1], label=pred_name)
        plt.legend(loc=0)
        plt.xticks([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
        plt.yticks([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
        plt.grid(which='both')
        plt.xlabel("threshold")
        plt.ylabel("recall")
        plt.title("Recall matching")
        plt.show()
    
    # how to count model mistakes? 
    # use threshold where the two models have the same recall
    if model1 is not None and model2 is not None and threshold is not None:
        out = open(output_file, "w")
        print("writing to file {}".format(output_file))
        d = defaultdict(lambda : defaultdict(int)) # cat-> {'model1 correct, model 2 wrong': count, ...}
        single_cat_auc = defaultdict(lambda: {model1: [], model2: [], 'label':[]})
        cat_pairs = defaultdict(lambda : defaultdict(int)) # (cat1, cat2)->{'model1 correct, model 2 wrong': count, ...}
        cat_pos_rate = defaultdict(lambda: {True: 0, False: 0})
        cat_loss = defaultdict(lambda: {model1: [], model2: []})
        cat_auc_calc = defaultdict(lambda: {model1: [], model2: [], 'label':[]})
        
        ages = {'all_ages'}
        age_group = 'all_ages'
        if divide_by_age:
            ages = set(only_chronic['age_group'].values)
        
        if show_diseases:
            disease_data = defaultdict(lambda: {
                    'num_pairs':0, 
                    'classify_results': defaultdict(int),
                    'pos_rate': {True:0, False:0},
                    'auc_calc': {model1: [], model2: [], 'label': []},
                    'disease_name': ''
                })
            age_to_disease_data = defaultdict(lambda : 
                defaultdict(lambda: {
                    'num_pairs':0, 
                    'classify_results': defaultdict(int),
                    'pos_rate': {True:0, False:0},
                    'auc_calc': {model1: [], model2: [], 'label': []},
                    'disease_name': ''
                }))

        for i, r in only_chronic.iterrows():
            # who was right?
            label = r[label_name]
            if label == 1:
                model1_correct = r[model1] > threshold
                model2_correct = r[model2] > threshold
                model1_loss = 1 - r[model1]
                model2_loss = 1 - r[model2]
            else:
                model1_correct = r[model1] < threshold
                model2_correct = r[model1] < threshold
                model1_loss = r[model1]
                model2_loss = r[model2]
            if show_categories:
                source_cats = r['source_category']
                target_cats = r['target_category']
                for cat in source_cats+target_cats:
                    d[cat][(model1_correct, model2_correct)] += 1
                    single_cat_auc[cat][model1].append(r[model1])
                    single_cat_auc[cat][model2].append(r[model2])
                    single_cat_auc[cat]['label'].append(label)
                
                cat_pairs_for_row = []
                for cat1 in source_cats:
                    for cat2 in target_cats:
                        if cat1 > cat2:
                            cat_pairs_for_row.append((cat2, cat1))
                        else:
                            cat_pairs_for_row.append((cat1, cat2))
                cat_pairs_for_row = set(cat_pairs_for_row)
                for cat_pair in cat_pairs_for_row:
                    cat_pairs[cat_pair][(model1_correct, model2_correct)] += 1
                    cat_pos_rate[cat_pair][label] += 1
                    cat_loss[cat_pair][model1].append(model1_loss)
                    cat_loss[cat_pair][model2].append(model2_loss)
                    cat_auc_calc[cat_pair][model1].append(r[model1])
                    cat_auc_calc[cat_pair][model2].append(r[model2])
                    cat_auc_calc[cat_pair]['label'].append(label)
            if divide_by_age:
                age_group = r['age_group']
            # analyze per disease
            if show_diseases:
                for s in ('source', 'target'):
                    cui, disease_name = r['{}_cui'.format(s)], r[s]
                    age_to_disease_data[age_group][cui]['disease_name'] = disease_name
                    age_to_disease_data[age_group][cui]['num_pairs'] += 1
                    age_to_disease_data[age_group][cui]['classify_results'][(model1_correct, model2_correct)] += 1
                    age_to_disease_data[age_group][cui]['pos_rate'][label]+=1
                    age_to_disease_data[age_group][cui]['auc_calc'][model1].append(r[model1])
                    age_to_disease_data[age_group][cui]['auc_calc'][model2].append(r[model2])
                    age_to_disease_data[age_group][cui]['auc_calc']['label'].append(label)
                    
                    disease_data[cui]['disease_name'] = disease_name
                    disease_data[cui]['num_pairs'] += 1
                    disease_data[cui]['classify_results'][(model1_correct, model2_correct)] += 1
                    disease_data[cui]['pos_rate'][label]+=1
                    disease_data[cui]['auc_calc'][model1].append(r[model1])
                    disease_data[cui]['auc_calc'][model2].append(r[model2])
                    disease_data[cui]['auc_calc']['label'].append(label)

        if show_categories:
            out.write("category,auc1,auc2,both correct,model1 correct,model2 correct,both wrong\n")
            for cat in d:
                v = d[cat]
                auc1 = roc_auc_score(single_cat_auc[cat]['label'], single_cat_auc[cat][model1])
                auc2 = roc_auc_score(single_cat_auc[cat]['label'], single_cat_auc[cat][model2])
                out.write("{},{},{},{},{},{},{}\n".format(cat,
                      auc1, auc2,
                      v[(True, True)], v[(True, False)], 
                      v[(False, True)], v[(False, False)]))
                
            print("cat1,cat2,pos rate,model1 avg loss,model2 avg loss,auc1,auc2,auc1CI,auc2CI,auc diff pval,both correct,model1 correct,model2 correct,both wrong,pair count")
            for cat_pair in cat_pairs:
                v = cat_pairs[cat_pair]
                pos_rate = -1
                cat_count = (cat_pos_rate[cat_pair][True]+cat_pos_rate[cat_pair][False])
                if cat_count > 0:
                    pos_rate = cat_pos_rate[cat_pair][True]/cat_count
                if pos_rate in (0,1,-1): # can't calculate AUC in this case
                    auc1 = None
                    auc2 = None
                    delong_res = None
                    auc1_CI = [None, None]
                    auc2_CI = [None, None]
                else:
                    auc1 = roc_auc_score(cat_auc_calc[cat_pair]['label'], cat_auc_calc[cat_pair][model1])
                    auc2 = roc_auc_score(cat_auc_calc[cat_pair]['label'], cat_auc_calc[cat_pair][model2])
                    auc1_CI = AUROC_confidence_interval(auc1, cat_pos_rate[cat_pair][True], cat_pos_rate[cat_pair][False])
                    auc2_CI = AUROC_confidence_interval(auc2, cat_pos_rate[cat_pair][True], cat_pos_rate[cat_pair][False])
                    delong_res = 10**delong_roc_test(
                            np.array([int(x) for x in cat_auc_calc[cat_pair]['label']]),
                            np.array(cat_auc_calc[cat_pair][model1]),
                            np.array(cat_auc_calc[cat_pair][model2]))[0][0]
                out.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    cat_pair[0], cat_pair[1], pos_rate,
                    np.mean(cat_loss[cat_pair][model1]),
                    np.mean(cat_loss[cat_pair][model2]),
                    auc1,
                    auc2,
                    "{}-{}".format(auc1_CI[0],auc1_CI[1]),
                    "{}-{}".format(auc2_CI[0],auc2_CI[1]),
                    delong_res,
                    v[(True, True)],
                    v[(True, False)], 
                    v[(False, True)], 
                    v[(False, False)],
                    cat_count))
            
        if show_diseases and divide_by_age:
            out.write("disease;cui;num_pairs;pos_rate;auc1;auc2;auc1CI;auc2CI;auc diff pval;both correct;model1 correct;model2 correct;both wrong;age_group\n")
            for age_group in ages:
                disease_to_data = age_to_disease_data[age_group]
                for disease_cui in disease_to_data:
                    v = disease_to_data[disease_cui]
                    disease_pair_count = v['num_pairs']
                    pos_rate = None
                    if disease_pair_count > 0:
                        pos_rate = v['pos_rate'][True]/disease_pair_count
                    auc1, auc2, delong_res  = None, None, None
                    auc1_CI, auc2_CI = [None, None], [None, None]
                    if pos_rate is not None and pos_rate not in (0,1):    
                        auc1 = roc_auc_score(v['auc_calc']['label'], v['auc_calc'][model1])
                        auc2 = roc_auc_score(v['auc_calc']['label'], v['auc_calc'][model2])
                        auc1_CI = AUROC_confidence_interval(auc1, v['pos_rate'][True], v['num_pairs']-v['pos_rate'][True])
                        auc2_CI = AUROC_confidence_interval(auc2, v['pos_rate'][True], v['num_pairs']-v['pos_rate'][True])
                        delong_res = 10**delong_roc_test(
                                np.array([int(x) for x in v['auc_calc']['label']]),
                                np.array(v['auc_calc'][model1]),
                                np.array(v['auc_calc'][model2]))[0][0]
                    out.write("{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(
                            v['disease_name'],
                            disease_cui,
                            disease_pair_count,
                            pos_rate,
                            auc1,
                            auc2,
                            "{}-{}".format(auc1_CI[0],auc1_CI[1]),
                            "{}-{}".format(auc2_CI[0],auc2_CI[1]),
                            delong_res,
                            v['classify_results'][(True, True)],
                            v['classify_results'][(True, False)], 
                            v['classify_results'][(False, True)], 
                            v['classify_results'][(False, False)],
                            age_group
                            ))
        if show_diseases and not divide_by_age:
            #print(f"found {len(disease_data)} diseases. Found hyper? {'C0242339_hyper' in disease_data}")
            
            out.write("disease;cui;num_pairs;pos_rate;auc1;auc2;auc1CI;auc2CI;auc diff pval;both correct;model1 correct;model2 correct;both wrong;age_group\n")
            for disease_cui in disease_data:
                print(f"disease_cui: {disease_cui}")
                v = disease_data[disease_cui]
                disease_pair_count = v['num_pairs']
                pos_rate = None
                if disease_pair_count > 0:
                    pos_rate = v['pos_rate'][True]/disease_pair_count
                auc1, auc2, delong_res = None, None, None
                auc1_CI, auc2_CI = [0,0], [0,0]
                if pos_rate is not None and pos_rate not in (0,1):
                    auc1 = roc_auc_score(v['auc_calc']['label'], v['auc_calc'][model1])
                    auc2 = roc_auc_score(v['auc_calc']['label'], v['auc_calc'][model2])
                    auc1_CI = AUROC_confidence_interval(auc1, v['pos_rate'][True], v['num_pairs']-v['pos_rate'][True])
                    auc2_CI = AUROC_confidence_interval(auc2, v['pos_rate'][True], v['num_pairs']-v['pos_rate'][True])
                    delong_res = 10**delong_roc_test(
                            np.array([int(x) for x in v['auc_calc']['label']]),
                            np.array(v['auc_calc'][model1]),
                            np.array(v['auc_calc'][model2]))[0][0]
                out.write("{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(
                        v['disease_name'],
                        disease_cui,
                        disease_pair_count,
                        pos_rate,
                        auc1,
                        auc2,
                        "{:.2f}-{:.2f}".format(auc1_CI[0],auc1_CI[1]),
                        "{:.2f}-{:.2f}".format(auc2_CI[0],auc2_CI[1]),
                        delong_res,
                        v['classify_results'][(True, True)],
                        v['classify_results'][(True, False)], 
                        v['classify_results'][(False, True)], 
                        v['classify_results'][(False, False)],
                        age_group
                        ))
        out.close()
                
################### Trainable embeddings ###############
        
def build_trainable_embedding_classifier(vocab_size, emb_matrix, trainable):
    emb_size = emb_matrix.shape[1]
    model_input1 = layers.Input(shape=(1,)) #source token and target token
    model_input2 = layers.Input(shape=(1,))
    # TODO: freeze embedding layer?
    emb_layer = layers.Embedding(input_dim=vocab_size, 
                                output_dim=emb_size, 
                                embeddings_initializer=keras.initializers.Constant(emb_matrix),
                                trainable=trainable, # first false, then make it true and compare
                                mask_zero=True)
    print(f"emb_layer is trainable: {trainable}")
    embedded1 = emb_layer(model_input1)
    embedded2 = emb_layer(model_input2)
    concat = layers.Concatenate()([embedded1, embedded2])
    dense1 = layers.Dense(50, input_dim=2*emb_size, activation='relu')(concat)
    out = layers.Dense(1, activation='sigmoid')(dense1)
    model = keras.Model([model_input1, model_input2], out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_semi_trainable_embedding_classifier(vocab_size, emb_matrix, trainable_dim=10):
    emb_size = emb_matrix.shape[1]
    model_input1 = layers.Input(shape=(1,)) #source token and target token
    model_input2 = layers.Input(shape=(1,))
    frozen_emb_layer = layers.Embedding(input_dim=vocab_size, 
                                output_dim=emb_size, 
                                embeddings_initializer=keras.initializers.Constant(emb_matrix),
                                trainable=False,
                                mask_zero=True)
    trainable_emb_layer = layers.Embedding(input_dim=vocab_size, 
                                output_dim=trainable_dim, 
                                trainable=True,
                                mask_zero=True)
    print(f"emb_layer is semi trainable, trainable size: {trainable_dim}")
    embedded1_pubmed = frozen_emb_layer(model_input1)
    embedded2_pubmed = frozen_emb_layer(model_input2)
    embedded1_trainable = trainable_emb_layer(model_input1)
    embedded2_trainable = trainable_emb_layer(model_input2)
    concat = layers.Concatenate()([embedded1_pubmed,embedded1_trainable,embedded2_pubmed, embedded2_trainable])
    dense1 = layers.Dense(50, input_dim=2*emb_size, activation='relu')(concat)
    out = layers.Dense(1, activation='sigmoid')(dense1)
    model = keras.Model([model_input1, model_input2], out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class SymmetricComorbidityDataGenerator(keras.utils.Sequence):
    def __init__(self, X1, X2, y, shuffle=True, batch_size=32):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.flip_counter = 0
        self.getitem_counter = 0
        self.on_epoch_end()
        
    def on_epoch_end(self):
        print(f"flipped {self.flip_counter}/{self.getitem_counter} times")
        self.flip_counter = 0
        self.getitem_counter = 0
        self.indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return len(self.y)//self.batch_size
    
    def __getitem__(self, index):
        self.getitem_counter += 1
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print(indexes)
        X1 = self.X1[indexes]
        #print(f"shape of X1[indexes]: {X1.shape}")
        X2 = self.X2[indexes]
        y = self.y[indexes]
        if np.random.choice([0,1]) == 1: # Flip it!
            self.flip_counter += 1
            return [X2, X1], y
        return [X1, X2], y


class ComorbidityClassifierTrainableEmb(ComorbidityClassifierBase):
    def __init__(self, emb_files_to_filter, trainable, **kwargs):
        s = time.time()
        super(ComorbidityClassifierTrainableEmb, self).__init__(**kwargs)
        self.trainable = trainable # 'semi', 'frozen', or 'trainable'
        before = len(self.pairs)
        for emb_file in emb_files_to_filter:
            emb = read_embedding_file(emb_file)
            self.pairs = self.pairs[
                self.pairs['source_cui'].apply(lambda x: x in emb) &
                self.pairs['target_cui'].apply(lambda x: x in emb)]
        print(f"ComorbidityClassifierTrainableEmb: kept {len(self.pairs)}/{before} pairs where both diseases have an embedding.")
        print(f"initialization took {time.time()-s} seconds")
    
    def set_embedding(self, emb_file):
        self.emb_file = emb_file
        self.emb = read_embedding_file(self.emb_file)
        self.index_to_diag = list(self.emb.keys())
        self.diag_to_index = {cui: idx for idx, cui in enumerate(self.index_to_diag)}
        self.emb_matrix = make_embedding_matrix(self.index_to_diag, self.emb_file, output_file=None)

    def get_y_labels(self, df):
        # double labels are handled before in set_label_prefix_and_desc
        return df[self.label_name].values.astype('int')
        
    def df_to_xy(self, df):
        y = self.get_y_labels(df)
        X1 = df['source_cui'].apply(lambda x: self.diag_to_index[x]).values
        X2 = df['target_cui'].apply(lambda x: self.diag_to_index[x]).values
        return [X1, X2], y
    
    def train_and_predict(self, Xtrain, ytrain, Xtest, ytest):
        if self.trainable == 'trainable':
            model = build_trainable_embedding_classifier(len(self.index_to_diag),
                                                     self.emb_matrix, True)
        elif self.trainable == 'frozen':
            model = build_trainable_embedding_classifier(len(self.index_to_diag),
                                                     self.emb_matrix, False)
        elif self.trainable == 'semi':
            model = build_semi_trainable_embedding_classifier(len(self.index_to_diag),
                                                     self.emb_matrix, trainable_dim=5)
        #print(f'Xtrain: {len(Xtrain)}, {len(Xtrain[0])}, {type(Xtrain[0])}')
        data_gen = SymmetricComorbidityDataGenerator(Xtrain[0], Xtrain[1], ytrain, shuffle=True, batch_size=32)
#        model.fit(Xtrain, ytrain, epochs=EPOCHS_FOR_NN_CLASSIFIER, 
#                  batch_size=32, shuffle=True, verbose=1)
        print("using fit_generator")
        model.fit_generator(generator=data_gen, epochs=EPOCHS_FOR_NN_CLASSIFIER, verbose=1)
        pred = model.predict(Xtest).squeeze(1)
        loss, acc = model.evaluate(Xtest, ytest)
        return pred, acc


def run_trainable_emb_model(label='W_',
                            emb_dir='embs',
                            emb_sizes=[40],
                            trainable='trainable',  # 'semi', 'frozen', or 'trainable'
                            double_labels=False,
                            descs_and_emb_files=[],
                            custom_cv=None,
                            output_file='outputs/classify_res_trainable_emb_W.csv'):
    df = None
    res = {}
    descs = []
    if len(descs_and_emb_files) == 0:
        for emb_size in emb_sizes:
            emb_file_template = os.path.join(emb_dir, "{}_cui2vec_style_w2v_copyrightfix_{}_emb_filtered.tsv")
    #        emb_file_template = os.path.join(emb_dir, "{}_cui2vec_style_w2v_no_min_count_shuffle_{}_emb_filtered.tsv")
            versions = ['female', 'male', 'neutral']
            emb_files = [emb_file_template.format(version, emb_size) for version in versions]
            descs.extend([f"{version}{emb_size}_{trainable}" for version in versions])
            #versions.append("random")
            #emb_files.append(os.path.join(emb_dir, f'random{emb_size}.tsv'))
            # descs.append(f"random{emb_size}_{trainable}"
    else:
        emb_files = [os.path.join(emb_dir, x[1]) for x in descs_and_emb_files]
        descs = [x[0] for x in descs_and_emb_files]
    print(f"descs: {descs}")
    com_class = ComorbidityClassifierTrainableEmb(
            emb_files, trainable,
            double_labels=double_labels, model_name='NNemb', custom_cv_index=custom_cv)
    for i in range(len(emb_files)):
        desc = descs[i]
        com_class.set_label_prefix_and_desc(desc, label)
        com_class.set_embedding(emb_files[i])
        accs, aucs, all_test = com_class.evaluate()
        if df is None:
            df = all_test
        else:
            df[desc+"_pred_prob"] = all_test[desc+"_pred_prob"]
        res[desc] = (np.mean(accs), np.mean(aucs))
        print("finished {}, res: {}".format(desc, res[desc]))
    if output_file is not None:
        df.to_csv(output_file)
    return res
        

    
   
################### Hybrid Model #######################

def categories_into_matrix(df, cat_to_index):
    mat = np.zeros((len(df), len(cat_to_index)))
    row_index = 0
    for _, row in df.iterrows():
        for cat in row['source_category']:
            mat[row_index, cat_to_index[cat]] += 1
        for cat in row['target_category']:
            mat[row_index, cat_to_index[cat]] += 1
        row_index += 1
    return mat
        

def hybrid_model(embedding_file, label_prefix, model_name, model_desc, autoenc, emb_file_for_encode):
    # Embedding file female: "E:\\Shunit\\female1_emb.tsv"
    # label_prefix example: 'W_', 'M_', 'all_', 'W_>70_3_'...
    # supported options for model_name: 'NN', 'LogisticRegression', 'XGBoost' 
    #1. read pairs and their label - sig/ not sig
    pairs = pd.read_csv(DISEASE_PAIRS_FILE,
            #"E:\Shunit\cuis_filtered_glove_single1_min40_all_pairs_results_ztest_grouped_diags.csv",
            #"E:\Shunit\cuis_filtered_glove_single1_min40_all_pairs_results_ztest_grouped_diags_population_slices.csv",
            #"E:\Shunit\cuis_filtered_glove_single1_pairs_151nodes_ztest.csv",
            index_col=0)
    pairs, cat_to_index = read_categories(pairs)
    # after this we should have source_category, target_category
    label_name = label_prefix + 'pos_dep'
    
    print("disease pairs to classify: {}".format(len(pairs)))
     #2. Get embeddings for the pairs
    
    emb = read_embedding_file(embedding_file)
    pairs['source_emb'] = pairs['source_cui'].apply(lambda x: emb[x])
    pairs['target_emb'] = pairs['target_cui'].apply(lambda x: emb[x])
    print("finished reading embeddings")
    
    # Get encoding for the pairs
    emb_for_enc = read_embedding_file(emb_file_for_encode)
    pairs['source_emb_for_enc'] = pairs['source_cui'].apply(lambda x: emb_for_enc[x])
    pairs['target_emb_for_enc'] = pairs['target_cui'].apply(lambda x: emb_for_enc[x])
    pairs['pair_emb'] = (pairs['source_emb_for_enc'] + pairs['target_emb_for_enc'])/2
    pair_emb_field_name = 'pair_emb'
            
    #4. cross validation
    #shuff = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    all_tests = None
    CV_indices = pickle.load(open("CV_indices151.pickle","rb"))
    accs = []
    aucs = []
    
    for train_index, test_index in CV_indices:
        train = pairs.loc[train_index]
        test = pairs.loc[test_index]
        print("train index: {} train: {} test_index: {} test: {}".format(len(train_index), len(train), len(test_index), len(test)))
        train = train[(train[label_prefix+'1'] >= 30) & (train[label_prefix+'2'] >=30)]
        test = test[(test[label_prefix+'1'] >= 30) & (test[label_prefix+'2'] >=30)]
        print("after filter by number of patients with each disease:\ntrain index: {} train: {} test_index: {} test: {}".format(len(train_index), len(train), len(test_index), len(test)))
                
        before_enc_time = time.time()
        encX_train = encode_from_df(train, pair_emb_field_name, autoenc, encoded_field=None)
        encX_test = encode_from_df(test, pair_emb_field_name, autoenc, encoded_field=None)
        print("finished creating X and y and possibly encoding in {} secs".format(time.time()-before_enc_time))
        
        X_train_with_enc, y_train = dataframe_to_x_and_y(train, label_name, encX_train, mode='enc_emb')
        X_test_with_enc, y_test = dataframe_to_x_and_y(test, label_name, encX_test, mode='enc_emb')
        
        X_train_no_enc, _ = dataframe_to_x_and_y(train, label_name, encX_train, mode='emb')
        X_test_no_enc, _ = dataframe_to_x_and_y(test, label_name, encX_test, mode='emb')
        
        
        def find_preferred_model(row):
            label = row[label_name]
            if label == 0:
                if row['with_enc_pred_prob'] < row['no_enc_pred_prob']:
                    return 'with_enc'
                return 'no_enc'
            if label == 1:
                if row['with_enc_pred_prob'] < row['no_enc_pred_prob']:
                    return 'no_enc'
                return 'with_enc'
            
        def merge_results(row):
            if row['chosen_model'] == 'with_enc':
                return row['with_enc_pred_prob']
            if row['chosen_model'] == 'no_enc':
                return row['no_enc_pred_prob']
        
        #5. Train classifier
        if model_name == 'LogisticRegression':
            model_enc = LogisticRegression(random_state=0)
            model_enc.fit(X_train_with_enc, y_train)
            
            model_noenc = LogisticRegression(random_state=0)
            model_noenc.fit(X_train_no_enc, y_train)
            train['with_enc_pred_prob'] = model_enc.predict_proba(X_train_with_enc)[:, 1]
            train['no_enc_pred_prob'] = model_noenc.predict_proba(X_train_no_enc)[:, 1]
            test['with_enc_pred_prob'] = model_enc.predict_proba(X_test_with_enc)[:, 1]
            test['no_enc_pred_prob'] = model_noenc.predict_proba(X_test_no_enc)[:, 1]

        elif model_name == 'XGBoost':
            model_enc = xgb.XGBClassifier(objective="binary:logistic")
            model_enc.fit(X_train_with_enc, y_train)
            
            model_noenc = xgb.XGBClassifier(objective="binary:logistic")
            model_noenc.fit(X_train_no_enc, y_train)
            
            train['with_enc_pred_prob'] = model_enc.predict_proba(X_train_with_enc)[:, 1]
            train['no_enc_pred_prob'] = model_noenc.predict_proba(X_train_no_enc)[:, 1]
            test['with_enc_pred_prob'] = model_enc.predict_proba(X_test_with_enc)[:, 1]
            test['no_enc_pred_prob'] = model_noenc.predict_proba(X_test_no_enc)[:, 1]
            
        elif model_name == 'NN':
            model_enc = get_NN_model(X_train_with_enc.shape[1])
            model_enc.fit(X_train_with_enc, y_train, epochs=EPOCHS_FOR_NN_CLASSIFIER, batch_size=10, verbose=1)

            model_noenc = get_NN_model(X_train_no_enc.shape[1])
            model_noenc.fit(X_train_no_enc, y_train, epochs=EPOCHS_FOR_NN_CLASSIFIER, batch_size=10, verbose=1)
            
            train['with_enc_pred_prob'] = model_enc.predict(X_train_with_enc)
            train['no_enc_pred_prob'] = model_noenc.predict(X_train_no_enc)
            test['with_enc_pred_prob'] = model_enc.predict(X_test_with_enc)
            test['no_enc_pred_prob'] = model_noenc.predict(X_test_no_enc)
            
            
        train['preferred_model'] = train.apply(find_preferred_model, axis=1)
        decisionX = categories_into_matrix(train, cat_to_index)
        #decisionX = X_train_no_enc
        decisionY = train['preferred_model'].values
        #decision_maker = LogisticRegression(random_state=0)
        decision_maker = get_NN_model(decisionX.shape[1])
        decision_maker.fit(decisionX, decisionY, epochs=EPOCHS_FOR_NN_CLASSIFIER)
        train['chosen_model'] = decision_maker.predict(decisionX)
        
        # how good is the model selection?
        print("train preferred model counts: \n {}".format(train['preferred_model'].value_counts()))
        print("train chosen model counts: \n {}".format(train['chosen_model'].value_counts()))
        print("decision maker acc: {}".format(np.sum(train['chosen_model'] == train['preferred_model'])/len(train)))
        
        test['chosen_model'] = decision_maker.predict(categories_into_matrix(test, cat_to_index))
        #test['chosen_model'] = decision_maker.predict(X_test_no_enc)
        
        
        train['hybrid_pred_prob'] = train.apply(merge_results, axis=1)
        test['hybrid_pred_prob'] = test.apply(merge_results, axis=1)
        
        # Find a threshold, based on train, optimizing for FPR-TPR.
        fpr, tpr, thresh = roc_curve(y_train, train['hybrid_pred_prob'])
        chosen_thresh = thresh[np.argmax(tpr-fpr)]
        
        test_readable = test[['source', 'target','source_cui', 'target_cui',
                              label_name, label_prefix+"1",
                              label_prefix+"2", label_prefix+"both", 
                              label_prefix+"none", label_prefix+"pval",
                              'with_enc_pred_prob', 'no_enc_pred_prob', 
                              'chosen_model',
                              'hybrid_pred_prob', 'source_category', 'target_category']]
        if all_tests is None:
            all_tests = test_readable
        else:
            all_tests = pd.concat([all_tests, test_readable])
        
        # how many positives?
        pred_test = test['hybrid_pred_prob'] > chosen_thresh
        print("positives in train: {}, in test:{}, in prediction: {}".format(
                sum(y_train)/len(y_train),
                sum(y_test)/len(y_test),
                sum(pred_test)/len(pred_test)))
        acc = sum((y_test == pred_test))/ len(y_test)
        auc = roc_auc_score(y_test, test['hybrid_pred_prob'])
        accs.append(acc)
        aucs.append(auc)
    print("avg acc: {}, avg auc: {}, desc: {}".format(
            np.mean(accs),
            np.mean(aucs),
            '{}_pred_prob'.format(model_desc)))

    return accs, aucs, all_tests



def count_patient_pairs_in_each_category(pairs_folder,male_emb_file, female_emb_file):
    # first: read the CUIs and categories
    cat_df = pd.read_csv('E:\Shunit\cuis_filtered_glove_single1_nodes_with_categories_plia.csv')
    cat_df = cat_df.dropna(axis=0, subset=['category'])
    cat_df['category'] = cat_df['category'].apply(lambda x: x.split("/"))
    categories = list(union(cat_df['category'].values))
    
    cui_to_patient_pairs = defaultdict(int)
    
    trans = Translator()
    male_emb = read_embedding_file(male_emb_file)
    female_emb = read_embedding_file(female_emb_file)
    
    def count_patients_per_cui(row):
        icd9s = row['male_diseases'] + row['female_diseases']
        cuis = trans.many_icd9s_to_cuis(icd9s)
        for cui in cuis:
            if cui not in male_emb and cui not in female_emb:
                continue
            cui_to_patient_pairs[cui] += 1
        
    pairs_files = [os.path.join(pairs_folder, x) for x in os.listdir(pairs_folder)]
    for pf in pairs_files:
        df = pd.read_csv(pf, index_col=None, sep='\t')
        df['male_diseases'] = df['male_diseases'].apply(literal_eval)
        df['female_diseases'] = df['female_diseases'].apply(literal_eval)
        df.apply(count_patients_per_cui, axis=1)
    # sum all cui counts into category counts
    cat_to_patient_pairs = defaultdict(int)
    for cat in categories:
        cuis = cat_df[cat_df['category'].apply(lambda x: cat in x)]['cui'].values
        for cui in cuis:
            cat_to_patient_pairs[cat] += cui_to_patient_pairs[cui]
    return cat_to_patient_pairs

########################### debug Varda's problem ###################

def count_icd_appearances(diags, set1=None, set2=None):
    # if not given set1 and set2, counts appearances for each icd9 in the data
    # otherwise, counts people with diseases from set1, from set2, and people with both.
    icds_in_data = defaultdict(lambda: {'M':0, 'F':0})
    c = {1:0, 2:0, 'both':0}
    
    def process_row(set_of_icd9s, sex):
        got1, got2 = False, False
        for icd9 in set_of_icd9s:
            if set1 is None:
                icds_in_data[icd9][sex] += 1
            else:
                if icd9 in set1:
                    got1 = True
                if icd9 in set2:
                    got2 = True
        if got1 and got2:
            c['both'] += 1
        elif got1:
            c[1] += 1
        elif got2:
            c[2] += 1

    t = time.time()
    diags['M'].icd9.apply(lambda x: process_row(x, 'M'))
    print("finished going over men in {} seconds".format(time.time()-t))
    t = time.time()
    diags['F'].icd9.apply(lambda x: process_row(x, 'F'))
    print("finished going over women in {} seconds".format(time.time()-t))
    if set1 is None:
        return icds_in_data
    print("{}: {}\n{}: {}\npeople who have codes from both sets : {}".format(
            set1, c[1], set2, c[2], c['both']))
    
def calculate_comorbidity_similarity(disease1=None, disease2=None, label_name='W_pos_dep', emb_file='E:\\Shunit\\female2_emb.tsv'):
    emb = read_embedding_file(emb_file)
    pairs = pd.read_csv(DISEASE_PAIRS_FILE, index_col=0)
    # how many pairs do both diseases agree on?
    # on how many pairs do they disagree?
    all_diseases = set(pd.concat([pairs['source_cui'].rename({'source_cui':'cui'}),
                              pairs['target_cui'].rename({'target_cui':'cui'})]).values)


    if disease1 is not None:
        cui_pairs = [disease1, disease2]
    else:
        cui_pairs = pairs[['source_cui', 'target_cui']].values
    cos_sim = []
    comorbidity_sim = []
    
    for d1, d2 in tqdm(cui_pairs):
        cos_sim.append(cosine_similarity(np.array([emb[d1], emb[d2]]))[0][1])
        count_agree = 0
        for other in all_diseases:
            if other in [d1, d2]:
                continue
            d1_says = pairs[((pairs['source_cui'] == other) & (pairs['target_cui'] == d1)) | ((pairs['source_cui'] == d1) & (pairs['target_cui'] == other))][label_name].values
            d2_says = pairs[((pairs['source_cui'] == other) & (pairs['target_cui'] == d2)) | ((pairs['source_cui'] == d2) & (pairs['target_cui'] == other))][label_name].values
            if len(d1_says) != 1 or len(d2_says) != 1:
                print(f"problem with {other}")
            if d1_says == d2_says:
                count_agree += 1
        comorbidity_sim.append(count_agree/(len(all_diseases)-2))
        if disease1 is not None:
            print(f"{d1} and {d2} agree on {count_agree}/{len(all_diseases)-2} comorbidities.")
    plt.scatter(cos_sim, comorbidity_sim)
    plt.show()
    
    
####################################################
    
def aucdiff_vs_female_fraction():
    # TODO: add here correlation(ff, auc diff) for each age group
    # and remove the respiratory outlier
    with_prevalence = pd.read_csv("E:\\Shunit\\female_vs_female_male_enc_by_disease_by_age_after_merge.csv", index_col=0)
    map_age_group = {'W_18-30_': 1, 'W_30-50_':2, 'W_50-70_':3, 'W_>70_':4}
    with_prevalence['age_group_as_num'] = with_prevalence['age_group'].apply(lambda x: map_age_group[x])
    with_prevalence['female fraction'] = pd.to_numeric(with_prevalence['female fraction'], downcast="float")
    fig, ax = plt.subplots()
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    for age_group in map_age_group:
        df = with_prevalence[with_prevalence['age_group'] == age_group]
        print(len(df))
        ax.scatter(df['female fraction'], df['auc diff'],
                label=age_group, c=df['age_group_as_num'])
    ax.legend()
    plt.show()

    
        
if __name__=="__main__":
    pass

