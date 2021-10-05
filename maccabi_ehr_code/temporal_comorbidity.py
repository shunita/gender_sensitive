# -*- coding: utf-8 -*-
"""
Temporal comorbidity

Created on Thu Oct  1 17:28:06 2020

@author: shunit.agmon
"""

import time
import os
import random
from ast import literal_eval
import pickle
from matplotlib import pyplot as plt
import numpy as np
import scipy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.sparse import csr_matrix, lil_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from read_utilities import FOR_TRANSLATION, FOR_COMORBIDITY, FRAG_DIR
from read_utilities import DIAGNOSIS_FILES, DIAGNOSIS_FIELD_NAMES, Translator
from read_utilities import read_embedding_file, read_patients_with_measurements, make_embedding_matrix

#from verification import classify

from tqdm import tqdm
tqdm.pandas()


#FRAG_DIR = "E:\\Shunit\\temporal_comorbidity"
NUM_FRAGS = 4

TRAIN = 0
TEST = 1
VALIDATION = 2


# preprocess
def read_diagnosis_with_time(debug=False, diag_to_index=None):
    s = time.time()
    dropped = 0
    trans = Translator()

    def year_and_diag(gb_group):
        ret = []
        for i, r in gb_group.iterrows():
            year = int(str(r['DATE_DIAGNOSIS'])[:4])
            cuis = trans.maccabi_or_icd9_to_cui(r['DIAGNOSIS_CODE'])
            for cui in cuis:
                ret.append((year, cui))
        # possibly turn the CUIs to indexes already to save space
        if diag_to_index is not None:
            ret = [(year, diag_to_index[cui]) for year, cui in ret if cui in diag_to_index]
        return sorted(list(set(ret)), key=lambda x: x[0])
    
    def process_gb(gb, frag_id):
        res = {}
        for rand_id, subset in tqdm(gb[['DATE_DIAGNOSIS', 'DIAGNOSIS_CODE']]):
            years_and_diags = year_and_diag(subset)
            if len(years_and_diags) == 0:
                continue
            res[rand_id] = years_and_diags
        with open(f'E://Shunit/temporal_comorbidity/gb_temp_{frag_id}.tsv', 'w') as f:
            f.write("RANDOM_ID\tDIAG_AND_YEAR\n")
            for rand_id in res:
                f.write("{}\t{}\n".format(rand_id, res[rand_id]))
    
    def combine_lists_of_year_diag_pairs(list_of_lists):
        res = []
        for lst in list_of_lists:
            res.extend(lst)
        # if a patient has the same diagnosis twice in one year, we would only see one.
        res = list(set(res))
        return sorted(res, key=lambda x: x[0])

    num_files = len(DIAGNOSIS_FILES)
    num_rows = None
    if debug:
        num_files = 4
        num_rows = 100000
    print(f"Debug mode: {debug}")
    for i in range(num_files):
        print("reading {}".format(DIAGNOSIS_FILES[i]))
        diag = pd.read_csv(DIAGNOSIS_FILES[i],
                   index_col=False,
                   header=None,
                   names=DIAGNOSIS_FIELD_NAMES,
                   encoding='latin', nrows=num_rows) #, dtype={'DATE_DIAGNOSIS': 'str'})
        print("finished reading")
        diag = diag.drop(labels=['CUSTOMERIDCODE',
                                 'DIAGNOSIS_TYP_CD', 
                                 'STATUS_DIAGNOSIS'], axis=1)
        before = len(diag)
        diag = diag.dropna(subset=['DATE_DIAGNOSIS'])
        after = len(diag)
        dropped += (before-after)
        
        # groups by random_id and process it into a new dataframe, with one row per random_id.
        # save the dataframe to a temporary file.
        process_gb(diag.groupby(by=['RANDOM_ID']), frag_id=i)
        print("finished grouping by random_id and writing new df to file.")

    print("dropped {} rows without date".format(dropped))
    diags = []
    for i in range(num_files):
        diag = pd.read_csv(f'E://Shunit/temporal_comorbidity/gb_temp_{i}.tsv', sep='\t')
        diag['DIAG_AND_YEAR'] = diag['DIAG_AND_YEAR'].apply(literal_eval)
        diags.append(diag)
    all_diags = pd.concat(diags)
    del diags, diag
    all_diags = all_diags.groupby(by=['RANDOM_ID'])['DIAG_AND_YEAR'].progress_apply(
            combine_lists_of_year_diag_pairs).reset_index()
    
    frags = np.array_split(all_diags, NUM_FRAGS)
    for i in range(len(frags)):
        frags[i].to_csv(f'E://Shunit/temporal_comorbidity/diags_by_patient{i}.csv')
    #return all_diags


def generate_anonimized_data(diags, patients, output_file):
    # diags is indexed by RANDOM_ID and has one column of a list of (diag, year) pairs
    # merge with patient data to get birth year
    wpatients = diags.merge(patients, on='RANDOM_ID')
    def year_to_age_in_diag_seq(row):
        delta = random.choice([-3,-2,-1,0,1,2,3])
        birth = int(str(row['CUSTOMER_BIRTH_DAT'])[:4]) + delta
        diag_seq = sorted([(int(y)-birth, d) for y, d in row['DIAG_AND_YEAR']],
                           key=lambda x: x[0])
        return diag_seq

    wpatients['age_and_diag'] = wpatients.progress_apply(year_to_age_in_diag_seq, axis=1)
    wpatients = wpatients.drop(['DIAG_AND_YEAR'], axis=1)
    wpatients.to_csv(output_file)
    
################# generate data #########################
def generate_data_pipeline():
    diag_to_index, index_to_diag = pickle.load(
            open(os.path.join(FRAG_DIR,"tokenization.pickle"), "rb"))
    read_diagnosis_with_time(debug=False, diag_to_index=diag_to_index)
    patients = read_patients_with_measurements()
    for i in range(NUM_FRAGS):
        diags = pd.read_csv(f'E://Shunit/temporal_comorbidity/diags_by_patient{i}.csv', index_col=0)
        diags['DIAG_AND_YEAR'] = diags['DIAG_AND_YEAR'].apply(literal_eval)
        generate_anonimized_data(diags, patients, f'E://Shunit/temporal_comorbidity/diags_by_patient{i}_age.csv')
    
################
        
def make_diag_time_series_from_anonymized(input_csv_file, 
                          min_years_for_patient, num_years_to_predict=3,
                          diag_to_index_starter=None,
                          index_to_diag_starter=None):
    # Now diags is indexed by RANDOM_ID and has one column of a list of (diag, year) pairs
    # merge with patient data to get birth year
    data = pd.read_csv(input_csv_file, index_col=0)
    data['age_and_diag'] = data['age_and_diag'].apply(literal_eval)
    data['diag_header'] = None
    data['diag_footer'] = None
    diag_to_index = {'empty':0}
    index_to_diag = ['empty']
    if diag_to_index_starter is not None:
        diag_to_index = diag_to_index_starter
    if index_to_diag_starter is not None:
        index_to_diag = index_to_diag_starter
    print("processing diags into time series")
    for i, row in tqdm(data.iterrows(), total=len(data)):
        diag_seq = row['age_and_diag']
        ages = [x[0] for x in diag_seq]
        # take only patients with <min_years_for_patient> year span.
        if len(ages)<1 or (max(ages)-min(ages)) < min_years_for_patient:
            continue
        last_age = max(ages)
        # <num_years_to_predict> last years are to be predicted.
        threshold_age = last_age - num_years_to_predict
        before, after = [], []
        before_diags = set()
        for age, diag in diag_seq:
            if diag in diag_to_index:
                ind = diag_to_index[diag]
            else:
                ind = len(index_to_diag)
                diag_to_index[diag] = ind
                index_to_diag.append(diag)
                
            if age < threshold_age:
                before.append(ind)
                before_diags.add(ind)
            else:
                if ind not in before_diags:
                    after.append(ind)
        data.at[i, 'diag_header'] = before
        data.at[i, 'diag_footer'] = after
    data = data.dropna(axis=0, subset=['diag_header', 'diag_footer'])
    data = data.drop(['DIAG_AND_YEAR'], axis=1)
    return data, diag_to_index, index_to_diag

#def make_diag_time_series(diags, patients, 
#                          min_years_for_patient, num_years_to_predict=3,
#                          diag_to_index_starter=None,
#                          index_to_diag_starter=None):
#    # Now diags is indexed by RANDOM_ID and has one column of a list of (diag, year) pairs
#    # merge with patient data to get birth year
#    wpatients = diags.merge(patients, on='RANDOM_ID')
#    
#    wpatients['diag_header'] = None
#    wpatients['diag_footer'] = None
#    diag_to_index = {'empty':0}
#    index_to_diag = ['empty']
#    if diag_to_index_starter is not None:
#        diag_to_index = diag_to_index_starter
#    if index_to_diag_starter is not None:
#        index_to_diag = index_to_diag_starter
#    print("processing comorbidities into time series")
#    for i, row in tqdm(wpatients.iterrows(), total=len(wpatients)):
#        birth = int(str(row['CUSTOMER_BIRTH_DAT'])[:4])
#        # calculate age and sort by it
#        diag_seq = sorted([(int(y)-birth, d) for y, d in row['DIAG_AND_YEAR']],
#                           key=lambda x: x[0])
#        ages = [x[0] for x in diag_seq]
#        
#        # take only patients with <min_years_for_patient> year span.
#        if len(ages)<1 or (max(ages)-min(ages)) < min_years_for_patient:
#            continue
#        last_age = max(ages)
#        # <num_years_to_predict> last years are to be predicted.
#        threshold_age = last_age - num_years_to_predict
#        before, after = [], []
#        before_diags = set()
#        for age, diag in diag_seq:
#            if diag in diag_to_index:
#                ind = diag_to_index[diag]
#            else:
#                ind = len(index_to_diag)
#                diag_to_index[diag] = ind
#                index_to_diag.append(diag)
#                
#            if age < threshold_age:
#                before.append(ind)
#                before_diags.add(ind)
#            else:
#                if ind not in before_diags:
#                    after.append(ind)
#        wpatients.at[i, 'diag_header'] = before
#        wpatients.at[i, 'diag_footer'] = after
#    wpatients = wpatients.dropna(axis=0, subset=['diag_header', 'diag_footer'])
#    wpatients = wpatients.drop(['DIAG_AND_YEAR'], axis=1)
#    return wpatients, diag_to_index, index_to_diag


def run_time_series_on_fragments(patients=None):
    if patients is None:
        patients = read_patients_with_measurements()
    
    diag_to_index, index_to_diag = None, None
    for i in range(NUM_FRAGS):
        # TODO: read something else here!
        frag_file = "temporal_frag{}.csv".format(i)
        print("processing {}".format(frag_file))
        df = pd.read_csv(os.path.join(FRAG_DIR, frag_file), index_col=0)
        df['DIAG_AND_YEAR'] = df['DIAG_AND_YEAR'].apply(literal_eval)
        data, diag_to_index, index_to_diag = make_diag_time_series_from_anonymized(
            df, patients, min_years_for_patient=5, num_years_to_predict=3,
            diag_to_index_starter=diag_to_index,
            index_to_diag_starter=index_to_diag)
        data.to_csv(os.path.join(FRAG_DIR, "ts_frag{}.csv".format(i)))
    pickle.dump((diag_to_index,index_to_diag), 
                open(os.path.join(FRAG_DIR, "tokenization.pickle"), "wb"))
    return diag_to_index, index_to_diag        


    


def make_label_matrix(labels, num_diags):
    # labels is an iterable of lists. Each list contains diagnoses indices.
    mat = lil_matrix((len(labels), num_diags))
    for i, label_list in tqdm(enumerate(labels), total=len(labels)):
        mat[i, label_list] = 1
    return mat.tocsr()


def build_model(emb_matrix, zero_weights, one_weights, lstm_units=128, trainable_emb='trainable', trainable_dim=10):
    print("building model with {} units".format(lstm_units))
    vocab_size=emb_matrix.shape[0]
    emb_size = emb_matrix.shape[1]
    model_input = layers.Input(shape=(None,))
    if trainable_emb in ['trainable', 'frozen']:
        should_train = trainable_emb == 'trainable'
        embedded = layers.Embedding(input_dim=vocab_size, 
                                output_dim=emb_size, 
                                embeddings_initializer=keras.initializers.Constant(emb_matrix),
                                trainable=should_train,
                                mask_zero=True)(
            model_input
        )
        print(f"emb_layer is {trainable_emb}, emb size: {emb_size}")
    else: # semi trainable
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
        embedded_pubmed = frozen_emb_layer(model_input)
        embedded_trainable = trainable_emb_layer(model_input)
        embedded = layers.Concatenate()([embedded_pubmed,embedded_trainable])
    
    output, state_h, state_c = layers.LSTM(lstm_units, return_state=True, name="encoder")(
            embedded
        )
    # From here connect to a dense layer, give it the state or the last output.
    # Using sigmoid instead of softmax, because there is no reason for all the 
    # label probabilities to sum to 1. (this is a multilabel problem).
    output1= layers.Dense(vocab_size, activation='sigmoid')(state_h)
    model = keras.Model(model_input, output1)
    print(model.summary())
    
    mets = MultilabelMetrics(one_weights)
    metrics = [
        mets.multilabel_acc,
        mets.pos_labels_pred,
        mets.pos_labels_true,
        mets.multilabel_precision,
        mets.multilabel_recall,
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        ]
    
#    keras.losses.binary_crossentropy
    model.compile(loss=weighted_multilabel_binary_crossentropy(zero_weights, one_weights),
                  optimizer='adam',
                  metrics=metrics)
    return model



def save_model_to_dir(model, emb_matrix, zero_weights, one_weights, lstm_units, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_weights(os.path.join(output_dir, 'model.weights'))
    with open(os.path.join(output_dir, 'additional_params.pickle'), 'wb') as out:
        pickle.dump((emb_matrix, zero_weights, one_weights, lstm_units), out)


def read_model_from_dir(input_dir):
    emb_matrix, zero_weights, one_weights, lstm_units = pickle.load(
                open(os.path.join(input_dir, 'additional_params.pickle'), 'rb'))
    model = build_model(emb_matrix, zero_weights, one_weights, lstm_units)
    model.load_weights(os.path.join(input_dir, 'model.weights'))
    return model

    

    
def build_set_based_model():
    # no embedding here?
    pass



def weighted_multilabel_binary_crossentropy(zero_weight_vec=None, one_weight_vec=None):
#    if zero_weight_vec is None:
#        one_weight_vec = 5
#        zero_weight_vec = 1
    def loss_func(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred) # shape: (batch_size, num_labels)
        # element-wise multiplication. the vector/scalar is broadcasted.
        weights = y_true * one_weight_vec + (1.0 - y_true)*zero_weight_vec # shape should be (batch_size, num_labels)
        middle = weights * bce # shape should be (batch_size, num_labels)
        return K.mean(K.mean(middle, axis=1)) # mean per sample, then batch mean (over samples).
    return loss_func
        
    
   

class MultilabelMetrics:
    def __init__(self, one_weight_vec=None):
        self.seen_in_train = None
        if one_weight_vec is not None:
            self.seen_in_train = (one_weight_vec > 0).astype(np.float32)
       
    def multilabel_acc(self, y_true, y_pred):
        rounded_pred = K.round(y_pred)
        #print("multilabel acc: shape of rounded_pred: {}".format(rounded_pred.shape))
        if self.seen_in_train is None:
            intersect_size = K.sum(rounded_pred*y_true, axis=1)
            union_size = K.sum(K.minimum(K.ones_like(y_true+y_pred), y_true+y_pred), axis=1)
        else:
            intersect_size = K.sum((rounded_pred*y_true)*self.seen_in_train, axis=1)
            union_size = K.sum(K.minimum(K.ones_like(y_true+y_pred), (y_true+y_pred)*self.seen_in_train), axis=1)
        return K.mean(intersect_size/union_size)
    
    def pos_labels_pred(self, y_true, y_pred):
        # sums the ones in each sample, then returns batch mean.
        return K.mean(K.sum(K.round(y_pred), axis=1))
    
    def pos_labels_true(self, y_true, y_pred):
        # sums the ones in each sample, then returns batch mean.
        return K.mean(K.sum(y_true, axis=1))
    
    def multilabel_precision(self, y_true, y_pred):
        rounded_pred = K.round(y_pred)
        if self.seen_in_train is not None:
            rounded_pred = rounded_pred*self.seen_in_train
        
        intersect_size = K.sum(rounded_pred*y_true, axis=1)
        pred_size = K.sum(rounded_pred, axis=1)
        denom = K.maximum(pred_size, K.ones_like(pred_size))
        # calculate precision per sample and return batch mean precision.
        return K.mean(intersect_size/denom)

    def multilabel_recall(self, y_true, y_pred):
        rounded_pred = K.round(y_pred)
        if self.seen_in_train is not None:
            rounded_pred = rounded_pred*self.seen_in_train
        intersect_size = K.sum(rounded_pred*y_true, axis=1)
        # assuming we don't have samples where the y_true is all zeros
        if self.seen_in_train is not None:
            true_size = K.sum(y_true*self.seen_in_train, axis=1)
        else:
            true_size = K.sum(y_true, axis=1)
        # calculate precision per sample and return batch mean precision.
        return K.mean(intersect_size/true_size)
    
    def avg_recall(self, ytrue, ypred):
        rounded_pred = K.round(ypred)
        # TP / (TP+ FN)
        true_pos = K.sum(rounded_pred*ytrue, axis=0) # for each disease
        false_neg = K.sum((1-rounded_pred)*ytrue, axis=0)
        recall_per_disease = true_pos/(true_pos + false_neg)
        return recall_per_disease @ self.seen_in_train
    
    def avg_precision(self, ytrue, ypred):
        rounded_pred = K.round(ypred)
        # TP / (TP+FP)
        true_pos = K.sum(rounded_pred*ytrue, axis=0) # for each disease
        false_pos = K.sum(rounded_pred*(1-ytrue), axis=0)
        precision_per_disease = true_pos/(true_pos + false_pos)
        print(f"ppd: {precision_per_disease.shape}, {type(precision_per_disease)}")
        return precision_per_disease.get_value() @ self.seen_in_train.get_value()
        
    def avg_auc(self, ytrue, ypred):
        return tf.py_func(roc_auc_score, (ytrue, ypred), tf.double)



def calculate_class_weights(ymatrix):
    # based on sklearn's compute_class_weight, 'balanced' version, implemented for multilabel.
    nsamples = ymatrix.shape[0]
    nclasses = 2
    ones_count = np.array(np.sum(ymatrix, axis=0))[0] # shape: (5414,)
    zeros_count = nsamples - ones_count # shape: (5414,)
    
    one_weights = (nsamples/nclasses)/ones_count
    # deal with empty columns - they should not affect the loss.
    one_weights[ones_count == 0] = 0
    zero_weights = (nsamples/nclasses)/zeros_count
    return zero_weights, one_weights

    
def from_dataframe_to_xy(data, assignment, vocab_size):
    if assignment is None:
        X = data['diag_header'].values
        y = data['diag_footer'].values
    else:
        X = data[data['secondary_assignment'] == assignment]['diag_header'].values
        y = data[data['secondary_assignment'] == assignment]['diag_footer'].values
    paddedX = pad_sequences(X, padding='post')
    ymatrix = make_label_matrix(y, vocab_size)
    return paddedX, ymatrix


def prettify_history(hist):
    s = []
    for key in hist:
        vals = ", ".join(["{:.3f}".format(x) for x in hist[key]])
        s.append("{}: {}".format(key, vals))
    return "\n".join(s)


def calculate_AUCs(ytrue, ypred, per_disease=False, threshold=0.5):
    # all labels that appear in Maccabi data are weighted the same. Labels that don't appear have weight=0.
    ones_count = np.array(np.sum(ytrue, axis=0))[0] # shape: (5414,)
    label_weights = np.ones(ones_count.shape[0])
    label_weights[ones_count == 0] = 0
    roc_aucs = []
    pr_aucs = []
    sensitivity = []
    specificity = []
    num_labels = ytrue.shape[1]
    is_ypred_sparse = ("sparse" in str(type(ypred)))
    for i in tqdm(range(num_labels)):
        if ones_count[i] == 0:
            roc_aucs.append(0)
            pr_aucs.append(0)
            sensitivity.append(0)
            specificity.append(0)
        else:
            ytrue_column = ytrue[:, i].toarray()
            if is_ypred_sparse:
                ypred_column = ypred[:, i].toarray()
            else:
                ypred_column = ypred[:, i]
            roc_aucs.append(roc_auc_score(ytrue_column, ypred_column))
            precision, recall, thresh = precision_recall_curve(ytrue_column, ypred[:, i])
            pr_aucs.append(auc(recall, precision))
            
            rounded_pred = (ypred_column > threshold).astype(np.float32)
            positive = np.sum(ytrue_column)
            true_pos = (rounded_pred @ ytrue_column).squeeze()
            negative = len(ytrue_column) - positive
            true_neg = ((1 - rounded_pred) @ (1-ytrue_column)).squeeze()
            sensitivity.append(true_pos/positive)
            specificity.append(true_neg/negative)
    weight_sum = np.sum(label_weights)
    roc_aucs = np.array(roc_aucs)
    avg_roc_auc = (roc_aucs @ label_weights)/weight_sum
    pr_aucs = np.array(pr_aucs)
    avg_pr_auc = (pr_aucs @ label_weights)/weight_sum
    sensitivity = np.array(sensitivity)
    print(f'sens shape: {sensitivity.shape}')
    print(f'label_weights shape: {label_weights.shape}')
    avg_sensitivity = (sensitivity @ label_weights)/weight_sum
    specificity = np.array(specificity)
    avg_specificity = (specificity @ label_weights)/weight_sum
    
    print("avg roc auc: {}, avg PR auc: {}\nthreshold {} avg sensitivity: {}, avg specificity: {}".format(
            avg_roc_auc, avg_pr_auc, 
            threshold, avg_sensitivity, avg_specificity))
    
    if per_disease:
        return roc_aucs, pr_aucs, sensitivity, specificity

    return avg_roc_auc, avg_pr_auc, avg_sensitivity, avg_specificity

def sensitivity_specificity_pairs(ytrue, ypred):
    sens = []
    spec = []
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        _,_,avg_sensitivity, avg_specificity = calculate_AUCs(ytrue, ypred, per_disease=False, threshold=thresh)
        sens.append(avg_sensitivity)
        spec.append(avg_specificity)
    return sens, spec

def plot_sens_spec_multiple_models(models_and_descs, X, ytrue):
    plt.rcParams['figure.figsize'] = 10,10
    print("plot_sens_spec_multiple_models")
    for model, desc in models_and_descs:
        ypred = model.predict(X)
        sens, spec = sensitivity_specificity_pairs(ytrue, ypred)
        print(f"desc: {desc} sens: {sens} spec: {spec}")
        plt.plot(sens, spec, label=desc)
    plt.legend(loc=0)
    plt.grid(which='both')
    plt.xlabel("sensitivity")
    plt.ylabel("specificity")
    plt.title("sensitivity-specificity curve")
    plt.show()

    
def AUC_per_disease_df(ytrue, ypred, index_to_diag, desc):
    roc_aucs, pr_aucs, sense, spec = calculate_AUCs(ytrue, ypred, per_disease=True, threshold=0.5)
    pos_rates = np.array(np.mean(ytrue, axis=0))[0] # shape=(5414,)
    num_patients = pos_rates * ytrue.shape[0]
    cui2name = pd.read_csv('E:\\Shunit\\code_mappings\\cui_to_name2118.csv', index_col=0).to_dict(orient='dict')['name']
    rows = []
    for i in range(len(index_to_diag)):
        cui = index_to_diag[i]
        if cui not in cui2name:
            name = cui
        else:
            name = cui2name[cui]
        rows.append({'name': name, 'cui': cui, 'pos_rate': pos_rates[i], 'num_patients': num_patients[i], 
                     f'{desc}_ROC_AUC': roc_aucs[i], f'{desc}_PR_AUC': pr_aucs[i],
                     f'{desc}_sensitivity': sense[i], f'{desc}_specificity': spec[i]})
    return pd.DataFrame.from_records(rows)


def AUC_per_disease_multiple_models(models_and_descs, X, ytrue, index_to_diag, output_file):
    combined = None
    for model, desc in models_and_descs:
        ypred = model.predict(X)
        df = AUC_per_disease_df(ytrue, ypred, index_to_diag, desc)
        if combined is None:
            combined = df
        else:
            df = df.drop(columns=['name', 'pos_rate', 'num_patients'])
            combined = combined.merge(df, on='cui')
    combined.to_csv(output_file)
    return combined
    
    
    
    

def diags_with_emb(seq, embs_for_filter, index_to_diag):
    # filter the diags that are missing from one of the embeddings.
    res = []
    for x in seq:
        addx = True
        for i in range(len(embs_for_filter)):
            if index_to_diag[x] not in embs_for_filter[i]:
                addx = False
        if addx:
            res.append(x)
    return res


def workflow(emb_file_template, gender_filter=None, units=100, epochs=10,
             trainable_emb='semi', # one of 'trainable', 'frozen', 'semi'
             trainable_size=10,
             output_dir="E:\\Shunit\\temporal_comorbidity\\specific_run"):
    neutral_emb_file = emb_file_template.format("neutral")
    female_emb_file = emb_file_template.format("female")
    male_emb_file = emb_file_template.format("male")

    # read tokenization info
    diag_to_index, index_to_diag = pickle.load(
            open(os.path.join(FRAG_DIR,"tokenization.pickle"), "rb"))
    
    data = read_temporal_records_not_seen_by_comorbidity_classifier(
            emb_file_template, index_to_diag)
    
    if gender_filter is not None:
        data = data[data['SEX'] == gender_filter]
        print("working only on {}".format(gender_filter))

    # read each embedding file into a matrix
    print("making embedding matrices")
    neutral_emb_matrix = make_embedding_matrix(
            index_to_diag, neutral_emb_file,
            os.path.join(FRAG_DIR, "diags_missing_from_neutral.txt"))
    female_emb_matrix = make_embedding_matrix(
            index_to_diag, female_emb_file,
            os.path.join(FRAG_DIR, "diags_missing_from_female.txt"))
    male_emb_matrix = make_embedding_matrix(
            index_to_diag, male_emb_file,
            os.path.join(FRAG_DIR, "diags_missing_from_male.txt"))
    print("finished making embedding matrices")
    
    # separation by assignment, padding (for later masking) and matrix building
    vocab_size = len(diag_to_index)
    Xtrain, ytrain = from_dataframe_to_xy(data, TRAIN, vocab_size)
    Xtest, ytest = from_dataframe_to_xy(data, TEST, vocab_size)

    print("converted data to X and y. train: {}, test: {}".format(
            len(Xtrain), len(Xtest)))
    zero_weights, one_weights = calculate_class_weights(ytrain)
    print("calculated class weights according to ytrain")
    # build a model and initialize it with the embedding matrix and the custom loss

    if trainable_emb == 'trainable':
        suffix = '_trainable'
    elif trainable_emb == 'semi':
        suffix = f'_semi_trainable{trainable_size}'
    else:
        suffix = ''
    embs = [
            ('female'+suffix, female_emb_matrix),
            ('neutral'+suffix, neutral_emb_matrix), 
            #('male'+suffix, male_emb_matrix)
            ]
    emb_size = embs[0][1].shape[1]
    print("working with embedding size: {}".format(emb_size))

    results = {}
    for desc, emb_matrix in embs:
        model = build_model(emb_matrix, zero_weights, one_weights,
                            lstm_units=units, trainable_emb=trainable_emb, trainable_dim=10)    
        history = model.fit(Xtrain, ytrain.toarray(),
                            #validation_data=(Xtest, ytest), 
                            batch_size=32, epochs=epochs) # verbose=0
        results[desc] = {'hist': history.history, 'model': model}
        save_dir = os.path.join(output_dir, f'{desc}_model')
        save_model_to_dir(model, emb_matrix, zero_weights, one_weights, units, save_dir)
        print("***************Finished {}".format(desc))
    models_and_descs = [(results[k]['model'], k) for k in results]
    combined = AUC_per_disease_multiple_models(
            models_and_descs, 
            Xtest, ytest, index_to_diag, 
            os.path.join(output_dir, 'results_by_disease.csv'))
    #pretty_print_training_metrics(results)
    return results, combined
    
def pretty_print_training_metrics(models_and_hists):
    versions = list(models_and_hists.keys())
    metrics = list(models_and_hists[versions[0]]['hist'].keys())
    for metric in metrics:
        print(metric)
        print(",".join(versions))
        for i in range(10):
            print(",".join([str(models_and_hists[version]['hist'][metric][i]) for version in versions]))

    
def read_temporal_records_not_seen_by_comorbidity_classifier(emb_file_template, index_to_diag):
    patients = read_patients_with_measurements(read_from_file=True)
    data = pd.read_csv(os.path.join(FRAG_DIR, "combined_with_assignment.csv"),
                       index_col=0)
    data = data.drop(columns = ['SEX', 'age', 'smoking', 'assignment'])
    data = data.merge(patients, on='RANDOM_ID')
    data = data[data['assignment'] == FOR_TRANSLATION]
    data['diag_header'] = data['diag_header'].apply(literal_eval)
    data['diag_footer'] = data['diag_footer'].apply(literal_eval)
    
    # remove diags that don't have all three embeddings.
    neutral_emb_file = emb_file_template.format("neutral")
    female_emb_file = emb_file_template.format("female")
    male_emb_file = emb_file_template.format("male")
    embs_for_filter = [read_embedding_file(neutral_emb_file),
                   read_embedding_file(female_emb_file),
                   read_embedding_file(male_emb_file)]
    data['diag_header'] = data['diag_header'].apply(
            lambda seq: diags_with_emb(seq, embs_for_filter, index_to_diag))
    data['diag_footer'] = data['diag_footer'].apply(
            lambda seq: diags_with_emb(seq, embs_for_filter, index_to_diag))
    data['diag_header_len'] = data['diag_header'].apply(len)
    data['diag_footer_len'] = data['diag_footer'].apply(len)
    
    # remove patients without enough diags in header or footer
    before = len(data)
    data = data[data['diag_footer_len'] >= 1]
    print("removed {} samples without new diagnosis in footer".format(before-len(data)))
    before = len(data)
    data = data[data['diag_header_len'] > 1]
    print("removed {} samples with short header".format(before-len(data)))
    print("remaining samples: {}".format(len(data)))
    return data


########################## static model #######################################


#def run_static_classifier():
#    version = "male"
#    emb_file_template = "E:\\Shunit\\embs\\{}_cui2vec_style_w2v_no_min_count_shuffle_40_emb_filtered.tsv"
#    embedding_file = emb_file_template.format(version)
#    model = classify(embedding_file, label_prefix="W_", model_name="NN",
#                     model_desc="{}40_NN".format(version), mode='emb', cross_validation=False,
#                     read_cv_index=None, big_dataset=True, pairs=None,
#                     return_model=True)
#    print("got trained model")
#    emb = read_embedding_file(embedding_file)
#    print("read embedding file: {}".format(embedding_file))
#    diag_to_index, index_to_diag = pickle.load(
#            open(os.path.join(FRAG_DIR,"tokenization.pickle"), "rb"))
#    emb_matrix = make_embedding_matrix(index_to_diag, embedding_file, None)
#
#    # calculate in advance the probability of each disease pair according to the classifier
#    # then later only calculate the max for each patient, for each disease
#
#    s = time.time()
#    classifier_results = lil_matrix((len(index_to_diag), len(index_to_diag)))
#    for i in range(len(index_to_diag)):
#        
#        if i % 100 == 1:
#            print("done: {}/{} rows in {} seconds per disease".format(i, len(index_to_diag), (time.time()-s)/i))
#        diag1 = index_to_diag[i]
#        #print("diag1 {}:{}".format(i, diag1))
#        if diag1 not in emb:
#            continue
#        for j in range(i):
#            diag2 = index_to_diag[j]
#            #print("diag2 {}:{}".format(j, diag2))
#            if diag2 not in emb:
#                continue
#            X = np.concatenate((emb[diag1], emb[diag2]))
#            #print(X.shape)
#            classifier_results[i,j] = model.predict(X.reshape(1,X.shape[-1]))
#            # TODO: remove this line after the classifier knows the difference between source and target.
#            classifier_results[j,i] = classifier_results[i,j]
#    print("finished generating classifier results in {} seconds".format(time.time()-s))
#            
#    
#    data = read_temporal_records_not_seen_by_comorbidity_classifier(
#            emb_file_template, index_to_diag)
#    print("got temporal data")
#    ypred = []
#    def predict(diag_header):
#        if len(ypred)% 100 == 1:
#            print("ypred: {}".format(len(ypred)))
#        previous_diags = list(set(diag_header))
#        res = np.max(classifier_results[previous_diags, :], axis=0).todense()
#        # TODO: how to aggregate? sum (how to normalize)? max?
#        ypred.append(res)
#        
#    frags = np.array_split(data, 100)
#    ypreds = []
#    for frag_index in range(len(frags)):
#        print("frag {}/{}".format(frag_index, 100))
#        frag = frags[frag_index]
#        ypred = []
#        frag['diag_header'].apply(predict)
#        ypreds.append(ypred)
#    #data['diag_header'].progress_apply(predict)
#    print("generated predictions")
#    ypred = np.stack(ypred)
#    print("ypred: shape {}, type {}".format(ypred.shape, type(ypred)))
#    ytrue = make_label_matrix(data['diag_footer'].values, len(index_to_diag))
#    ytrue = ytrue[:, 1:] # remove the first column, it represents "empty" diag.
#    # Now evaluate: how did we do?
#    avg_roc_auc, avg_pr_auc = calculate_AUCs(ytrue, ypred)
#    print("{} AUROC: {}, AUC precision recall: {}".format(
#            version, avg_roc_auc, avg_pr_auc))
    
            
########################################################

    
        
