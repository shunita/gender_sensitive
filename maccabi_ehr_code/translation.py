# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from ast import literal_eval
from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse.linalg
import time
import pickle
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from collections import defaultdict
from keras.models import Model, Sequential
from keras.layers import Dense, Input, concatenate, Flatten


# for translation seq2seq model:
from keras.layers import LSTM, Embedding, RepeatVector, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import categorical_accuracy

from keras.optimizers import SGD, Adam, RMSprop
from keras import backend as K
from sklearn.preprocessing import normalize
from verification import classify
from keras.callbacks import LearningRateScheduler

from read_utilities import *


RASHAMIM_FILE = "E:\Techniyon\RASHAMIM_processed.csv"







def num_disease_buckets(number):
    if number<=10:
        return (int(number), int(number))
    if number<=50:
        if number % 10 == 0:
            lb = number-10
        else:
            lb = int(number/10) * 10
        return (lb+1, lb+10)
    if number <= 75:
        return (51,75)
    if number <= 100:
        return (76,100)
    return (101,350)
        
    
def get_maccabi_codes_with_embedding():
    nodes = pd.read_csv('E:\\Shunit\\cuis_filtered_glove_single1_min40_all_nodes.tsv', sep='\t')
    cuis = nodes['cui']
    trans = Translator()
    codes = trans.many_cuis_to_maccabi_or_icd9(cuis)
    return codes

def read_rashamim():
    rasham = pd.read_csv(RASHAMIM_FILE, index_col=0)


#CUSTOMERS_FILE = 'E:\RawData\customers.csv'

# 1. Create data for translation: find pairs of a man and a woman that have the same:
# age, medications, k shared diseases, smoking habit.

# 2. extract the diseases each of them has.

#patients = pd.read_csv(CUSTOMERS_FILE, index_col=0)


#
#def read_patients_with_measurements():
#    patients = read_customers(keep_fields=['RANDOM_ID', 'CUSTOMER_BIRTH_DAT', 'DATE_DEATH'])
#    patients['age'] = patients.apply(age_buckets, axis=1)
#    
#    medidot = pd.read_csv(MEDIDOT_FILE, encoding='latin')
#    smoking = medidot[medidot['Madad_Refui_CD'] == 40]
#    # TODO: handle 4 (unknown)
#    latest_smoking = smoking.sort_values('Date_Medida_Refuit').groupby(by='RANDOM_ID').tail(1)
#    latest_smoking['smoking'] = latest_smoking['Value1'].apply(int)
#    customers_smoking = patients.merge(latest_smoking[['RANDOM_ID', 'smoking']], how='inner', on='RANDOM_ID')
#    return customers_smoking[['RANDOM_ID', 'SEX', 'age', 'smoking']]


def process_diag_dataframe(df, patients, disease_set_field): 
    with_patients = df.merge(patients, on='RANDOM_ID', how='inner')
    with_patients['num_diseases'] = with_patients[disease_set_field].apply(len)
    with_patients = with_patients[with_patients['num_diseases'] > 1]
    with_patients['num_disease_bin'] = with_patients['num_diseases'].apply(
            num_disease_buckets)
    return with_patients
    

def create_matrix_from_df(df, disease_to_index, disease_set_field):
    start = time.time()
    matrix = lil_matrix((len(df), len(disease_to_index)), dtype=np.float)

    def matrix_update(row):
        disease_set = row[disease_set_field]
        disease_indices = [disease_to_index[disease] for disease in disease_set
                           if disease in disease_to_index]
        matrix[row.name, sorted(disease_indices)] = 1

    df.apply(matrix_update, axis=1)
    print("created {} row matrix in {} seconds".format(len(df), time.time()-start))
    start = time.time()
#    matrix = matrix.tocsr()
#    norm = scipy.sparse.linalg.norm(matrix, axis=1)
#    nonzero = norm > 0
#    norm = norm[:, np.newaxis]
#    matrix[nonzero] /= norm[nonzero]
    matrix = normalize(matrix, norm='l2', axis=1) # normalize each row and keep is sparse!
    print("normalized in {} seconds".format(time.time()-start))
    return matrix


def get_subset_from_df_and_reset_index(df, age, smoking, num_disease_range):
    bin_df = df[(df['age'] == age) & 
                (df['smoking'] == smoking) & 
                (df['num_disease_bin'] == num_disease_range)]
    bin_df1 = bin_df.reset_index(drop=True)
    return bin_df1
    

def make_patient_pairs(allow_repeat=True, allow_identical=True, 
                       M_patients=None, F_patients=None,
                       match_by_rasham=False, rasham=None, chronics=None):
    trans = Translator()
    if M_patients is None or F_patients is None:
        patients = read_patients_with_measurements()
        patients = patients[patients['assignment'] == FOR_TRANSLATION]
        diags = read_diagnosis(patients)
    disease_field = 'icd9'
    if M_patients is None:
        M_patients = process_diag_dataframe(diags['M'], patients, disease_field)
    if F_patients is None:
        F_patients = process_diag_dataframe(diags['F'], patients, disease_field)
    
    diseases = trans.all_icd9_list()
    disease_to_index = {diseases[i]: i for i in range(len(diseases))}
    
    if match_by_rasham:
        if rasham is None or chronics is None:
            rasham, chronics = read_rasham()
        M_patients = M_patients.merge(rasham[chronics+['RANDOM_ID']], on='RANDOM_ID', how='inner')
        F_patients = F_patients.merge(rasham[chronics+['RANDOM_ID']], on='RANDOM_ID', how='inner')
    
    age_groups = ['0-18', '18-30', '30-50', '50-70', '>70']
    #age_groups = ['>70']
    smoking_status = (1, 2, 3) # not including 4 - unknown
    num_disease_bins = ((2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
                        (9, 9), (10, 10),
                        (11, 20), (21, 30), (31, 40), (41, 50),
                        (51, 75),(76, 100),(101, 350))
    already_done = []
    # binning by age, smoking, num_diseases
    # matching by similarity of disease set
    for age in age_groups:
        for smoking in smoking_status:
            if (age, smoking) in already_done:
                continue
            f = open("pairs_smoke{}_age{}.tsv".format(smoking, age.replace("-","_").replace(">","over")), "w")
            f.write("male_id\tfemale_id\tmale_diseases\tnum_male_diseases\t"
                    "female_diseases\tnum_female_diseases\tsimilarity\tdisease_bin\n")
            
            for num_disease_range in num_disease_bins:
                Mbin = get_subset_from_df_and_reset_index(M_patients, age, smoking, num_disease_range)
                Fbin = get_subset_from_df_and_reset_index(F_patients, age, smoking, num_disease_range)
                
                print("\nage {}, smoking {} , num_diseases:{}: male patients:{}, female patients:{}".format(
                        age, smoking, num_disease_range, len(Mbin), len(Fbin)))
                if len(Mbin) == 0 or len(Fbin)==0:
                    continue
                if match_by_rasham:
                    M_matrix = Mbin[chronics].values
                    M_matrix = normalize(M_matrix, norm='l2', axis=1)
                    F_matrix = Fbin[chronics].values
                    F_matrix = normalize(F_matrix, norm='l2', axis=1)
                else:
                    # make a sparse disease vector for each patient. Normalize each vector.
                    M_matrix = create_matrix_from_df(Mbin, disease_to_index, disease_field)
                    F_matrix = create_matrix_from_df(Fbin, disease_to_index, disease_field)
          
                # for each male patient find the closest female patient
                participated_fem = set()
                start = time.time()
                for male_row_index, row in Mbin.iterrows():
                    if male_row_index % 1000 == 0:
                        print("iteration {}/{}, elapsed time: {}".format(male_row_index, len(Mbin), time.time()-start))
                    male_vector = M_matrix[male_row_index]
                    similarities = F_matrix @ male_vector.T # similarity of one man to each woman
                    if not allow_repeat:
                        similarities[list(participated_fem)] = 0
                    if not allow_identical:
                        # if the woman has exactly the same diseases - the input
                        # and output of the autoencoder are the same.
                        similarities[similarities > 0.9999] = 0
            
                    fem_row_index = np.argmax(similarities)
                    if match_by_rasham:
                        max_sim = similarities[fem_row_index]
                    else:
                        max_sim = similarities[fem_row_index].todense().item()
                    
                    if max_sim < 0.001: # no common diseases, similarity is 0
                        continue
                    if not allow_repeat:
                        participated_fem.add(fem_row_index)
                    
                    male_diseases = [d for d in row[disease_field] if d in disease_to_index]
                    female_diseases = [d for d in Fbin.iloc[fem_row_index][disease_field] if d in disease_to_index]
                    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                            row['RANDOM_ID'],
                            Fbin.iloc[fem_row_index]['RANDOM_ID'],
                            male_diseases,
                            len(male_diseases),
                            female_diseases,
                            len(female_diseases),
                            max_sim,
                            row['num_disease_bin']))
            f.close()

def count_common(row):
    return len(set(row['male_diseases']).intersection(set(row['female_diseases'])))

                
def read_pairs_for_autoencoder(female_emb_file, male_emb_file, 
                               pairs_folder=None, pairs_file=None, mode='M2F',
                               use_idf_weights=False, fraction_common=0.25):
    # mode in {'M2F', 'F2F'}, maybe later: 'F2M', 'M2M'
    if pairs_folder is None and pairs_file is None:
        print("No human pairs found for autoencoder training.")
        return
    if pairs_file is not None:
        pairs_files = [pairs_file]
    elif pairs_folder is not None:
        pairs_files = [os.path.join(pairs_folder, x) for x in os.listdir(pairs_folder)]
    
    # read_embeddings
    fem_emb = read_embedding_file(female_emb_file)
    male_emb = read_embedding_file(male_emb_file)
    trans = Translator()
    if use_idf_weights:
        IDFs = get_IDF_dict()
    x = []
    y = []
#    disease_counts = defaultdict(int)
    
    def calc_average_embedding(icd9s, emb_dict):
        cuis = trans.many_icd9s_to_cuis(icd9s)
        filtered = [disease for disease in cuis if disease in emb_dict]
        if len(filtered) < 2:
            return None
        if use_idf_weights:
            filtered = [disease for disease in filtered if disease in IDFs]
            vec = np.average([emb_dict[cui] for cui in filtered], 
                         axis=0, 
                         weights=[IDFs[cui] for cui in filtered])
        else:
            vec = np.mean([emb_dict[cui] for cui in filtered], axis=0)
        return vec
        
    # read pairs
    for pf in pairs_files:
        df = pd.read_csv(pf, index_col=None, sep='\t')
        df['male_diseases'] = df['male_diseases'].apply(literal_eval)
        df['female_diseases'] = df['female_diseases'].apply(literal_eval)
        df['common'] = df.apply(count_common, axis=1)
        df['union_size'] = df['num_male_diseases'] + df['num_female_diseases'] - df['common']
        df['fraction_common'] = df['common']/df['union_size']
        df.to_csv(pairs_file, sep='\t')
        df1 = df[(df['fraction_common'] > fraction_common) & (df['num_male_diseases']>1) & (df['num_female_diseases']>1)]
        print("{}: # male-female pairs: {}".format(pf, len(df1)))
              
        for i, r in df1.iterrows():
            # transform icd9 to CUIs
            mvec = calc_average_embedding(r['male_diseases'], male_emb)
            fvec = calc_average_embedding(r['female_diseases'], fem_emb)
#            md = trans.many_icd9s_to_cuis(r['male_diseases'])
#            md = [disease for disease in md if disease in male_emb]
#            
#            fd = trans.many_icd9s_to_cuis(r['female_diseases'])
#            fd = [disease for disease in fd if disease in fem_emb]
            
            if mvec is None or fvec is None:
                continue
            
            if mode == 'M2F' and mvec is not None and fvec is not None:
                # get each disease vector and average them
#                mvec = np.average([male_emb[cui] for cui in md], axis=0, weights=[IDFs[cui] for cui in md])
#                fvec = np.average([fem_emb[cui] for cui in fd], axis=0, weights=[IDFs[cui] for cui in fd])
                x.append(mvec)
                y.append(fvec)
            elif mode == 'F2F' and fvec is not None:
                #fvec = np.average([fem_emb[cui] for cui in fd], axis=0, weights=[IDFs[cui] for cui in fd])
                x.append(fvec)
                y.append(fvec)
            elif mode not in ('M2F', 'F2F'):
                print("Unsupported mode in read_pairs_for_autoencoder: {}".format(mode))
                return None, None
    x = np.stack(x)
    y = np.stack(y)
    return x,y
    

def lr_scheduler(epoch, lr):
    print("lr: {}".format(lr))
    if epoch < 10:
        return lr
    return lr*0.95


class AutoEncoder(object):
    def __init__(self, single_layer=True, input_size=300, code_size=None):
        self.input_size = input_size
        if code_size is None:
            code_size = int(input_size/2)
        
        input_patient = Input(shape=(self.input_size,))
        if single_layer:
            self.hidden_size = 0
            self.code_size = code_size
            self.code_layer_index = 1
            code = Dense(self.code_size, activation='relu')(input_patient)
            output_patient = Dense(self.input_size, activation='sigmoid')(code)
        else:
            self.hidden_size = int(code_size/2)
            self.code_size = code_size
            self.code_layer_index = 2
            hidden1 = Dense(self.hidden_size, activation='relu')(input_patient)
            code = Dense(self.code_size, activation='relu')(hidden1) # dense_2
            hidden2 = Dense(self.hidden_size, activation='relu')(code)
            # TODO: Why sigmoid?
            output_patient = Dense(self.input_size, activation='sigmoid')(hidden2)
        
        self.autoenc = Model(input_patient, output_patient)
        self.scheduler = LearningRateScheduler(lr_scheduler)
#        lr_schedule = optimizers.schedules.ExponentialDecay(
#                initial_learning_rate=1e-2, decay_steps=100, decay_rate=0.95)
#        opt = optimizers.Adam(learning_rate=lr_schedule)
        self.autoenc.compile(optimizer=Adam(), loss='mean_squared_error')
        # Save a part of the autoencoder for encoding only.
        code_layer = self.autoenc.get_layer(index=self.code_layer_index)
        self.encoder = Model(inputs=self.autoenc.input, outputs=code_layer.output)

    def train(self, X,Y, epochs=3):
        # x and y are of the same shape: each row is a vector of size <self.input_size>.
        self.autoenc.fit(X, Y, epochs=epochs, 
                         #callbacks=[self.scheduler]
                         )
        
    def encode(self, x):
        # x is a vector of size <self.input_size>.
        # E.g., an average of several disease embeddings.
        # It also works with a text representation of a vector.
        code = self.encoder(K.constant(np.matrix(x)))
        return K.get_value(code)
    
    def encode_matrix(self, x):
        return K.get_value(self.encoder(K.constant(x)))

    
def run_autoencoder():
    folder = 'E:\\Shunit\\'
    pairs_file = 'match_by_all_diseases\\pairs_smoke1_age50_70.tsv'
    pairs_folder = 'match_by_all_diseases_patient_separation'
    female_emb_file = folder + 'female2_emb.tsv'
    male_emb_file = folder + 'male2_emb.tsv'
    USE_IDF_WEIGHTS = True
    #encoding_file = folder + "cuis_filtered_glove_single1_pairs_151nodes_encoded_by_50_70_smokers_AE.csv"
    #encoding_file = folder + 'cuis_filtered_glove_single1_pairs_151nodes_encoded_by_all_AE.csv'
    print("reading people pairs for autoencoder: {}".format(pairs_file))
    #x, y = read_pairs_for_autoencoder(female_emb_file, male_emb_file, pairs_file=pairs_file)
    x, y = read_pairs_for_autoencoder(female_emb_file, male_emb_file,
                                      pairs_folder=pairs_folder, mode='M2F',
                                      use_idf_weights=USE_IDF_WEIGHTS)
    ae = AutoEncoder(single_layer=True, input_size=40, code_size=20)
    print("training autoencoder")
    ae.train(x, y, epochs=100)
    print("training classifier on autoencoder output")
    accs, aucs, all_test = classify(male_emb_file, "W_50-70_1_", "NN", 
                          model_desc="AE_50-70smokers_NN", autoenc=ae,
                          #encoding_file=encoding_file, 
                          use_idf_weights=USE_IDF_WEIGHTS)
    #accs, aucs = classify(male_emb_file, "W_", "NN", autoenc=ae)
    print('avg acc: {}, avg auc: {},\naccs:{}\naucs:{}'.format(
            np.mean(accs), np.mean(aucs), accs, aucs))
    
def batch_runs_with_autoencoder(pairs, df=None, model="NN", output_file=None):
    # shameless copy-paste from verification.py
    # model = "NN" or "LogisticRegression"
    model_to_str = {"NN":"NN", "LogisticRegression": "logreg"}
    #sizes = (8, 10, 20, 30, 40, 50)
    sizes = (8, 10, 20, 30, 40, 50)
    #sizes = (8,)
    human_pairs_folder = 'match_by_all_diseases_patient_separation'
    emb_file_template = 'E:\\Shunit\\embs\\{}_cui2vec_style_w2v_no_min_count_shuffle_{}_emb_filtered.tsv'
    res = {}
    for size in sizes:
        x, y = read_pairs_for_autoencoder(
                emb_file_template.format("female", size),
                emb_file_template.format("male", size), 
                pairs_folder=human_pairs_folder,
                pairs_file=None,
                mode='M2F',
                use_idf_weights=False, 
                fraction_common=0.25)
        ae_m2f = AutoEncoder(single_layer=True, input_size=size, code_size=int(size/2))
        ae_m2f.train(x, y, epochs=30)
        for version in ('neutral', 'female', 'male'):
            desc = "{}{}+enc{}to{}_{}".format(version, size, size, int(size/2), model_to_str[model])
            accs, aucs, all_test = classify(
                    emb_file_template.format(version, size),
                    "W_", model, model_desc=desc, 
                    autoenc=ae_m2f, mode='enc_emb',
                    emb_file_for_encoding=emb_file_template.format("male", size),
                    use_idf_weights=False, 
                    test_label_prefix='W_', cross_validation=True, 
                    read_cv_index='big_dataset_cui2vec_style_w2v_no_min_count_shuffle_CV.pickle', 
                    custom_test_file=None, filter_by_embedding_file=False, 
                    combine_mode='concat', pairs=pairs)
            if df is None:
                df = all_test
            else:
                df[desc+"_pred_prob"] = all_test[desc+"_pred_prob"]
            res[desc] = (np.mean(accs), np.mean(aucs))
            print("finished {}".format(desc))
            
        # and once without embedding - only encoding
        desc = "enc{}to{}_{}".format(size, int(size/2), model_to_str[model])
        accs, aucs, all_test = classify(
                emb_file_template.format(version, size),
                "W_", model, model_desc=desc, 
                autoenc=ae_m2f, mode='enc',
                emb_file_for_encoding=emb_file_template.format("male", size),
                use_idf_weights=False, 
                test_label_prefix='W_', cross_validation=True, 
                read_cv_index='big_dataset_cui2vec_style_w2v_no_min_count_shuffle_CV.pickle', 
                custom_test_file=None, filter_by_embedding_file=False, 
                combine_mode='concat', pairs=pairs)
        if df is None:
            df = all_test
        else:
            df[desc+"_pred_prob"] = all_test[desc+"_pred_prob"]
        res[desc] = (np.mean(accs), np.mean(aucs))
        print("finished {}".format(desc))
    if output_file is not None:
        df.to_csv(output_file)
    return res
    
    
def run_configurations():
    folder = 'E:\\Shunit\\'
    pairs_folder = 'match_by_all_diseases_patient_separation'
    female_emb_file = folder + 'female2_emb.tsv'
    male_emb_file = folder + 'male2_emb.tsv'
    model_name = "LogisticRegression"
    x, y = read_pairs_for_autoencoder(female_emb_file, male_emb_file,
                                      pairs_folder=pairs_folder, mode='M2F',
                                      use_idf_weights=False)
    ae_m2f = AutoEncoder(single_layer=True)
    ae_m2f.train(x, y, epochs=30)
    
    
    messages = {}
    for label_name, resfile in [("W_", "classify_res_female_logreg_patient_sep.csv"),
                                ("M_", "classify_res_male_logreg_patient_sep.csv"),
                                ("all_", "classify_res_neutral_logreg_patient_sep.csv")]:
        final_message = ""
        df = pd.read_csv(resfile, index_col=0)
        # only autoencoder
        model_desc = "M2F_singleAE100_logreg"
        accs, aucs, all_test = classify(
                    male_emb_file, label_name , model_name, model_desc,
                    autoenc=ae_m2f, emb_file_for_encoding=male_emb_file,
                    mode='enc', use_idf_weights=False)
        s = "\n{} avg acc: {} avg auc: {}\n".format(model_desc, np.mean(accs), np.mean(aucs))
        final_message += s
        df[model_desc+"_pred_prob"] = all_test[model_desc+"_pred_prob"]
        
        for emb, model_desc in [(male_emb_file, "M2F_singleAE100_concat_male_logreg"),
                                (female_emb_file, "M2F_singleAE100_concat_female_logreg")]:
            # autoencoder + embedding
            accs, aucs, all_test = classify(
                    emb, label_name, model_name, model_desc,
                    autoenc=ae_m2f, emb_file_for_encoding=male_emb_file,
                    mode='enc_emb', use_idf_weights=False)
            s = "\n{} avg acc: {} avg auc: {}\n".format(model_desc, np.mean(accs), np.mean(aucs))
            final_message += s
            df[model_desc+"_pred_prob"] = all_test[model_desc+"_pred_prob"]
            
            # random + embedding
            model_desc2 = model_desc.replace("M2F_singleAE","random")
            accs, aucs, all_test = classify(
                    emb, label_name, model_name, model_desc2,
                    autoenc=ae_m2f, emb_file_for_encoding=male_emb_file,
                    mode='emb_random', use_idf_weights=False)
            s = "\n{} avg acc: {} avg auc: {}\n".format(model_desc2, np.mean(accs), np.mean(aucs))
            final_message += s
            df[model_desc2+"_pred_prob"] = all_test[model_desc2+"_pred_prob"]
        df.to_csv(resfile.replace("sep", "sep1"))
        messages[label_name] = final_message
    for label_name in messages:
        print("label: {}".format(label_name))
        print(messages[label_name])
   
    
    
    
############## different translation model: seq2seq ##################

def read_sequences_for_translation(
        female_emb_file, male_emb_file, fraction_common=0.25, 
        pairs_folder=None, pairs_file=None):
    if pairs_folder is None and pairs_file is None:
        print("No human pairs found for autoencoder training.")
        return
    if pairs_file is not None:
        pairs_files = [pairs_file]
    elif pairs_folder is not None:
        pairs_files = [os.path.join(pairs_folder, x) for x in os.listdir(pairs_folder)]
    
    # read_embeddings
    fem_emb = read_embedding_file(female_emb_file)
    male_emb = read_embedding_file(male_emb_file)
    trans = Translator()
    max_m_len, max_f_len = 0,0
    men, women = [], []
    for pf in pairs_files:
        df = pd.read_csv(pf, index_col=None, sep='\t')
        df['male_diseases'] = df['male_diseases'].apply(literal_eval)
        df['female_diseases'] = df['female_diseases'].apply(literal_eval)
        df['common'] = df.apply(count_common, axis=1)
        df['union_size'] = df['num_male_diseases'] + df['num_female_diseases'] - df['common']
        df['fraction_common'] = df['common']/df['union_size']
        df1 = df[(df['fraction_common'] > fraction_common) & (df['num_male_diseases']>1) & (df['num_female_diseases']>1)]
        print("# male-female pairs: {}".format(len(df1)))
        
        for i,r in df1.iterrows():
            mcuis = trans.many_icd9s_to_cuis(r['male_diseases'])
            fcuis = trans.many_icd9s_to_cuis(r['female_diseases'])
            m_filtered = [disease for disease in mcuis if disease in male_emb]
            f_filtered = [disease for disease in fcuis if disease in fem_emb]
            # TODO: or maybe include these too?
            if len(m_filtered) < 2 or len(f_filtered) < 2:
                continue
            men.append(m_filtered)
            women.append(f_filtered)
            max_m_len = max(max_m_len, len(m_filtered))
            max_f_len = max(max_f_len, len(f_filtered))
    return men, women, max_m_len, max_f_len
            
    


class TranslationModel(object):

    def __init__(self, tokenizer, units, max_input_length, max_output_length, emb_matrix):
        model = Sequential()
        vocab_size = len(tokenizer.word_index) + 1
        model.add(Embedding(input_dim=vocab_size, output_dim=emb_size,
                            embeddings_initializer=keras.initializers.Constant(emb_matrix),
                            input_length=max_input_length, mask_zero=True))
        model.add(Bidirectional(LSTM(units), name="encoding_layer"))
        model.add(RepeatVector(max_output_length))
        # TODO: return_sequences? will we ever use it?
        # return_sequences means returning the outputs for each point in the sequence.
        model.add(Bidirectional(LSTM(units, return_sequences=True)))
        model.add(Dense(vocab_size, activation='softmax'))
        rms = RMSprop(lr=0.001)
        model.compile(rms, loss='sparse_categorical_crossentropy', metrics=[categorical_accuracy])
        
        self.tokenizer = tokenizer
        self.model = model
        self.max_input_len = max_input_length
        self.max_output_len = max_output_length
        # TODO use state instead of output?
        self.encoder = Model(inputs=self.model.input, outputs=model.get_layer(name="encoding_layer").output)
        
    def tokenize_sequences(self, list_of_sentences, maxlen):
        seq = self.tokenizer.texts_to_sequences(list_of_sentences)
        seq = pad_sequences(seq, maxlen=maxlen, padding='post')
        return seq
    
    def train(self, X, y, epochs=10):
        Xtrain = self.tokenize_sequences(X, self.max_input_len)
        ytrain = self.tokenize_sequences(y, self.max_output_len)
        return self.model.fit(Xtrain, ytrain.reshape(ytrain.shape[0], ytrain.shape[1], 1),
                              epochs=epochs, batch_size=32, verbose=0)
        #can use validation_data in fit
        
    def predict(self, d1, d2):
        return self.model.predict_classes(self.tokenize_sequences([[d1,d2]], maxlen=self.max_input_len))
        
    def encode(self, list_of_disease_lists):
        return self.encoder(self.tokenize_sequences(diseases_list, maxlen=self.max_input_len))
    
    
def run_translation_model():
    folder = 'E:\\Shunit\\'
    pairs_folder = 'match_by_all_diseases_patient_separation'
    female_emb_file = folder + 'female2_emb.tsv'
    male_emb_file = folder + 'male2_emb.tsv'
    # TODO: do we want only people with at least 2 diseases?
    men_corpus, women_corpus, max_men, max_women = read_sequences_for_translation(
            female_emb_file, male_emb_file, fraction_common=0.25, pairs_folder=pairs_folder)
    tokenizer = Tokenizer()
    # check if maybe we want to separate this to 2 tokenizers for some reason
    tokenizer.fit_on_texts(men_corpus+women_corpus)
    
    emb_matrix = make_embedding_matrix(tokenizer.index_word, female_emb_file, output_file=None)
    
    # TODO: number of units?
    tm = TranslationModel(tokenizer, 5, max_men, max_women, emb_matrix)
    tm.train(men_corpus, women_corpus)
    

############## different translation model: emb adjustment ##################


def read_sequences_for_emb_adjust(
        female_emb_file, male_emb_file, fraction_common=0.25, 
        pairs_folder=None, pairs_file=None):
    if pairs_folder is None and pairs_file is None:
        print("No human pairs found for autoencoder training.")
        return
    if pairs_file is not None:
        pairs_files = [pairs_file]
    elif pairs_folder is not None:
        pairs_files = [os.path.join(pairs_folder, x) for x in os.listdir(pairs_folder)]
    
    # read_embeddings
    fem_emb = read_embedding_file(female_emb_file)
    male_emb = read_embedding_file(male_emb_file)
    trans = Translator()
    max_m_len, max_f_len = 0,0
    men, women = [], []
    for pf in pairs_files:
        df = pd.read_csv(pf, index_col=None, sep='\t')
        df['male_diseases'] = df['male_diseases'].apply(literal_eval)
        df['female_diseases'] = df['female_diseases'].apply(literal_eval)
        df['common'] = df.apply(count_common, axis=1)
        df['union_size'] = df['num_male_diseases'] + df['num_female_diseases'] - df['common']
        df['fraction_common'] = df['common']/df['union_size']
        df1 = df[(df['fraction_common'] > fraction_common) & (df['num_male_diseases']>1) & (df['num_female_diseases']>1)]
        print("# male-female pairs: {}".format(len(df1)))
        
        for i,r in df1.iterrows():
            mcuis = trans.many_icd9s_to_cuis(r['male_diseases'])
            fcuis = trans.many_icd9s_to_cuis(r['female_diseases'])
            m_filtered = [disease for disease in mcuis if disease in male_emb]
            f_filtered = [disease for disease in fcuis if disease in fem_emb]
            # TODO: or maybe include these too?
            if len(m_filtered) < 2 or len(f_filtered) < 2:
                continue
            men.append(m_filtered)
            women.append(f_filtered)
            max_m_len = max(max_m_len, len(m_filtered))
            max_f_len = max(max_f_len, len(f_filtered))
    return men, women, max_m_len, max_f_len


class EmbAdjust(object):
    def __init__(self, emb_matrix):
        vocab_size, emb_size = emb_matrix.shape
        inputA = Input(shape=(1,))
        inputB = Input(shape=(1,))
        
        emb_layer = Embedding(input_dim=vocab_size, 
                         output_dim=emb_size, 
                         embeddings_initializer=keras.initializers.Constant(emb_matrix),
                         trainable=True,
                         mask_zero=False)
        
        #emb_layer = Dense(emb_size, activation='relu')
        outA = emb_layer(inputA)
        outB = emb_layer(inputB)
        transformed = concatenate([outA, outB]) # TODO check shape
        print("transformed shape: {}".format(transformed.shape))
        flat = Flatten()(transformed)
        classify_layer = Dense(1, activation='sigmoid')(flat)
        print("classify shape: {}".format(classify_layer.shape))
        
        self.model = Model(inputs=[inputA, inputB], outputs=classify_layer)
        
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#        opt = optimizers.Adam(learning_rate=lr_schedule)
        #self.autoenc.compile(optimizer=Adam(), loss='mean_squared_error')
        # Save a part of the autoencoder for encoding only.
        #code_layer = self.autoenc.get_layer(index=self.code_layer_index)
        #self.encoder = Model(inputs=self.autoenc.input, outputs=code_layer.output)

    def train(self, X1, X2, Y, epochs=3):
        # X1 is the source indexes, X2 is the target indexes, y are the comorbidity labels.
        self.model.fit(x=[X1, X2], y=Y, epochs=epochs)
        
    def predict(self, X1, X2):
        return self.model.predict(x=[X1,X2])
    
#    def encode(self, x):
#        # x is a vector of size <self.input_size>.
#        # E.g., an average of several disease embeddings.
#        # It also works with a text representation of a vector.
#        code = self.encoder(K.constant(np.matrix(x)))
#        return K.get_value(code)
#    
#    def encode_matrix(self, x):
#        return K.get_value(self.encoder(K.constant(x)))
    
def emb_adjust_workflow(
        emb_file_template, 
        read_cv_index='big_dataset_cui2vec_style_w2v_no_min_count_shuffle_CV.pickle'):
    male_emb = read_embedding_file(emb_file_template.format("male"))
    index_to_diag = list(male_emb.keys())
    diag_to_index = {diag: i for i, diag in enumerate(index_to_diag)}
    
    # read disease pairs file
    pairs = pd.read_csv(DISEASE_PAIRS_V2, index_col=0)
    pairs = pairs.rename({'source_name':'source', 'target_name':'target'}, axis=1)
    
    
    # create X (index pairs) and y (comorbidity labels)
    pairs['source_idx'] = pairs['source_cui'].apply(diag_to_index.get)
    pairs['target_idx'] = pairs['target_cui'].apply(diag_to_index.get)
    before = len(pairs)
    pairs = pairs.dropna(axis=0, subset=['source_idx', 'target_idx'])
    print("{}/{} disease pairs with embedding left".format(len(pairs), before))
    
    # create emb_matrix
    emb_matrix = make_embedding_matrix(index_to_diag, emb_file_template.format("male"), output_file=None)
    
    # divide to train and test to compare to other models
    # how will we compare to verification?
    CV_indices = pickle.load(open(read_cv_index, "rb"))
    aucs = []
    for train_index, test_index in CV_indices:
        X1_train = pairs.loc[train_index]['source_idx'].values
        X2_train = pairs.loc[train_index]['target_idx'].values
        y_train = pairs.loc[train_index]['W_pos_dep'].values
        
        X1_test = pairs.loc[test_index]['source_idx'].values
        X2_test = pairs.loc[test_index]['target_idx'].values
        y_test = pairs.loc[test_index]['W_pos_dep'].values
        print("len train: {}, len test: {}".format(len(y_train), len(y_test)))

        model = EmbAdjust(emb_matrix)
        model.train(X1_train, X2_train, y_train, epochs=5)
        y_pred = model.predict(X1_test, X2_test)
        aucs.append(roc_auc_score(y_test, y_pred))
    print("avg auc: {}".format(np.mean(aucs)))
    
    
    # which embeddings have changed?
    new_emb_matrix = model.model.layers[2].get_weights()[0]
    diffnorms = np.linalg.norm(emb_matrix - new_emb_matrix, axis=1)
    changes = pd.DataFrame(diffnorms)
    changes = changes.rename({0: "diff norm"}, axis=1)
    changes['cui'] = index_to_diag
    cui2name = pd.read_csv("E://Shunit//code_mappings/cui_table_for_cui2vec_with_abstract_counts.csv", index_col=0)[['cui','name']].set_index('cui').to_dict(orient='dict')['name']
    changes['name'] = changes['cui'].apply(lambda x: cui2name[x] if x in cui2name else x)
    # TODO did the cui even appear in the data?    
    cuis_in_data = set(pairs['source_cui'].values).union(set(pairs['target_cui'].values))
    changes['cui_in_data'] = changes['cui'].apply(lambda x: x in cuis_in_data)
    # TODO: how many people have the disease??
    
    