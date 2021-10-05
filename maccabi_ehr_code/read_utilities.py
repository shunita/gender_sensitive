# -*- coding: utf-8 -*-

import os
import time
import pandas as pd
import numpy as np
from ast import literal_eval
from collections import defaultdict
from tqdm import tqdm, tqdm_gui
tqdm.pandas()

SEX_CODES = {1: 'F', 2: 'M', 3:'N'}

DIAGNOSIS_DIR = "E:\RawData\Diagnosys"
DIAGNOSIS_FILES = [os.path.join(DIAGNOSIS_DIR, x) for x in
                   ["DIAG_2003Correct.txt", 
                    "DIAG_2003_2007Correct.txt",
                    "DIAG_2008_2011Correct.txt",
                    "DIAG_2012_2016Correct.txt"
                   ]]

DIAGNOSIS_FIELD_NAMES = ['RANDOM_ID', 'CUSTOMERIDCODE', 'DIAGNOSIS_CODE', 
                         'DATE_DIAGNOSIS', 'DIAGNOSIS_TYP_CD',
                         'STATUS_DIAGNOSIS']

#ICD9_CUI_FILE = "E:\Shunit\code_mappings\icd9merge_with_cuis.csv"
ICD9_CUI_FILE = "data/icd9merge_with_cuis.csv"
CUSTOMERS_FILE = 'E:\RawData\customers.csv'

MEDIDOT_FILE = "E:\Techniyon\Medidot\Measurments\Measurments.txt"
MEDIDOT_COLUMNS = ['RANDOM_ID', 'Date_Medida_Refuit', 'Madad_Refui_CD', 'Value1', 'Value2']

RASHAMIM_FILE = "E:\Techniyon\RASHAMIM_processed.csv"

PATIENTS_WITH_MEASUREMENTS = "patients_with_measurements.csv"
# assignment values
FOR_TRANSLATION = 1
FOR_COMORBIDITY = 0

DISEASE_PAIRS_FILE = "E:\Shunit\cuis_filtered_glove_single2_pairs_151nodes_ztest_fix.csv"


# pairs of CUIs that have embedding in cui2vec_style_w2v and an icd9 match. (2118 different CUIs.)
DISEASE_PAIRS_V2 = "E:\\Shunit\\comorbidity_pairs\\cui2vec_style_w2v_cuis_disease_pairs_labeled.csv"
#DISEASE_PAIRS_DOUBLE_LABELS = "E:\\Shunit\\comorbidity_pairs\\cui2vec_style_w2v_cuis_disease_pairs_column_subset_double_labels.csv"
DISEASE_PAIRS_DOUBLE_LABELS = "data/cui2vec_style_w2v_cuis_disease_pairs_column_subset_double_labels.csv"

FRAG_DIR = "E:\\Shunit\\temporal_comorbidity"

def read_customers(keep_fields):
    customers = pd.read_csv(CUSTOMERS_FILE, index_col=0)
    customers['SEX'] = customers['CUSTOMER_SEX_CODE'].apply(lambda x: SEX_CODES[x])
    #customers = customers.drop(labels=['CUSTOMER_BIRTH_DAT', 
    #                                   'CUSTOMERSTATUSYNCD', 
    #                                   'CUSTOMER_STATUS_DATE', 
    #                                   'DATE_DEATH',
    #                                   'CUSTOMER_SEX_CODE'], axis=1)
    return customers[keep_fields+['SEX']]


def union(list_of_sets):
    s = set()
    for s1 in list_of_sets:
        s = s.union(s1)
    return s

# for reading a string rep of an array from pandas csv
def make_array(string):
    return np.array([float(x) for x in string.strip('[]').split()], dtype="float32")

#def keep_icd9(code):
#    if code is None:
#        return False
#    if any(c.isalpha() for c in code):
#        return False
#    return True
#
#def filter_diag_set(diag_set, trans):
#    icd9s = []
#    for diag in diag_set:
#        icd9 = trans.maccabi_to_icd9(diag)
#        if icd9 is None:
#            continue
#        if any(c.isalpha() for c in icd9):
#            continue
#        icd9s.append(icd9)
#    return set(icd9s)
    
def read_rasham():
    rasham = pd.read_csv("E:\Techniyon\RASHAMIM.txt")
    # TODO: fix dates? they are strings
    rasham['DIAB1'] = (rasham['DIAB_TYPE'] == 1).apply(int)
    rasham['DIAB2'] = (rasham['DIAB_TYPE'] == 2).apply(int)
    rasham['DIAB9'] = (rasham['DIAB_TYPE'] == 9).apply(int)
    chronics = ['CARDIO_YN_CD', 'CARDIO_MAJDESY_CD', 'CARDIO_CHF_CD', 
                'CARDIO_PIRPUR_CD', 'CARDIO_OTHER_YN_CD', 'CARDIO_IHD', 
                'CARDIO_MI', 'CVD_CD', 'CVA_CD', 'TIA_CD', 'NONCVA_CD', 
                'DIAB1', 'DIAB2', 'DIAB9', 'BLOODPRESURE_CD']
    return rasham, chronics


def read_diagnosis(customers):
    trans = Translator()
    all_groups = {'M': None, 'F': None}
    for i in range(len(DIAGNOSIS_FILES)):
        print("reading {}".format(DIAGNOSIS_FILES[i]))
        diag = pd.read_csv(DIAGNOSIS_FILES[i],
                   index_col=False,
                   header=None,
                   names=DIAGNOSIS_FIELD_NAMES,
                   encoding='latin', dtype={'DATE_DIAGNOSIS': 'str'})
        diag = diag.drop(labels=['CUSTOMERIDCODE',
                                 'DIAGNOSIS_TYP_CD', 
                                 'STATUS_DIAGNOSIS'], axis=1)
        diag['DIAGNOSIS_CODE'] = diag['DIAGNOSIS_CODE'].str.strip()
        merged = diag.merge(customers, on='RANDOM_ID')
        print("total records: {}, after merge with sex: {}".format(len(diag), len(merged)))
        for sex in ('M', 'F'):
            diag_filtered = merged[merged.SEX == sex]
            diags_as_list = diag_filtered.groupby(by=['RANDOM_ID'])['DIAGNOSIS_CODE'].apply(set).reset_index()
            groups = diags_as_list
            if all_groups[sex] is None:
                all_groups[sex] = groups
            else:
                all_groups[sex] = pd.concat([all_groups[sex], groups])
    for sex in ('M', 'F'):
        all_groups[sex] = all_groups[sex].groupby(by=['RANDOM_ID'])['DIAGNOSIS_CODE'].apply(union).reset_index()
        all_groups[sex]['icd9'] = all_groups[sex]['DIAGNOSIS_CODE'].apply(trans.many_maccabi_to_icd9_no_actions)
    return all_groups


def verify_year(year):
    if year<1900 or year>2020:
        return "weird year"
    else:
        return ""


def age_buckets(row):
    birth_year = int(str(row['CUSTOMER_BIRTH_DAT'])[:4])
    #status_year = int(row['CUSTOMER_STATUS_DATE'][:4])
    death_year = int(row['DATE_DEATH'][:4])
    if death_year == 9999: # patient is alive
        reference_year = 2020
    else:
        reference_year = death_year
    for year in [birth_year, reference_year]:
        s = verify_year(year)
        if len(s)>0:
            print("{} at {}".format(s, row.name))
    age = reference_year - birth_year
    if age<18:
        return "0-18"
    if age<30:
        return "18-30"
    if age<50:
        return "30-50"
    if age<70:
        return "50-70"
    return ">70"


def read_patients_with_measurements(read_from_file=False):
    print("started read_patients_with_measurements")
    start = time.time()
    if read_from_file and os.path.isfile(PATIENTS_WITH_MEASUREMENTS):
        return pd.read_csv(PATIENTS_WITH_MEASUREMENTS, index_col=0)
    patients = read_customers(keep_fields=['RANDOM_ID', 'CUSTOMER_BIRTH_DAT', 'DATE_DEATH'])
    patients['age'] = patients.apply(age_buckets, axis=1)
    
    medidot = pd.read_csv(MEDIDOT_FILE, encoding='latin')
    # value of 4 means "unknown" so it is not a real measurement.
    smoking = medidot[(medidot['Madad_Refui_CD'] == 40) & (medidot['Value1'].isin([1,2,3]))]
    latest_smoking = smoking.sort_values('Date_Medida_Refuit').groupby(by='RANDOM_ID').tail(1)
    latest_smoking['smoking'] = latest_smoking['Value1'].apply(int)
    customers_smoking = patients.merge(latest_smoking[['RANDOM_ID', 'smoking']], how='inner', on='RANDOM_ID')
    customers_smoking['assignment'] = np.random.randint(0,2,len(customers_smoking)).tolist()
    print("read patients with measurements in {} seconds".format(time.time()-start))
    return customers_smoking[['RANDOM_ID', 'SEX', 'age', 'smoking', 'assignment',
                              'CUSTOMER_BIRTH_DAT', 'DATE_DEATH']] # last two are only for temporal comorbidity.


class Translator(object):
    def __init__(self, translation_file=ICD9_CUI_FILE):
        # cui to icd9 by manual match
        # 1. go over manual file and create two dictionaries: cui-> icd9 and icd9->cui.
        #    The manual file has only the most common CUIs.
        CUI_TO_ICD9S_FILE = "E:\\Shunit\\code_mappings\\cui_to_icd9_manual.csv"
        cui_to_icd9_df = pd.read_csv(CUI_TO_ICD9S_FILE, index_col=0)
        cui_to_icd9_df['ICD9'] = cui_to_icd9_df['ICD9'].apply(literal_eval)
        self.cui_to_icd9s = defaultdict(lambda: set())
        self.icd9_to_cuis = defaultdict(lambda: set())
        for i,r in cui_to_icd9_df.iterrows():
            cui = r['CUI']
            icd9s = r['ICD9']
            for icd9 in icd9s:
                self.cui_to_icd9s[cui].add(icd9)
                self.icd9_to_cuis[icd9].add(cui)
        # 2. go over automatic file and add more cui<->icd9 mappings.
        df = pd.read_csv(translation_file, index_col=None)
        df['seed_cuis'] = df['seed_cuis'].apply(literal_eval)
        for i,r in df.iterrows():
            icd9 = r['ICD9']
            cuis = r['seed_cuis']
            for cui in cuis:
                self.cui_to_icd9s[cui].add(icd9)
                self.icd9_to_cuis[icd9].add(cui)
        # split some cui mappings to subgroups
        self.cui_overrides()
                
        # 3. maccabi to icd9 mapping
        icd9_maccabi = pd.read_csv(
                'E:\\Shunit\\code_mappings\\original_maccabi_to_icd9_map.csv')
        self.mac_to_icd9 = {}
        self.icd9_to_macs = defaultdict(lambda: set())
        for i,r in icd9_maccabi.iterrows():
            mac = r['DIAGNOSIS_CODE']
            icd9 = r['ICD9']
            if any(c.isalpha() for c in icd9):
                continue
            self.mac_to_icd9[mac] = icd9
            self.icd9_to_macs[icd9].add(mac)
        # icd9 to CCS (category)
        self.icd9_to_cat = {}
        ccs = pd.read_csv('E:\\Techniyon\\CCS\\Single_Level_CCS_2015\\$dxref 2015_translated.csv', index_col=0)
        for i,r in ccs.iterrows():
            icd9_str = r['ICD-9-CM CODE'].strip("'")
            icd9 = icd9_str[:3]+"."+icd9_str[3:]
            icd9_second_version = r['icd9Code']
            category = r['CCS CATEGORY DESCRIPTION']
            self.icd9_to_cat[icd9] = category
            self.icd9_to_cat[icd9_second_version] = category
            
    def cui_overrides(self):
        self.overrides = {
                'C0242339': {'C0242339_hypo': {'272.5'},
                             'C0242339_hyper': {'272.0', '272.1', '272.2', '272.3', '272.4'},
                             'C0242339_disorder': {'272', '272.6', '272.7', '272.8', '272.9'}
                    }
                }
        for cui_to_override in self.overrides:
            # remove previous mapping / Keep previous mapping??
#            icd9s = self.cui_to_icd9s.pop(cui_to_override)
#            for icd9 in icd9s:
#                self.icd9_to_cuis[icd9].remove(cui_to_override)
            # create new mapping
            for new_cui in self.overrides[cui_to_override]:
                icd9s = self.overrides[cui_to_override][new_cui]
                self.cui_to_icd9s[new_cui] = icd9s
                for icd9 in icd9s:
                    self.icd9_to_cuis[icd9].add(new_cui)
        
    
    def maccabi_or_icd9_to_cui(self, code):
        if code[0] == 'Y': # maccabi code
            if code not in self.mac_to_icd9:
                return []
            icd9 = self.mac_to_icd9[code]
            return list(self.icd9_to_cuis[icd9])
        # otherwise - this is ICD9
        return list(self.icd9_to_cuis[code])
    
    def icd9_to_maccabi(self, code):
        return list(self.icd9_to_macs[code])
        
    def maccabi_to_icd9_no_actions(self, code):
        if code[0]!='Y': # this is already icd9
            if any(c.isalpha() for c in code): # action
                return None
            return code
        # maccabi code - try to translate
        if code not in self.mac_to_icd9:
            return None
        return self.mac_to_icd9[code]
    
    def many_maccabi_to_icd9_no_actions(self, sequence):
        res = []
        for code in sequence:
            icd9 = self.maccabi_to_icd9_no_actions(code)
            if icd9 is not None:
                res.append(icd9)
        return set(res)
    
    def many_codes_to_cui(self, sequence):
        ret = []
        for code in sequence:
            cuis = self.maccabi_or_icd9_to_cui(code)
            ret.extend(cuis)
        return list(set(ret))
    
    def cui_to_maccabi_or_icd9(self, code):
        icd9s = self.cui_to_icd9s[code]
        ret = []
        for icd9 in icd9s:
            ret.append(icd9)
            ret.extend(list(self.icd9_to_mac[icd9]))
        return ret
    
    def cui_to_icd9(self, code):
        return list(self.cui_to_icd9s[code])        

    def many_cuis_to_maccabi_or_icd9(self, sequence):
        ret = []
        for code in sequence:
            ret.extend(self.cui_to_maccabi_or_icd9(code))
        return list(set(ret))
    
    def all_icd9_list(self):
        # This returns all icd9s that appear in maccabi data.
        return list(self.icd9_to_macs.keys())
    
    def get_icd9_to_cuis(self, icd9):
        return self.icd9_to_cuis[icd9]
    
    def many_icd9s_to_cuis(self, sequence):
        ret = []
        for code in sequence:
            ret.extend(self.icd9_to_cuis[code])
        return list(set(ret))
    

def get_IDF_dict():
    df = pd.read_csv(DISEASE_PAIRS_FILE, index_col=0)
    r = df.iloc[0]
    N = r['all_1']+r['all_2']+r['all_both']+r['all_none']
    df['count_source'] = df['all_1'] + df['all_both']
    df = df[['source_cui', 'count_source']]
    df.drop_duplicates(inplace=True)
    df['IDF'] = N/df['count_source']
    df = df[['source_cui', 'IDF']]
    df = df.set_index('source_cui')
    idfs = df.to_dict(orient='dict')['IDF']
    for cui in idfs:
        if idfs[cui] == np.inf:
            idfs[cui] = 0
    return idfs
    
class EmbeddingWithOverrides(object):
    def __init__(self, emb_file):
        self.size = None
        if 'cui2vec_pretrained' in emb_file:
            df = pd.read_csv(emb_file, index_col=0)
            self.emb = {}
            for cui,r in df.iterrows():
                self.emb[cui] = r.values
                if self.size is None:
                    self.size = len(self.emb[cui])
        else:
            def string_to_array(s):
                if s.startswith('['):
                    parts = s.strip('[]').split()
                else:
                    parts = s.split(",")
                return np.array([float(x) for x in parts])
            
            emb = pd.read_csv(emb_file, sep="\t", header=None, index_col=0,
                              names=['cui', 'vector'])
            emb['vector'] = emb['vector'].apply(string_to_array)
            self.size = len(emb.iloc[0]['vector'])
            # make a dict from cui to embedding
            self.emb = emb.to_dict(orient='dict')['vector']
        
    def __getitem__(self, key):
        if type(key)!= str:
            print("invalid key for embedding: {} of type {}".format(key, type(key)))
        new_key = key
        parts = key.split("_")
        if len(parts) > 1:
            new_key = parts[0]
        if new_key not in self.emb:
            return np.zeros(self.size)
        return self.emb[new_key]
    
    def __contains__(self, key):
        parts = key.split("_")
        return (parts[0] in self.emb)
    
    def keys(self):
        return self.emb.keys()
        

def read_embedding_file(emb_file):
    return EmbeddingWithOverrides(emb_file)



def make_embedding_matrix(index_to_diag, emb_file, output_file=None):
    # For each diag in the data, add a line to the matrix
    # the mapping should be cui-> embedding
    # if there is no embedding for a diag, initialize it to zeros. 
    # remember which diags were missing.
    emb = read_embedding_file(emb_file)
    vecs = []
    misses = []
    for ind, diag in enumerate(index_to_diag):
        lookup = diag
        if diag.startswith("c"):
            lookup = "C" + diag[1:]
        if lookup in emb:
            v = emb[diag]
        else:
            misses.append(diag)
            v = np.zeros(emb.size)
        vecs.append(v)
    print("{} misses out of {} diags in the data".format(len(misses), len(index_to_diag)))
    if output_file is not None:
        f = open(output_file, "w")
        f.write("\n".join(misses))
        f.close()
    return np.stack(vecs)

