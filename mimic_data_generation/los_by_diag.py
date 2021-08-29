import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()

MIMIC_PATH = "/home/shunita/mimic3/physionet.org/files/mimiciii/1.4/"
CUSTOM_TASKS_PATH = "/home/shunita/mimic3/custom_tasks"


def combine(list_of_sets):
    s = []
    for s1 in list_of_sets:
        s.extend(list(s1))
    return s

    
def safe_age(row):
    try:
        return row['ADMITTIME'] - row['DOB']
    except:
        return None


# take all the diagnoses and group them by admission
diag = pd.read_csv(os.path.join(MIMIC_PATH, 'DIAGNOSES_ICD.csv'), index_col=0)
#admission_diags = diag.groupby(by='HADM_ID')['ICD9_CODE'].apply(set).reset_index()
admission_diags = diag.sort_values(by=['HADM_ID','SEQ_NUM']).groupby(by='HADM_ID')['ICD9_CODE'].apply(list).reset_index()

admis = pd.read_csv(os.path.join(MIMIC_PATH, 'ADMISSIONS.csv'), index_col=0)
admis.DISCHTIME = pd.to_datetime(admis.DISCHTIME)
admis.ADMITTIME = pd.to_datetime(admis.ADMITTIME)
admis['LOS'] = (admis.DISCHTIME - admis.ADMITTIME).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24
admission_diags = admission_diags.merge(admis[['HADM_ID', 'SUBJECT_ID', 'LOS', 'ADMITTIME', 'DISCHTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']], on='HADM_ID')
#records = admission_diags

# get length of stay from ICUSTAYS
#stays = pd.read_csv(os.path.join(MIMIC_PATH, 'ICUSTAYS.csv'), index_col=0)
#stays.INTIME = pd.to_datetime(stays.INTIME)
#stays = stays[['SUBJECT_ID', 'ICUSTAY_ID', 'INTIME', 'LOS', 'HADM_ID']]

procs = pd.read_csv(os.path.join(MIMIC_PATH, 'PROCEDURES_ICD.csv'), index_col=0)
admission_procs = procs.sort_values(by=['HADM_ID','SEQ_NUM']).groupby(by='HADM_ID')['ICD9_CODE'].apply(list).reset_index()
admission_procs = admission_procs.merge(admis[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME']], on='HADM_ID')

# Add previous diagnoses to each admission
records = []
for i,r in admission_diags.iterrows():
    subj_id = r['SUBJECT_ID']
    in_time = r['ADMITTIME']
    prev_stays = admission_diags[(admission_diags.SUBJECT_ID == subj_id) & (admission_diags.DISCHTIME < in_time)]
    
    new_row = r
    new_row['NUM_PREV_ADMIS'] = len(prev_stays)
    if len(prev_stays) == 0:
        new_row['PREV_DIAGS'] = {}
        new_row['DAYS_SINCE_LAST_ADMIS'] = 0
    else:
        new_row['PREV_DIAGS'] = set(combine(prev_stays['ICD9_CODE'].values))
        last_discharge = prev_stays['DISCHTIME'].max()
        new_row['DAYS_SINCE_LAST_ADMIS'] = (in_time - last_discharge)/np.timedelta64(1, 's')/60./60/24
    prev_procs = admission_procs[(admission_procs.SUBJECT_ID == subj_id) & (admission_procs.DISCHTIME < in_time)]
    new_row['NUM_PREV_PROCS'] = len(set(combine(prev_procs.ICD9_CODE.values)))
    records.append(new_row)
records = pd.DataFrame.from_records(records)

#records = []
#print("Iterating over ICU_STAYS:")
#for i, r in tqdm(stays.iterrows(), total=len(stays)):
#    subj_id = r['SUBJECT_ID']
#    in_time = r['INTIME']
#    # We're looking for all diagnoses the patient had in previous admissions to the one with the ICU stay.
#    # In one admission we don't know which diagnoses happened before the ICU in time and which happened during the ICU stay.
#    prev_stays = admission_diags[(admission_diags.SUBJECT_ID == subj_id) & (admission_diags.DISCHTIME < in_time)]
#    if len(prev_stays) == 0:
#        continue
#    all_diags = combine(prev_stays['ICD9_CODE'].values)
#    records.append({'SUBJECT_ID': subj_id, 'INTIME': in_time, 'PREV_DIAGS': all_diags, 'LOS': r['LOS'], 'HADM_ID': r['HADM_ID']})
#records = pd.DataFrame.from_records(records)
#print(f"found {len(records)} records who had previous hospital visits from {records.SUBJECT_ID.nunique()} unique subjects.")

# get demographics from ADMISSIONS and gender from PATIENTS
patients = pd.read_csv(os.path.join(MIMIC_PATH, 'PATIENTS.csv'), index_col=0)
patients.DOB = pd.to_datetime(patients.DOB)


#records = records.merge(admis[['ETHNICITY', 'HADM_ID']], on='HADM_ID')
records = records.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB']], on='SUBJECT_ID')
records['AGE'] = records.apply(safe_age, axis=1).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
records.loc[records.AGE.isna(), 'AGE'] = 90

# assign to train and test (0.2)
#subjects = patients.SUBJECT_ID.unique()
#train_subjects, test_subjects = train_test_split(subjects, test_size=0.15)
#train_subjects = set(train_subjects)
assignment = pd.read_csv(os.path.join(CUSTOM_TASKS_PATH, 'data', 'train_test_patients_full.csv'))
records = records.merge(assignment, on='SUBJECT_ID')
#records.loc[records.SUBJECT_ID.apply(lambda x: x in train_subjects), 'ASSIGNMENT'] = 'train'
#num_test = len(records[records.ASSIGNMENT.isna()])
#num_train = len(records)-num_test
#records.loc[records.ASSIGNMENT.isna(), 'ASSIGNMENT'] = 'test'
print(f"Divided records to {len(records[records.ASSIGNMENT == 'train'])} train and {len(records[records.ASSIGNMENT == 'test'])} test records")
records.to_csv(os.path.join(CUSTOM_TASKS_PATH, 'data', 'los_by_diag_v4.csv'))