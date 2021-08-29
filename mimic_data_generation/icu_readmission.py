import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()

MIMIC_PATH = "/home/shunita/mimic3/physionet.org/files/mimiciii/1.4/"
CUSTOM_TASKS_PATH = "/home/shunita/mimic3/custom_tasks"
CREATE_ICU_STAYS = True

# based on this: https://github.com/Jeffreylin0925/MIMIC-III_ICU_Readmission_Analysis/blob/master/mimic3-readmission/scripts/create_readmission.py
# will need to cite it https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0218942#sec005

def merge_stays_counts(table1, table2):
    return table1.merge(table2, how='inner', left_on=['HADM_ID'], right_on=['HADM_ID'])

def add_inhospital_mortality_to_icustays(stays):
    mortality_all = stays.DOD.notnull() | stays.DEATHTIME.notnull()
    stays['MORTALITY'] = mortality_all.astype(int)

    # in hospital mortality
    mortality = stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME))

    stays['MORTALITY0'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY0']
    return stays


def add_inunit_mortality_to_icustays(stays):
    mortality = stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME))

    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays

def read_stays():
    stays = pd.read_csv(os.path.join(MIMIC_PATH, 'ICUSTAYS.csv'), index_col=0)
    # needs merge to get all the fields
    admis = pd.read_csv(os.path.join(MIMIC_PATH, 'ADMISSIONS.csv'), index_col=0)
    admission_fields = ['HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']
    
    stays = stays.merge(admis[admission_fields], on='HADM_ID', how='inner')
    patients = pd.read_csv(os.path.join(MIMIC_PATH, 'PATIENTS.csv'), index_col=0)
    stays = stays.merge(patients[['SUBJECT_ID', 'DOB', 'DOD', 'GENDER']], on='SUBJECT_ID', how='inner')
    
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.ADMITTIME = pd.to_datetime(stays.ADMITTIME)
    stays.DISCHTIME = pd.to_datetime(stays.DISCHTIME)
    stays.DOB = pd.to_datetime(stays.DOB)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays
    
def get_next_intime(row, stays):
    subj_id = row['SUBJECT_ID']
    outtime = row['OUTTIME']
    later_stays = stays[(stays.SUBJECT_ID == subj_id) & (stays.INTIME > outtime)]
    if len(later_stays) == 0:
        return None
    return later_stays.sort_values(by='INTIME', ascending=True).iloc[0].INTIME



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

if CREATE_ICU_STAYS:
    stays = read_stays()
    stays = add_inhospital_mortality_to_icustays(stays)
    stays = add_inunit_mortality_to_icustays(stays)
    #stays = stays.drop(stays[(stays.MORTALITY == 1) & (stays.MORTALITY_INHOSPITAL == 1) & (stays.MORTALITY_INUNIT == 1)].index)

    # how many icu stays per hospital admission
    counts = stays.groupby(['HADM_ID']).size().reset_index(name='COUNTS')
    stays = merge_stays_counts(stays, counts)
    # binary column: is this stay the last one in the hospital admission?
    max_outtime = stays.groupby(['HADM_ID'])['OUTTIME'].transform(max) == stays['OUTTIME']
    stays['MAX_OUTTIME'] = max_outtime.astype(int)

    # was the patient transferred back to the icu, during this admission, after this stay?    
    transferback = (stays.COUNTS > 1) & (stays.MAX_OUTTIME == 0)
    stays['TRANSFERBACK'] = transferback.astype(int)

    # Did the patient die in the hospital but out of the icu?    
    dieinward = (stays.MORTALITY == 1) & (stays.MORTALITY_INHOSPITAL == 1) & (stays.MORTALITY_INUNIT == 0)
    stays['DIEINWARD'] = dieinward.astype(int)


    # take only icu stays that were the last in their admission and calculate the time until the next admission
    # TODO: this is wrong!!! Maybe we're taking admissions of other patients??
    #next_admittime = stays[max_outtime]
    #next_admittime = next_admittime[['HADM_ID', 'ICUSTAY_ID', 'ADMITTIME', 'DISCHTIME']]
    #next_admittime['NEXT_ADMITTIME'] = next_admittime.ADMITTIME.shift(-1)
    #next_admittime['DIFF'] = next_admittime.NEXT_ADMITTIME - stays.DISCHTIME
    #stays = merge_stays_counts(stays, next_admittime[['HADM_ID', 'DIFF']])

    stays['NEXT_INTIME'] = stays.progress_apply(lambda row: get_next_intime(row, stays), axis=1)
    stays['DIFF'] = stays['NEXT_INTIME'] - stays['OUTTIME']


    less_than_30days = stays.DIFF.notnull() & (stays.DIFF < '30 days 00:00:00')
    #less_than_30days = stays.DIFF.notnull() & (stays.DIFF < 30)
    stays['LESS_THAN_30DAYS'] = less_than_30days.astype(int)

    # did the patient die after being discharged? (from the hospital, not from the ICU)
    #stays['DISCHARGE_DIE'] = (stays.DOD - stays.DISCHTIME).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24
    stays['DISCHARGE_DIE'] = stays.DOD - stays.DISCHTIME
    stays['DIE_LESS_THAN_30DAYS'] = (stays.MORTALITY == 1) & (stays.MORTALITY_INHOSPITAL == 0) & (stays.MORTALITY_INUNIT == 0) & (stays.DISCHARGE_DIE < '30 days 00:00:00')
    stays['DIE_LESS_THAN_30DAYS'] = stays['DIE_LESS_THAN_30DAYS'].astype(int)

    # final label calculation

    stays['READMISSION'] = ((stays.TRANSFERBACK==1) | (stays.DIEINWARD==1) | (stays.LESS_THAN_30DAYS==1) | (stays.DIE_LESS_THAN_30DAYS==1)).astype(int)

    # patients who died in the ICU - mark with "2" in the READMISSION column
    stays.loc[(stays.MORTALITY == 1) & (stays.MORTALITY_INHOSPITAL == 1) & (stays.MORTALITY_INUNIT == 1), 'READMISSION'] = 2
    stays.to_csv(os.path.join(CUSTOM_TASKS_PATH, 'data', 'stays_readmission.csv'))
    print(f"{len(stays)} ICU stays. \מ{stays.READMISSION.value_counts()}") 

else:
    stays = pd.read_csv(os.path.join(CUSTOM_TASKS_PATH, 'data', 'stays_readmission.csv'), index_col=0)

stays['AGE'] = stays.apply(safe_age, axis=1).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
stays.loc[stays.AGE.isna(), 'AGE'] = 90

# drugs in this icu stay
drugs = pd.read_csv(os.path.join(MIMIC_PATH, 'PRESCRIPTIONS.csv'), index_col=0)
icu_drugs = drugs.groupby(by='ICUSTAY_ID')['DRUG'].apply(list).reset_index()
stays = stays.merge(icu_drugs[['ICUSTAY_ID', 'DRUG']], on='ICUSTAY_ID', how='left')



# procedures in this admission (even if not in this ICU stay)
procs = pd.read_csv(os.path.join(MIMIC_PATH, 'PROCEDURES_ICD.csv'), index_col=0)
admission_procs = procs.sort_values(by=['HADM_ID','SEQ_NUM']).groupby(by='HADM_ID')['ICD9_CODE'].apply(list).reset_index()
admission_procs = admission_procs.rename({'ICD9_CODE': 'PROCEDURES'}, axis=1)
stays = stays.merge(admission_procs[['HADM_ID', 'PROCEDURES']], on='HADM_ID', how='left')


# diags in this admission (even if not in this ICU stay)
diag = pd.read_csv(os.path.join(MIMIC_PATH, 'DIAGNOSES_ICD.csv'), index_col=0)
admission_diags = diag.sort_values(by=['HADM_ID','SEQ_NUM']).groupby(by='HADM_ID')['ICD9_CODE'].apply(list).reset_index()
admission_diags = admission_diags.rename({'ICD9_CODE': 'DIAGS'}, axis=1)

admis = pd.read_csv(os.path.join(MIMIC_PATH, 'ADMISSIONS.csv'), index_col=0)
admis = admis[['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME']]
admis.ADMITTIME = pd.to_datetime(admis.ADMITTIME)
admis.DISCHTIME = pd.to_datetime(admis.DISCHTIME)
admission_diags = admission_diags.merge(admis, on='HADM_ID')
admission_procs = admission_procs.merge(admis, on='HADM_ID')


# previous diags - from previous admissions
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
        new_row['PREV_DIAGS'] = combine(prev_stays['DIAGS'].values)
        last_discharge = prev_stays['DISCHTIME'].max()
        new_row['DAYS_SINCE_LAST_ADMIS'] = (in_time - last_discharge)/np.timedelta64(1, 's')/60./60/24
    prev_procs = admission_procs[(admission_procs.SUBJECT_ID == subj_id) & (admission_procs.DISCHTIME < in_time)]
    new_row['NUM_PREV_PROCS'] = len(combine(prev_procs.PROCEDURES.values))
    records.append(new_row)
records = pd.DataFrame.from_records(records)

stays = stays.merge(records[['HADM_ID', 'DIAGS', 'PREV_DIAGS', 'DAYS_SINCE_LAST_ADMIS', 'NUM_PREV_ADMIS', 'NUM_PREV_PROCS']], on='HADM_ID', how='left')


assignment = pd.read_csv(os.path.join(CUSTOM_TASKS_PATH, 'data', 'train_test_patients_full.csv'))
stays = stays.merge(assignment, on='SUBJECT_ID')
print(f"Divided stays to {len(stays[stays.ASSIGNMENT == 'train'])} train and {len(stays[stays.ASSIGNMENT == 'test'])} test stays")


stays.to_csv(os.path.join(CUSTOM_TASKS_PATH, 'data', 'stays_readmission_plus.csv'))

sys.exit()


























# read ICU stays
stays = pd.read_csv(os.path.join(MIMIC_PATH, 'ICUSTAYS.csv'), index_col=0)
stays.INTIME = pd.to_datetime(stays.INTIME)

admis = pd.read_csv(os.path.join(MIMIC_PATH, 'ADMISSIONS.csv'), index_col=0)
admis.DISCHTIME = pd.to_datetime(admis.DISCHTIME)
admis.ADMITTIME = pd.to_datetime(admis.ADMITTIME)
admis = admis.sort_values('ADMITTIME', ascending=True)
admis_gb = admis.groupby('SUBJECT_ID')
records = []
for subj_id, group in admis_gb:
    if len(group) < 2:
        continue
    else:
        prev_visit_end = None
        prev_visit_start = None
        prev_adm_id = None
        prev_ethnic = None
        
        for i, admission_row in group.iterrows():
            if prev_visit_end is None:
                prev_visit_end = admission_row['DISCHTIME']
                prev_visit_start = admission_row['ADMITTIME']
                prev_adm_id = admission_row['HADM_ID']
                prev_ethnic = admission_row['ETHNICITY']
            else:
                record = {'SUBJECT_ID': subj_id,
                          'ETHNICITY':  prev_ethnic,
                          'HADM_ID': prev_adm_id,
                          'ADMITTIME': prev_visit_start,
                          'DISCHTIME': prev_visit_end,
                          'NEXT_ADMITTIME': admission_row['ADMITTIME']}
                records.append(record)
                prev_visit_end = admission_row['DISCHTIME']
                prev_visit_start = admission_row['ADMITTIME']
                prev_adm_id = admission_row['HADM_ID']
                prev_ethnic = admission_row['ETHNICITY']

admissions = pd.DataFrame.from_records(records)
# We count the number of days to the next admission.
admissions['TIME_TO_READMISSION'] = (admissions['NEXT_ADMITTIME'] - admissions['DISCHTIME']).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24
# and convert to a binary label.
admissions['READMITTED30'] = admissions['TIME_TO_READMISSION'] < 30


# take all the diagnoses and group them by admission
diag = pd.read_csv(os.path.join(MIMIC_PATH, 'DIAGNOSES_ICD.csv'), index_col=0)
admission_diags = diag.sort_values(by=['HADM_ID','SEQ_NUM']).groupby(by='HADM_ID')['ICD9_CODE'].apply(list).reset_index()
admission_diags = admission_diags.rename({'ICD9_CODE': 'DIAGS'}, axis=1)
admissions = admissions.merge(admission_diags[['HADM_ID', 'DIAGS']], on='HADM_ID', how='left')
del admission_diags, diag

# same for procedures
procs = pd.read_csv(os.path.join(MIMIC_PATH, 'PROCEDURES_ICD.csv'), index_col=0)
admission_procs = procs.sort_values(by=['HADM_ID','SEQ_NUM']).groupby(by='HADM_ID')['ICD9_CODE'].apply(list).reset_index()
admission_procs = admission_procs.rename({'ICD9_CODE': 'PROCEDURES'}, axis=1)
admissions = admissions.merge(admission_procs[['HADM_ID', 'PROCEDURES']], on='HADM_ID', how='left')
del admission_procs, procs

# and the same for drugs
drugs = pd.read_csv(os.path.join(MIMIC_PATH, 'PRESCRIPTIONS.csv'), index_col=0)
admis_drugs = drugs.groupby(by='HADM_ID')['DRUG'].apply(set).reset_index()
admissions = admissions.merge(admis_drugs[['HADM_ID', 'DRUG']], on='HADM_ID', how='left')
del drugs, admis_drugs

print(f"found {len(admissions)} records who had previous hospital visits from {admissions.SUBJECT_ID.nunique()} unique subjects.")

# get demographics from ADMISSIONS and gender from PATIENTS
patients = pd.read_csv(os.path.join(MIMIC_PATH, 'PATIENTS.csv'), index_col=0)
patients.DOB = pd.to_datetime(patients.DOB)

admissions = admissions.merge(patients[['SUBJECT_ID', 'GENDER', 'DOB']], on='SUBJECT_ID')
admissions['AGE'] = admissions.apply(safe_age, axis=1).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60/24/365
admissions.loc[admissions.AGE.isna(), 'AGE'] = 90

# assign to train and test (0.15)
#TODO: use the same assignment from the los_by_diag task!!
subject_assignment = pd.read_csv(os.path.join(CUSTOM_TASKS_PATH, 'data', 'train_test_patients.csv'), index_col=0)['ASSIGNMENT'].to_dict()
admissions['ASSIGNMENT'] = admissions['SUBJECT_ID'].apply(lambda x: subject_assignment[x])

print(f"Train-test division: {admissions.ASSIGNMENT.value_counts()}")
admissions.to_csv(os.path.join(CUSTOM_TASKS_PATH, 'data', 'readmission_by_diag_plus.csv'))
