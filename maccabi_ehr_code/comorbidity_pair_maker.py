# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:54:41 2021

@author: shunit.agmon
"""
import time
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import norm
from scipy.sparse import lil_matrix
from tqdm import tqdm
from read_utilities import Translator, FOR_COMORBIDITY, read_patients_with_measurements



# create a dataset of comorbidty pairs
def positive_dependency_pval(d1, d2, both, none):
    sick_d1 = d1+both
    not_sick_d1 = none+d2
    if sick_d1 == 0 or not_sick_d1 == 0:
        return "reject no patients"
    # proportion of d2 out of d1 patients
    Px = both/sick_d1
    # proportion of d2 out of non-d1 patients
    Py = d2/not_sick_d1
    if Px<=0 or Py<=0:
        return "reject no patients"
    #variance of X - normal approximation for binomial distribution
    var_x = Px*(1-Px)/sick_d1
    var_y = Py*(1-Py)/not_sick_d1
    # test statistic
    z = (Px-Py)/np.sqrt(var_x+var_y)
    if z < 0:
        return "reject neg"
    pvalue = 1 - norm.cdf(z)
    return pvalue

def pval_to_bool(pval, threshold=0.05):
    if isinstance(pval, str):
        return False
    return (pval < threshold)

def positive_dependency_row(row, sex):
    # sex='M' or 'W' or ''
    both = row[sex+'count_both']
    d1 = row[sex+'count1']
    d2 = row[sex+'count2']
    none = row[sex+'count_none']
    
    sick_d1 = d1+both
    not_sick_d1 = none+d2
    
    # proportion of d2 out of d1 patients
    Px = both/sick_d1
    # proportion of d2 out of non-d1 patients
    Py = d2/not_sick_d1
    if Px<=0 or Py<=0:
        return "reject no patients"
    #variance of X - normal approximation for binomial distribution
    var_x = Px*(1-Px)/sick_d1
    var_y = Py*(1-Py)/not_sick_d1
    # test statistic
    z = (Px-Py)/np.sqrt(var_x+var_y)
    if z < 0:
        return "reject neg"
    pvalue = 1 - norm.cdf(z)
    return pvalue
  

def positive_dependency_M(row):
    return positive_dependency_row(row, 'M')


def positive_dependency_W(row):
    return positive_dependency_row(row, 'W')


def positive_dependency_N(row):
    return positive_dependency_row(row, '')
    

def patient_bin_to_matrix(cuis, index_to_cui, cui_to_index):
    # cuis is an iterable of lists, in each list there are cuis.
    mat = lil_matrix((len(cuis), len(index_to_cui)), dtype=int)
    for i, cui_list in tqdm(enumerate(cuis), total=len(cuis)):
        cui_indices = [cui_to_index[c] for c in cui_list if c in cui_to_index]
        mat[i, cui_indices] = 1
    return mat.tocsr()

def count_cui_combinations(cui_pairs, diags, translator, 
                            age_categories, smoking_categories,
                            index_to_cui, cui_to_index):
    print("count cui combinations take 2 (even faster) started")
    counter = {'total': {}}
    cui_trans = {}
    for c1, c2 in cui_pairs:
        if c1 not in cui_trans:
            cui_trans[c1] = translator.cui_to_icd9(c1)
        if c2 not in cui_trans:
            cui_trans[c2] = translator.cui_to_icd9(c2)
        if c1 not in counter:
            counter[c1] = {}
        if c2 not in counter:
            counter[c2] = {}
        if (c1, c2) not in counter:
            counter[(c1,c2)] = {}
    # go over age and smoking categories
    # in each, count d1, d2, both. only1 = d1-both, only2 = d2-both
    for age in age_categories:
        for smoke in smoking_categories:
            print("bucket: age: {} smoke: {}".format(age, smoke))
            s = time.time()
            patient_bin = diags[(diags['age'] == age) &
                                (diags['smoking'] == smoke) &
                                (diags['assignment'] == FOR_COMORBIDITY)]
            mat = patient_bin_to_matrix(patient_bin['cuis'], index_to_cui, cui_to_index)
            disease_count = np.sum(mat, axis=0)
            cooccur = mat.T @ mat
            if (age, smoke) not in counter['total']:
                counter['total'][(age, smoke)] = len(patient_bin)
            print("going over disease pairs")
            for c1, c2 in cui_pairs:
                if (age, smoke) not in counter[c1]:
                    counter[c1][(age, smoke)] = disease_count[0, cui_to_index[c1]]
                if (age, smoke) not in counter[c2]:
                    counter[c2][(age, smoke)] = disease_count[0, cui_to_index[c2]]
                if (age, smoke) not in counter[(c1, c2)]:
                    got_both = cooccur[cui_to_index[c1], cui_to_index[c2]]
                    counter[(c1,c2)][(age, smoke)] = got_both
            print("finished bucket in {} seconds".format(time.time()-s))
    return counter
                    
def adjust_disease_pairs_with_cui_overrides(pairs, translator):
    cols = ['source', 'target', 'weight_female', 'weight_male', 
            'weight_neutral', 'abs_diff', 'source_cui', 'source_label',
            'target_cui', 'target_label']
    pairs = pairs[cols]
    new_rows = []
    untouched_idx = pairs.apply(lambda row: 
        row['source_cui'] not in translator.overrides
        and row['target_cui'] not in translator.overrides, axis=1)
    #untouched = pairs[untouched_idx]
    need_change = pairs[~untouched_idx]
    for i, row in need_change.iterrows():
        source_options = [row['source_cui']]
        target_options = [row['target_cui']]
        source_name = row['source']
        target_name = row['target']
        change_source_name, change_target_name = False, False
        if row['source_cui'] in translator.overrides:
            source_options = list(translator.overrides[row['source_cui']].keys())
            change_source_name = True
        if row['target_cui'] in translator.overrides:
            target_options = list(translator.overrides[row['target_cui']].keys())
            change_target_name = True
        for source in source_options:
            if change_source_name:
                source_name = row['source']+ "_" +source.split("_")[1]
            for target in target_options:
                if change_target_name:
                    target_name = row['target'] + "_" +target.split("_")[1]
                new_row = [source_name, target_name, row['weight_female'],
                           row['weight_male'], row['weight_neutral'], 
                           row['abs_diff'], source, row['source_label'],
                           target, row['target_label']]
                new_rows.append(new_row)
        original_row = [row['source'], row['target'], row['weight_female'],
                           row['weight_male'], row['weight_neutral'], 
                           row['abs_diff'], row['source_cui'], row['source_label'],
                           row['target_cui'], row['target_label']]
        new_rows.append(original_row)
    # Keep all the lines and add the new versions
    #ret = pd.concat([untouched, pd.DataFrame(new_rows, columns=cols)], ignore_index=True)            
#    print(f"removed {len(need_change)} lines, added {len(new_rows)} new lines,"
#          f"before: {len(pairs)}, after: {len(ret)}")
    ret = pd.DataFrame(new_rows, columns=cols)
    print(f"created {len(ret)} rows involving overrides")
    print(ret.head())
    return ret
                
    

def count_and_test_for_relation_of_disease_pairs(
        diag_groups=None, pairs=None, 
        M_patients=None, F_patients=None, 
        translator=None, read_counter_from_pickle=False,
        write_to_file="E:\Shunit\cuis_filtered_glove_single1_pairs_151nodes_ztest_fix.csv"):
    
    if pairs is None:
        pairs = pd.read_csv(
                "E:\Shunit\cuis_filtered_glove_single1_pairs_151nodes.csv",
                #"E:\Shunit\cuis_filtered_glove_single1_min40_all_pairs.csv",
                #"E:\Shunit\cuis_filtered_glove_merged5_pairs.csv",
                index_col=0)
        #pairs = adjust_disease_pairs_with_cui_overrides(pairs, translator)

    ages = ['0-18', '18-30', '30-50', '50-70', '>70']
    smokings = [1,2,3]
    if read_counter_from_pickle:
        male_counter = pickle.load(open("male_counter.pickle", "rb"))
        female_counter = pickle.load(open("female_counter.pickle", "rb"))
    else:
        if translator is None:
            translator = Translator()
        cui_pairs = pairs[['source_cui','target_cui']].values.tolist()
        
        cuis_in_pairs = set(pairs['source_cui'].values).union(set(pairs['target_cui'].values))
        index_to_cui = list(cuis_in_pairs)
        cui_to_index = {cui: i for i, cui in enumerate(index_to_cui)}
        
        print("read {} disease pairs made of {} diseases.".format(len(cui_pairs), len(index_to_cui)))
    
        patients = read_patients_with_measurements(read_from_file=True)
        print("read {} patients".format(len(patients)))
        patients = patients[patients['assignment'] == FOR_COMORBIDITY]
        print("keeping {} patients whose assignment is for comorbidity".format(len(patients)))
        if M_patients is None:
            M_patients = diag_groups['M'].merge(patients, on='RANDOM_ID', how='inner')
            
            s = time.time()
            M_patients['cuis'] = M_patients['icd9'].progress_apply(translator.many_icd9s_to_cuis)
            print("finished mapping M_patients icd9 to cuis: {} seconds".format(time.time()-s))
            
        if F_patients is None:
            F_patients = diag_groups['F'].merge(patients, on='RANDOM_ID', how='inner')
            
            s = time.time()
            F_patients['cuis'] = F_patients['icd9'].progress_apply(translator.many_icd9s_to_cuis)
            print("finished mapping F_patients icd9 to cuis: {} seconds".format(time.time()-s))
            
        
        start = time.time()
        male_counter = count_cui_combinations(
                cui_pairs, M_patients, translator, 
                age_categories=ages, 
                smoking_categories=smokings, index_to_cui=index_to_cui, cui_to_index=cui_to_index)
        with open('male_counter.pickle', 'wb') as out:
            pickle.dump(male_counter, out)
        female_counter = count_cui_combinations(
                cui_pairs, F_patients, translator, 
                age_categories=ages, 
                smoking_categories=smokings, index_to_cui=index_to_cui, cui_to_index=cui_to_index)
        with open('female_counter.pickle', 'wb') as out:
            pickle.dump(female_counter, out)
        end = time.time()
        print(f"finished counting female and male diseases in {end-start} seconds")
    counters = {'W_': female_counter, 'M_': male_counter}
    print("starting to fill the dataframe dict")
    res = defaultdict(list)
    done = 0
    for i, row in pairs.iterrows():
        done+=1
        if done % 10000 == 0:
            print("done: {}/{}".format(done, len(pairs)))
        cui1 = row['source_cui']
        cui2 = row['target_cui']
        res['source_cui'].append(cui1)
        res['target_cui'].append(cui2)
        res['source_name'].append(row['source_name'])
        res['target_name'].append(row['target_name'])
        totals = {'M_only1': 0, 'M_only2': 0, 'M_both': 0, 'M_none': 0,
                  'W_only1': 0, 'W_only2': 0, 'W_both': 0, 'W_none': 0}
        for age in ages:
            for smoke in smokings:
                for gender in ['M_', 'W_']:
                    field_prefix = f'{gender}{age}_{smoke}'
                    counter = counters[gender]
                    both = counter[(cui1, cui2)][(age, smoke)]
                    only1 = counter[cui1][(age, smoke)] - both
                    only2 = counter[cui2][(age, smoke)] - both
                    none = counter['total'][(age, smoke)] - only1 - only2 - both
                    res[field_prefix+'_1'].append(only1)
                    res[field_prefix+'_2'].append(only2)
                    res[field_prefix+'_both'].append(both)
                    res[field_prefix+'_none'].append(none)
                    totals[f'{gender}only1'] += only1
                    totals[f'{gender}only2'] += only2
                    totals[f'{gender}both'] += both
                    totals[f'{gender}none'] += none
                    # first direction
                    pval12 = positive_dependency_pval(
                        only1, only2, both, none)
                    res[field_prefix+'_pval'].append(pval12)
                    res[field_prefix+'_pos_dep'].append(pval_to_bool(pval12))
                    # second direction
                    pval21 = positive_dependency_pval(only2, only1, both, none)
                    res[field_prefix+'_pval21'].append(pval21)
                    res[field_prefix+'_pos_dep21'].append(pval_to_bool(pval21))
                field_prefix = f'{age}_{smoke}'
                fem_field_prefix = f'W_{age}_{smoke}'
                male_field_prefix = f'M_{age}_{smoke}'
                values_for_pval_calc = []
                for suffix in ['_1', '_2','_both','_none']:
                    combined = res[male_field_prefix+suffix][-1] + res[fem_field_prefix+suffix][-1]
                    values_for_pval_calc.append(combined)
                    res[field_prefix+suffix].append(combined)

                bin_pval12 =  positive_dependency_pval(*values_for_pval_calc)
                res[field_prefix+'_pval'].append(bin_pval12)
                res[field_prefix+'_pos_dep'].append(pval_to_bool(bin_pval12))
                values_for_pval_calc[0], values_for_pval_calc[1] = values_for_pval_calc[1], values_for_pval_calc[0]
                bin_pval21 =  positive_dependency_pval(*values_for_pval_calc)
                res[field_prefix+'_pval21'].append(bin_pval21)
                res[field_prefix+'_pos_dep21'].append(pval_to_bool(bin_pval21))
        ########### sum over age and smoking bins ############
        for gender in ('M_', 'W_'):
            res[f'{gender}1'].append(totals[f'{gender}only1'])
            res[f'{gender}2'].append(totals[f'{gender}only2'])
            res[f'{gender}both'].append(totals[f'{gender}both'])
            res[f'{gender}none'].append(totals[f'{gender}none'])    
            pval = positive_dependency_pval(res[f'{gender}1'][-1], res[f'{gender}2'][-1], 
                                            res[f'{gender}both'][-1], res[f'{gender}none'][-1])
            res[f'{gender}pval'].append(pval)
            res[f'{gender}pos_dep'].append(pval_to_bool(pval))
            pval21 = positive_dependency_pval(res[f'{gender}2'][-1], res[f'{gender}1'][-1], 
                                            res[f'{gender}both'][-1], res[f'{gender}none'][-1])
            res[f'{gender}pval21'].append(pval21)
            res[f'{gender}pos_dep21'].append(pval_to_bool(pval21))
        values_for_pval_calc = []
        for suffix in ['_1', '_2','_both','_none']:
            combined = res['M'+suffix][-1] + res['W'+suffix][-1]
            values_for_pval_calc.append(combined)
            res['all'+suffix].append(combined)

        pval = positive_dependency_pval(*values_for_pval_calc)
        res['all_pval'].append(pval)
        res['all_pos_dep'].append(pval_to_bool(pval))
        values_for_pval_calc[0], values_for_pval_calc[1] = values_for_pval_calc[1], values_for_pval_calc[0]
        pval21 = positive_dependency_pval(*values_for_pval_calc)
        res['all_pval21'].append(pval21)
        res['all_pos_dep21'].append(pval_to_bool(pval21))

    print("converting to dataframe")
    s = time.time()
#    for k in res:
#        print(f'{k}: {len(res[k])}')
    res_df = pd.DataFrame(data=res)
    print("converted results to dataframe in {} seconds".format(time.time()-s))
    if write_to_file is not None:
        print("writing to file: {}".format(write_to_file))
        res_df.to_csv(write_to_file)
    else:
        return res_df


#    
# all prefixes
# prefixes = ['W_', 'M_', 'all_'] + ['{}{}_{}_'.format(sex, age, smoke) for sex in ('W_', 'M_', '') for age in age_categories for smoke in smoking_categories]

def custom_test_set(diag_groups, patients=None, 
                    disease_set={'C0020473', 'C0020443'}):
    pairs = pd.read_csv(
            "E:\Shunit\cuis_filtered_glove_single1_pairs_151nodes.csv",
            index_col=0)
    trans = Translator()
    pairs = adjust_disease_pairs_with_cui_overrides(pairs, trans)
    pairs = count_and_test_for_relation_of_disease_pairs(
        diag_groups, pairs, patients=patients, translator=trans, write_to_file=None)
    
    # add the pairs with diseases from disease_set (for later comparison)
    more_pairs = pd.read_csv(DISEASE_PAIRS_FILE, index_col=0)
    subset = more_pairs[more_pairs.apply(lambda row: row['source_cui'] in disease_set or row['target_cui'] in disease_set, axis=1)]
    together = pd.concat([pairs, subset])
    together.to_csv("comorbidity_test_set_with_cui_overrides.csv")