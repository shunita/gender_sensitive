# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:26:12 2021

@author: shunit.agmon
"""
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict

def avg_auc_by_female_prevalence(results_df, auc_columns, auc_labels, accumulate=True):
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    avg_aucs = {label: defaultdict(int) for label in auc_labels}
    lens = defaultdict(int)
    for i in range(len(thresholds)-1):
        if accumulate:
            subset = results_df[results_df['female_prevalence_proportion'] < thresholds[i+1]]
        else:
            subset = results_df[(results_df['female_prevalence_proportion'] > thresholds[i]) & (results_df['female_prevalence_proportion'] < thresholds[i+1])]
        lens[thresholds[i+1]] = len(subset)
        for j in range(len(auc_labels)):
            avg_aucs[auc_labels[j]][thresholds[i+1]] = subset[auc_columns[j]].mean()
    show_thresholds = [t for t in thresholds[1:] if lens[t]>5]
    for auc_label in auc_labels:
        plt.scatter(show_thresholds, [avg_aucs[auc_label][t] for t in show_thresholds], label=auc_label)
    plt.xlabel('female prevalence proportion')
    plt.ylabel('average AUC')
    plt.ylim(0.5,1)
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    plt.show()
    print(lens)
    
    
def boxplot_auc_by_female_prevalence(results_df, auc_columns, auc_labels, accumulate=True):
#    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    thresholds = [0, 0.25, 0.5, 0.75, 1]
    aucs = {label: defaultdict(int) for label in auc_labels}
    lens = defaultdict(int)
    for i in range(len(thresholds)-1):
        if accumulate:
            subset = results_df[results_df['female_prevalence_proportion'] < thresholds[i+1]]
        else:
            subset = results_df[(results_df['female_prevalence_proportion'] > thresholds[i]) & (results_df['female_prevalence_proportion'] < thresholds[i+1])]
        lens[thresholds[i+1]] = len(subset)
        for j in range(len(auc_labels)):
            #print(f'{auc_columns[j]}, {subset[auc_columns[j]].values}')
            aucs[auc_labels[j]][thresholds[i+1]] = list(subset[auc_columns[j]].values)
    show_thresholds = [t for t in thresholds[1:] if lens[t]>5]
    data = []
    xlabels = []
    for t in show_thresholds:

        for auc_label in auc_labels:    
            xlabels.append(f'{t}_{auc_label}')
            data.append(aucs[auc_label][t])
    print(f'xlabels:{xlabels}')
    fig, ax= plt.subplots(nrows=1, ncols=1, figsize=(7, 5))   
    bplot = ax.boxplot(data, showmeans=True, showfliers=False, patch_artist=True)
    colors = ['pink', 'moccasin']
    strong_colors=['red', 'orange']
#    for item in ['boxes', 'means']:
    for i, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(colors[i%2])
    for item in ['means', 'medians']:
        for i, thing in enumerate(bplot[item]):
            plt.setp(thing, color=strong_colors[i%2])
        
    xticks=list(range(len(xlabels)))
    ax.set_xticks(xticks)
    plt.xlabel('female prevalence proportion')
    plt.ylabel('AUC')
    plt.ylim(0.5,1)
    plt.grid(axis='y')
#    plt.legend(loc='lower right')
    plt.xticks(xticks, xlabels, rotation=45)
    plt.show()
    print(lens)

#df = pd.read_csv('//MKM-FS-INT01/OutgoingToApprove/shunit.agmon/analysis_classify_res_non_trainable_emb40_W_female_v_neutral_weighted.csv', index_col=0)
#df = df[df['abstracts']>5]
#avg_auc_by_female_prevalence(df, ['female auc', 'neutral auc'], ['female', 'neutral'])