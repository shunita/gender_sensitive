from ast import literal_eval

import numpy as np
import pandas as pd
from collections import defaultdict
from bias_embedding_wrapper import EmbedModelWrapper
from bias_embeddings import repeat_by_participants

ABSTRACT_FILENAME = 'abstracts_and_population_tokenized_for_cui2vec_copyrightfix_sent_sep.csv'

#### Policies ####

def neutral(participants):
    return 1

def default_policy(participants):
    if participants == 0:
        repetitions = 0
    elif participants < 10:
        repetitions = 1
    elif participants < 100:
        repetitions = 10
    else:
        repetitions = 20
    return repetitions

def heur2_policy(participants):
    if participants == 0:
        repetitions = 1
    elif participants < 10:
        repetitions = 2
    elif participants < 100:
        repetitions = 3
    else:
        repetitions = 10
    return repetitions

def heur3_policy(participants):
    if participants == 0:
        repetitions = 1
    elif participants < 10:
        repetitions = 4
    elif participants < 100:
        repetitions = 8
    else:
        repetitions = 16
    return repetitions


def identity_policy(participants):
    return int(participants)

def log_policy(participants):
    return int(np.log(max(1, participants)))

def capped_log_policy(participants):
    res = log_policy(participants)
    res = max(1, res)  # each abstract appears at least once
    return res

def triple_log_policy(participants):
    res = log_policy(participants)
    res = 3 * res
    res = max(1, res)
    return res

def percent_policy(percent):
    return int(percent*15)

if __name__ == '__main__':
    # read the abstracts
    df = pd.read_csv(ABSTRACT_FILENAME, index_col=0)
    df['tokenized_sents'] = df['tokenized_sents'].apply(literal_eval)
    df['tokenized'] = df['tokenized_sents'].progress_apply(lambda x: " ".join(x).split())
    df['female_percent'] = df['female']/(df['female'] + df['male'])

    #df = df.dropna(axis=0, subset=['tokenized'])
    #df['tokenized'] = df['tokenized'].progress_apply(lambda x: x.split())

    policies = {
        'heur': default_policy,
        'log': log_policy,
        'capped_log': capped_log_policy,
        'heur2': heur2_policy,
        'heur3': heur3_policy,
        'triple_log': triple_log_policy,
        'percent15': percent_policy,
        'neutral': neutral,
    }

    for desc, policy in policies.items():
        print(f"training {desc}")
        model = EmbedModelWrapper(use_glove=False, min_count=0,
                                  vector_size=40, iterations=15,
                                  window=10)
        field = 'female'
        if desc.startswith('percent'):
            field = 'female_percent'
        model.train(repeat_by_participants(df, field, word_counter=defaultdict(int),
                                           threshold=0, policy=policy))
        output_fname = f'embedding_models/fem40_{desc}_emb.tsv'
        model.export_embeddings(output_fname)
        print(f"Written to: {output_fname}")
        emb = pd.read_csv(output_fname, sep='\t', names=['word', 'vector'])
        emb = emb.dropna(subset=['word'], axis=0)
        # take only CUI embeddings
        emb = emb[emb['word'].apply(lambda x: len(x) == 8 and x[0] == 'C')]
        emb.to_csv(output_fname, sep='\t', header=False, index=False)
