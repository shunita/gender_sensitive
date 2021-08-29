import sys
import random
import re
import pandas as pd
from ast import literal_eval
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from visualization import scatter_with_hover_annotation

tqdm.pandas(desc="pandas tqdm")

import text_utils as tu
from metamap_wrapper import extract_metamap_concepts
import string
from nltk.corpus import stopwords

from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.spatial.distance import cosine as cos_dist
# from glove import Glove, Corpus
from collections import defaultdict

from pubmed_w2v_embed import *
from bias_embedding_wrapper import EmbedModelWrapper, ModelMerger

SEMANTIC_TYPES = ['dsyn',  # disease or syndrome
                  #'sosy',  # sign or symptom
                  #'clnd',  # clinical drug
                  #'topp',  # therapeutic or preventive procedure
                  ]
FILTER_BY_SEMTYPES = True

mode_to_filename = {'only_concepts': 'abstracts_and_population_tokenized_cui.csv',
                    'plain_text': 'abstracts_and_population_tokenized_no_concepts.csv',
                    'cui2vec_style': 'abstracts_and_population_tokenized_for_cui2vec.csv'}

copyright_sign = '©'

class CUIMapper(object):
    def __init__(self, cui_tsv_file):
        self.df = pd.read_csv(cui_tsv_file, index_col=0, sep='\t')
        self.df['semtypes'] = self.df['semtypes'].apply(literal_eval)

    def cui_to_name(self, cui):
        return self.df.at[cui, 'preferred_name']

    def cui_to_name_safe(self, cui):
        if cui in self.df.index:
            return self.df.at[cui, 'preferred_name']
        return None

    def cui_to_first_semtype(self, cui):
        return self.df.at[cui, 'semtypes'][0]

    def cui_matches_semtypes(self, cui, semtypes):
        cui_types = self.df.at[cui, 'semtypes']
        for t in cui_types:
            if t in semtypes:
                return True
        return False

    def name_to_cui(self, name):
        return self.df[self.df['name'] == name]['cui'].values[0]


# Yes this is ugly here but more efficient to call it once.
cui_mapper = CUIMapper("cui_table.tsv")


def clean_list(str_rep_of_list):
    # remove " and ', remove [], and make a list.
    l = literal_eval("[" + str_rep_of_list.strip('"\'').replace("[", "").replace("]", "") + "]")
    only_ncts = []
    for item in l:
        if item.startswith("NCT"):
            only_ncts.append(item)
    return only_ncts


# 1. extract how many women, men in each clinical trial (from AACT).
# Saved to "pmids_with_participants.tsv".
def create_pmid_to_participants():
    print("reading nct_to_participants")
    nct_to_participants = pd.read_csv('clinical_trial_to_participants_by_sex.csv', index_col=0)
    print("reading pmids_with_ncts")
    with open('pmids_with_ncts.tsv', "r", encoding='utf-8', errors="replace") as f:
        lines = [line.split("\t") for line in f.read().split("\n")]
    lines[0] += ["female", "male"]
    for i in range(1, len(lines)):
        if len(lines[i]) < 2:
            print("faulty line: {}".format(lines[i]))
            continue
        pmid = lines[i][0]
        ncts = clean_list(lines[i][1])
        if len(ncts) == 0:
            print(pmid, ncts, type(ncts), lines[i][1], type(lines[i][1]))
        lines[i][1] = str(ncts)
        female = 0
        male = 0
        for nct in ncts:
            if nct in nct_to_participants.index:
                female += nct_to_participants.loc[nct]['Female']
                male += nct_to_participants.loc[nct]['Male']
        lines[i] += [str(female), str(male)]
    with open("pmids_with_participants.tsv", "w") as out:
        for line in lines:
            try:
                out.write("{}\n".format("\t".join(line)))
            except:
                print("trouble writing a line to file:")
                return line


def merge_with_abstracts():
    main_df = pd.read_csv("pmids_with_participants.tsv", sep="\t", index_col=0)
    main_df = main_df[main_df['male'] + main_df['female'] > 0]
    print(main_df.head())
    shards = ['../pubmed_2018/pubmed_v2_shard_{}.csv'.format(i) for i in range(50)]
    merged = []
    for shard in shards:
        df = pd.read_csv(shard, index_col=0)
        print(df.iloc[0])
        new_df = pd.merge(left=main_df, right=df[['title', 'abstract', 'date', 'mesh_headings', 'keywords']], how='inner', left_index=True, right_index=True)
        print(new_df.iloc[0])
        merged.append(new_df)
    concated = pd.concat(merged)
    print("len(merged): {}, len(pmids_with_participants):{}".format(len(concated), len(main_df)))
    concated.to_csv("abstracts_population_date_topics.csv")


def topic_statistics():
    df = pd.read_csv("abstracts_population_date_topics.csv")
    topics = df.dropna(axis=0, subset=["mesh_headings"]).copy()
    topics['mesh_headings'] = topics['mesh_headings'].apply(lambda x: x.split(";"))
    topic_to_participants = defaultdict(lambda: {'f': 0, 'm': 0})
    for _, row in topics.iterrows():
        for topic in row['mesh_headings']:
            topic_to_participants[topic]['f'] += row['female']
            topic_to_participants[topic]['m'] += row['male']
    tdf = pd.DataFrame.from_records([(topic, p['m'], p['f'], p['abstracts']) for topic, p in topic_to_participants.items()],
                                    columns=['topic', 'm total', 'f total', 'num_abstracts'])
    df['year'] = df['date'].apply(lambda x: int(x[-4:]))
    df['fem_percent'] = df['female'] / (df['female'] + df['male'])
    gb = df.groupby('year')
    gb['male'].sum()
    gb['female'].sum()
    gb['fem_percent'].mean()


def read_tokenized_df(mode='only_concepts', print_stats=False):
    # mode: 'only_concepts'/ 'plain_text'/ 'cui2vec_style'
    filename = mode_to_filename[mode]
    print("reading {}".format(filename))
    df = pd.read_csv(filename, index_col=0)
    if mode == 'cui2vec_style':
        df = df.dropna(axis=0, subset=['tokenized_sents'])
        df['tokenized_sents'] = df['tokenized_sents'].apply(literal_eval)
        df['tokenized'] = df['tokenized_sents'].apply(lambda sents: " ".join(sents))
        df['tokenized'] = df['tokenized'].apply(lambda x: x.split())
    else:
        df = df.dropna(axis=0, subset=['tokenized'])
        df['tokenized'] = df['tokenized'].progress_apply(literal_eval)
    if mode == 'only_concepts' and FILTER_BY_SEMTYPES:
        print("Filtering by semtypes to keep only: {}".format(SEMANTIC_TYPES))
        # cui_mapper = CUIMapper("cui_table.tsv")

        def filter_by_semtypes(row):
            return [cui for cui in row if cui_mapper.cui_matches_semtypes(cui, SEMANTIC_TYPES)]

        df['tokenized'] = df['tokenized'].progress_apply(filter_by_semtypes)
    if print_stats:
        print("Corpus stats:")
        print("# abstracts: {}".format(len(df)))
        list_of_lists = df['tokenized'].values.tolist()
        all_words = [item for sublist in list_of_lists for item in sublist]
        print("# words: {}".format(len(all_words)))
        unique_words = set(all_words)
        print("# unique words: {}".format(len(unique_words)))
        cuis = [w for w in all_words if w.startswith('C') and len(w) == 8 and w[1:].isdigit()]
        print("# cuis: {}".format(len(cuis)))
        print("# unique cuis: {}".format(len(set(cuis))))
        print("calculating length of abstracts")

    lens = df['tokenized'].progress_apply(len)
    set_lens = df['tokenized'].progress_apply(lambda x: len(set(x)))
    print("average length of abstract: {}, max length: {}, min length: {}".format(np.mean(lens), max(lens), min(lens)))
    print("without repetitions in one abstract: average length: {} max length: {}, min length: {}".format(
        np.mean(set_lens), max(set_lens), min(set_lens)))

    return df


def word_count(list_of_word_lists):
    word_to_count = defaultdict(int)
    for lst in list_of_word_lists:
        for w in lst:
            word_to_count[w] += 1
    return word_to_count

def word_to_abstract_count(list_of_word_lists):
    word_to_abs_count = defaultdict(int)
    print("{} abstracts.".format(len(list_of_word_lists)))
    print("first abstract length: {}".format(len(list_of_word_lists[0])))
    print("first abstract type: {}".format(type(list_of_word_lists[0])))
    print("first abstract start: {}".format(list_of_word_lists[0][:5]))
    for lst in list_of_word_lists:
        for w in set(lst):
            word_to_abs_count[w] += 1
    return word_to_abs_count

def word_to_participant_count(df_tokenized):
    word_to_participants = defaultdict(lambda: {'m': 0, 'f': 0})
    for _, row in df_tokenized.iterrows():
        lst = row['tokenized']
        for w in set(lst):
            word_to_participants[w]['m'] += row['male']
            word_to_participants[w]['f'] += row['female']
    return word_to_participants

#def participants_to_repetitions(participants):
#    if participants == 0:
#        repetitions = 0
#    elif participants < 10:
#        repetitions = 10
#    elif participants < 100:
#        repetitions = 100
#    else:
#        repetitions = 200
#    return repetitions

def participants_to_repetitions(participants):
    if participants == 0:
        repetitions = 0
    elif participants < 10:
        repetitions = 1
    elif participants < 100:
        repetitions = 10
    else:
        repetitions = 20
    return repetitions


def repeat_by_participants(df, weight_col, word_counter, threshold,
                           policy=participants_to_repetitions):
    tokenized = []
    reps = defaultdict(int)
    total_len = 0
    for i, r in df.iterrows():
        word_list = r['tokenized']
        remaining = [word for word in word_list if word_counter[word] >= threshold]
        if len(remaining) < 2:
            reps['empty'] += 1
            continue
        total_len += len(remaining)
        repetitions = policy(r[weight_col])
        #DEBUG
        reps[repetitions] += 1
        for _ in range(repetitions):
            tokenized.append(remaining)
    # DEBUG
    print("repetitions histogram: {}".format(reps))
    remaining_abstracts = len(df)-reps['empty']
    print("remaining abstracts after removing rare words and then empty abstracts: {}, avg length: {}".format(
          remaining_abstracts,
          total_len/remaining_abstracts))
    random.shuffle(tokenized)
    return tokenized


def count_diff(neigh1, neigh2):
    diff = 0
    n1 = [x[0] for x in neigh1]
    n2 = [x[0] for x in neigh2]
    for item in n1:
        if item not in n2:
            diff += 1
    return diff


def read_models(fem_model_file, male_model_file, neutral_model_file):
    if USE_GLOVE:
        fem_model = Glove.load(fem_model_file)
        male_model = Glove.load(male_model_file)
        neutral_model = Glove.load(neutral_model_file)
    else:
        fem_model = Word2Vec.load(fem_model_file)
        male_model = Word2Vec.load(male_model_file)
        neutral_model = Word2Vec.load(neutral_model_file)
    return fem_model, male_model, neutral_model


def analyze_distances(model):
    distances = {i: [] for i in range(10)}
    for word in model.wv.vocab:
        neighbors = model.most_similar(word)
        for i in range(10):
            distances[i].append(neighbors[i][1])
    return distances


def fit_distribution(data_points):
    """gets a list of distances of the i's closest neighbor."""
    X = np.array(data_points).reshape(-1, 1)
    gmms = [GaussianMixture(i).fit(X) for i in range(1, 3)]
    aics = [gmm.aic(X) for gmm in gmms]
    chosen = gmms[np.argmin(aics)]

    # mean = np.mean(X)
    # var = np.var(X, ddof=1)
    # a = mean ** 2 * (1 - mean) / var - mean
    # b = a * (1 - mean) / mean
    print("min max data: {},{}".format(np.min(X), np.max(X)))
    a, b, loc, scale = beta.fit(X, floc=0.1, fscale=0.9)
    # print("beta: a {}, b {}, loc {}, scale {}".format(a,b,loc,scale))
    print("beta: a {}, b {}".format(a, b))

    # plot the fit
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    ax.hist(X, bins=50, density=True, histtype='stepfilled', alpha=0.4)
    logprob = chosen.score_samples(x_plot)
    pdf = np.exp(logprob)
    ax.plot(x_plot, pdf, '-k')
    means_str = ",".join(["{:.2f}".format(m) for m in chosen.means_.squeeze()])
    ax.set_title("GMM: means={}".format(means_str))

    ax2 = fig.add_subplot(122)
    ax2.hist(X, bins=50, density=True, histtype='stepfilled', alpha=0.4)
    # fitted = lambda x, a, b: gammaf(a + b) / gammaf(a) / gammaf(b) * x ** (a - 1) * (1 - x) ** (b - 1)  # pdf of beta
    x_plot = np.linspace(0.21, 0.99, 99).reshape(-1, 1)
    print("xplot: {},{}".format(x_plot[0], x_plot[-1]))
    ax2.plot(x_plot, beta.pdf(x_plot, a, b), 'k')
    ax2.set_title("Beta: a={:.2f}, b={:.2f}".format(a, b))
    plt.show()

def prettify_neighbors(neighbors):
    return ", ".join(["{}-{:.2f}".format(cui_mapper.cui_to_name(cui), d)
                      for cui, d in neighbors])


class GraphMaker(object):

    def __init__(self, models, descriptions):
        self.models = models  # list of 3 models: instances of EmbedModelWrapper.
        self.desc = descriptions  # list of descriptions in the same order as models list.
        self.cui_mapper = CUIMapper("cui_table.tsv")

    def label_word(self, count1, count2, count3, desc1, desc2, desc3):
        if count1 == 0 and count2 == 0:
            return self.desc[2]
        if count1 == 0 and count3 == 0:
            return self.desc[1]
        if count2 == 0 and count3 == 0:
            return self.desc[0]
        if count1*count2 > 0:
            return "both"
        if count1 > 0:
            return self.desc[0]
        if count2 > 0:
            return self.desc[1]

    def filter_cuis_by_icd9(self, cuis):
        matcher = pd.read_excel('cui_to_icd9_manual.xlsx')
        cols = ['ICD9 matches', 'manual match', 'approximate match']
        for col in cols:
            matcher[col] = matcher[col].apply(literal_eval)
        matcher['ICD9'] = matcher['ICD9 matches'] + matcher['manual match'] + matcher['approximate match']
        cuis_with_matching_icd9 = set(matcher[matcher['ICD9'].apply(lambda x: len(x) > 0)]['CUI'].values.tolist())
        # df = pd.read_csv('icd9merge_with_cuis.csv', index_col=0)
        # df['seed_cuis'] = df['seed_cuis'].apply(literal_eval)
        # single_list = []
        # for x in df['seed_cuis']:
        #     single_list.extend(x)
        # cuis_with_matching_icd9 = set(single_list)
        print("matched cuis in total: {}".format(len(cuis_with_matching_icd9)))
        remaining = [cui for cui in cuis if cui in cuis_with_matching_icd9]
        print("Number of cuis from abstracts remaining after icd9 match: {}".format(len(remaining)))
        return remaining


    def get_edges(self, word, name1, similarity_threshold, topn, nodes,
                  choose_edges_by_neighbors):
        neighbors = []
        if choose_edges_by_neighbors:
            for i, model in enumerate(self.models):
                top = model.most_similar(word, topn=topn)
                neighbors.extend([x[0] for x in top])
            neighbors = list(set(neighbors))
        else:  # use all nodes as neighbors
            neighbors = nodes
        edges_strings = []
        print("num edges for {}: {}".format(name1, len(neighbors)))
        for w2 in neighbors:
            if w2 == word:
                continue
            name2 = self.cui_mapper.cui_to_name(w2)
            if name1 > name2:
                first = name2
                second = name1
            else:
                first = name1
                second = name2
            for i, model in enumerate(self.models):
                vocab = model.get_vocab()
                if word not in vocab or w2 not in vocab:
                    continue
                sim = model.similarity(word, w2)
                if similarity_threshold is None or sim > similarity_threshold:
                    edges_strings.append("{}\t{}\t{:.4f}\t{}\n".format(first, second, sim, self.desc[i]))
        return edges_strings

    def make_graph(self, edges_output_file, nodes_output_file,
                   similarity_threshold, topn, choose_edges_by_neighbors,
                   filter_nodes_by_icd9):
        model1, model2, model3 = self.models
        desc1, desc2, desc3 = self.desc

        edges = open(edges_output_file, "w")
        edges.write("source\ttarget\tweight\tlabel\n")

        nodes = open(nodes_output_file, "w")
        nodes.write("cui\tname\tlabel\tsemtype\tcount_{}\tcount_{}\tcount_{}\n".format(desc1, desc2, desc3))
        cuis = list(set(model1.get_vocab_list() + model2.get_vocab_list() + model3.get_vocab_list()))
        if filter_nodes_by_icd9:
            cuis = self.filter_cuis_by_icd9(cuis)
            print("{} cui nodes after filter by icd9".format(len(cuis)))
        for i1, w1 in enumerate(tqdm(cuis)):
            name1 = self.cui_mapper.cui_to_name(w1)
            semtype = self.cui_mapper.cui_to_first_semtype(w1)
            count1 = model1.get_count(w1)
            count2 = model2.get_count(w1)
            count3 = model3.get_count(w1)
            label = self.label_word(count1, count2, count3, desc1, desc2, desc3)

            nodes.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(w1, name1, label, semtype, count1, count2, count3))
            # collects closest neighbors from all models
            # add (at most) 3 edges for each neighbor - one for each model
            edges_strings = self.get_edges(w1, name1, similarity_threshold, topn,
                                           cuis, choose_edges_by_neighbors)
            for edge in edges_strings:
                edges.write(edge)

        edges.close()
        edges = pd.read_csv(edges_output_file, sep='\t')
        print("before dedup: {}".format(len(edges)))
        edges.drop_duplicates(subset=['source', 'target', 'label'], inplace=True)
        print("after dedup: {}".format(len(edges)))
        edges.to_csv(edges_output_file, sep='\t')


def compare_models_by_neighbor_lists(fem_model, male_model,
                                    neutral_model, neutral_var_model,
                                    output_file, topn=10):
    stop_words = stopwords.words('english')
    with open(output_file, "w", encoding='utf-8', errors="replace") as out:
        out.write('{}\n'.format('\t'.join([
            'word',
            'count_neutral',
            'different_neighbors_male_female',
            'diff_neutral_variation',
            'dist_from_closest_word_female',
            'dist_from_second_closest_word_female',
            'dist_from_closest_word_male',
            'dist_from_second_closest_word_male',
            'dist_from_closest_word_neutral',
            'dist_from_second_closest_word_neutral',
            'fem_neighbors',
            'male_neighbors',
            'neutral_neighbors',
            'neutral_var_neighbors'])))

    for word in tqdm(fem_model.get_vocab_list()):
        if not ONLY_CONCEPTS and (word in string.punctuation or word in stop_words):
            continue
        if word in male_model.get_vocab():
            # how to compare?
            # option 1: difference in 10 closest neighbors
            # option 2: still missing
            neighbors_fem = fem_model.most_similar(word, topn)
            neighbors_male = male_model.most_similar(word, topn)
            # TODO: get rid of closest neighbor in some cases? When it's a very very close concept
            neighbors_neutral = []
            closest_neighbor_neutral = 0
            second_closest_neighbor_neutral = 0
            if neutral_model is not None:
                neighbors_neutral = neutral_model.most_similar(word, topn)
                if len(neighbors_neutral) >= 2:
                    closest_neighbor_neutral = neighbors_neutral[0][1]
                    second_closest_neighbor_neutral = neighbors_neutral[1][1]
            if neutral_var_model is not None:
                neighbors_neutral_var = neutral_var_model.most_similar(word, topn)

            diff = count_diff(neighbors_fem, neighbors_male)
            diff_neut = count_diff(neighbors_neutral, neighbors_neutral_var)

            with open(output_file, "a", encoding='utf-8', errors="replace") as out:
                out.write('{}\n'.format('\t'.join([str(x) for x in [
                    cui_mapper.cui_to_name(word),
                    neutral_model.get_count(word),
                    diff,
                    diff_neut,
                    neighbors_fem[0][1],
                    neighbors_fem[1][1],
                    neighbors_male[0][1],
                    neighbors_male[1][1],
                    closest_neighbor_neutral,
                    second_closest_neighbor_neutral,
                    prettify_neighbors(neighbors_fem),
                    prettify_neighbors(neighbors_male),
                    prettify_neighbors(neighbors_neutral),
                    prettify_neighbors(neighbors_neutral_var)]])))


def compare_models_arithmetic(model1, desc1, model2, desc2, neutral_model, output_file_single_word,
                              output_file_word_pairs):
    print("single: {} pairs: {}".format(output_file_single_word, output_file_word_pairs))
    # cui_mapper = CUIMapper("cui_table.tsv")
    words = set(list(model1.wv.vocab.keys()) + list(model2.wv.vocab.keys()))
    neutral_words = set(model1.wv.vocab.keys())
    print("only neutral: {}, only not neutral: {}".format(
        len(neutral_words.difference(words)),
        len(words.difference(neutral_words))
    ))
    words_in_all_models = neutral_words.intersection(set(model1.wv.vocab.keys()), set(model2.wv.vocab.keys()))

    if output_file_single_word is not None:
        out = open(output_file_single_word, "w")
        out.write("word\tcount_neutral\tcount_{}\tcount_{}\t{}-neutral\t{}-neutral\t"
                  "{}_cos_dist_neutral\t{}_cos_dist_neutral\n".format(desc1, desc2, desc1, desc2, desc1, desc2))
        for w in list(words_in_all_models):
            out.write("{}\n".format("\t".join([str(x) for x in [
                cui_mapper.cui_to_name(w),
                neutral_model.wv.vocab[w].count,
                model1.wv.vocab[w].count,
                model2.wv.vocab[w].count,
                np.linalg.norm(model1.wv[w] - neutral_model[w]),
                np.linalg.norm(model2.wv[w] - neutral_model[w]),
                cos_dist(model1.wv[w], neutral_model[w]),
                cos_dist(model2.wv[w], neutral_model[w]),
            ]])))
        out.close()
    if output_file_word_pairs is not None:
        out = open(output_file_word_pairs, "w")
        out.write(
            "w1\tw2\t"
            "w1_neutral_count\tw2_neutral_count\t"
            "w1_{}_count\tw2_{}_count\t"
            "w1_{}_count\tw2_{}_count\t"
            "neutral_dist\t{}_dist\t{}_dist\t{}_dist-{}dist\n".format(
                desc1, desc1, desc2, desc2, desc1, desc2, desc1, desc2))
        word_list = list(words_in_all_models)
        for i1, w1 in enumerate(word_list):
            if neutral_model.wv.vocab[w1].count < 10:
                continue
            w1_name = cui_mapper.cui_to_name(w1)
            for i2, w2 in enumerate(word_list):
                if i2 <= i1:
                    continue
                if neutral_model.wv.vocab[w2].count < 10:
                    continue
                out.write("{}\n".format("\t".join([str(x) for x in [
                    w1_name,
                    cui_mapper.cui_to_name(w2),
                    neutral_model.wv.vocab[w1].count,
                    neutral_model.wv.vocab[w2].count,
                    model1.wv.vocab[w1].count,
                    model1.wv.vocab[w2].count,
                    model2.wv.vocab[w1].count,
                    model2.wv.vocab[w2].count,
                    np.linalg.norm(neutral_model.wv[w1] - neutral_model[w2]),
                    np.linalg.norm(model1.wv[w1] - model1[w2]),
                    np.linalg.norm(model2.wv[w1] - model2[w2]),
                    np.linalg.norm(model1.wv[w1] - model1[w2]) - np.linalg.norm(model2.wv[w1] - model2[w2])
                ]])))
        out.close()


def compare_iterations_by_order_preservation(iteration1, iteration2):
    """The idea: if w1 and w2 are closer in female model than they are in male model, it should be that way 
    in both iterations."""
    print("comparing iterations {},{}".format(iteration1, iteration2))
    fem_model_file_template = "embedding_models/fem{}_w2v.model"
    male_model_file_template = "embedding_models/male{}_w2v.model"
    neutral_model_file_template = "embedding_models/neutral{}_w2v.model"
    fem1, male1, neutral1 = read_models(fem_model_file_template.format(iteration1),
                                        male_model_file_template.format(iteration1),
                                        neutral_model_file_template.format(iteration1))
    fem2, male2, neutral2 = read_models(fem_model_file_template.format(iteration2),
                                        male_model_file_template.format(iteration2),
                                        neutral_model_file_template.format(iteration2))
    # nodes are the same in all iterations.
    cuis = pd.read_csv('cuis_filtered_nodes_1.tsv', sep='\t')['cui'].get_values()
    same = 0
    total = 0
    for i, w1 in tqdm(enumerate(cuis)):
        for j, w2 in enumerate(cuis):
            if i <= j:
                continue
            total += 1
            sim = {iteration1: [], iteration2: []}
            if w1 in fem1.wv.vocab and w2 in fem1.wv.vocab:
                sim[iteration1].append(fem1.wv.similarity(w1, w2))
                sim[iteration2].append(fem2.wv.similarity(w1, w2))
            if w1 in male1.wv.vocab and w2 in male1.wv.vocab:
                sim[iteration1].append(male1.wv.similarity(w1, w2))
                sim[iteration2].append(male2.wv.similarity(w1, w2))
            if w1 in neutral1.wv.vocab and w2 in neutral1.wv.vocab:
                sim[iteration1].append(neutral1.wv.similarity(w1, w2))
                sim[iteration2].append(neutral2.wv.similarity(w1, w2))
            if np.all(np.argsort(sim[iteration1]) == np.argsort(sim[iteration2])):
                # Same order of proximity in both iterations
                same += 1
    print("same: {}/{} = {:.2f}%".format(same, total, (same/total) * 100))

def summarize_iterations(iterations, output_file):
    """aggregate the similarities from different iterations for each word pair."""
    fem_model_file_template = "embedding_models/fem{}_w2v.model"
    male_model_file_template = "embedding_models/male{}_w2v.model"
    neutral_model_file_template = "embedding_models/neutral{}_w2v.model"
    fem_models = {}
    male_models = {}
    neutral_models = {}
    cuis = None
    out = open(output_file, "w")
    out.write("cui1\tcui2\tword1\tword2\t"
              "closer_in_fem_times\tcloser_in_male_times\t"
              "fem_mean_sim\tfem_std_sim\tnum_fem_models\t"
              "male_mean_sim\tmale_std_sim\tnum_male_models\t"
              "neutral_mean_sim\tneutral_std_sim\tnum_neutral_models\n")
    for it in iterations:
        fem1, male1, neutral1 = read_models(fem_model_file_template.format(it),
                                            male_model_file_template.format(it),
                                            neutral_model_file_template.format(it))
        fem_models[it] = fem1
        male_models[it] = male1
        neutral_models[it] = neutral1

        cuis_it = set(list(fem1.wv.vocab.keys()) + list(male1.wv.vocab.keys()) + list(neutral1.wv.vocab.keys()))
        if cuis is None:
            print("found {} cuis in models from iteration {}".format(len(cuis_it), it))
            cuis = cuis_it
        else:
            missing = cuis_it.difference(cuis)
            print("found {} more cuis in models from iteration {}".format(len(missing), it))
            cuis = cuis.union(missing)

    for i, w1 in enumerate(tqdm(cuis)):
        name1 = cui_mapper.cui_to_name(w1)
        for j, w2 in enumerate(cuis):
            if i <= j:
                continue
            name2 = cui_mapper.cui_to_name(w2)
            sim = {it: {} for it in iterations}
            sim = {'fem': {}, 'male': {}, 'neutral': {}}
            closer_in_fem = 0
            closer_in_male = 0
            for it in iterations:
                if w1 in fem_models[it].wv.vocab and w2 in fem_models[it].wv.vocab:
                    sim['fem'][it] = fem_models[it].wv.similarity(w1, w2)
                if w1 in male_models[it].wv.vocab and w2 in male_models[it].wv.vocab:
                    sim['male'][it] = male_models[it].wv.similarity(w1, w2)
                if w1 in neutral_models[it].wv.vocab and w2 in neutral_models[it].wv.vocab:
                    sim['neutral'][it] = neutral_models[it].wv.similarity(w1, w2)

                if it in sim['male'] and it in sim['fem'] and sim['fem'][it] > sim['male'][it]:
                    closer_in_fem += 1
                elif it in sim['male'] and it in sim['fem'] and sim['fem'][it] < sim['male'][it]:
                    closer_in_male += 1
            fem_sim = list(sim['fem'].values())
            if len(fem_sim) == 0:
                fem_sim = [0]
            male_sim = list(sim['fem'].values())
            if len(male_sim) == 0:
                male_sim = [0]
            neutral_sim = list(sim['neutral'].values())
            if len(neutral_sim) == 0:
                neutral_sim = [0]
            out.write("{}\n".format("\t".join([
                w1, w2, name1, name2,
                str(closer_in_fem), str(closer_in_male),
                "{:.2f}".format(np.mean(fem_sim)),
                "{:.2f}".format(np.std(fem_sim)),
                str(len(fem_sim)),
                "{:.2f}".format(np.mean(male_sim)),
                "{:.2f}".format(np.std(male_sim)),
                str(len(male_sim)),
                "{:.2f}".format(np.mean(neutral_sim)),
                "{:.2f}".format(np.std(neutral_sim)),
                str(len(neutral_sim))]
            )))
    out.close()


def how_many_shared_edges(iteration1, iteration2):
    edges1 = pd.read_csv("embedding_graphs/cuis_filtered_glove{}_edges.tsv".format(iteration1), sep="\t", index_col=0)
    edges2 = pd.read_csv("embedding_graphs/cuis_filtered_glove{}_edges.tsv".format(iteration2), sep="\t", index_col=0)
    for label in ['male', 'female', 'neutral']:
        comb1 = set([tuple(x) for x in edges1[edges1.label == label][['source', 'target']].values])
        comb2 = set([tuple(x) for x in edges2[edges2.label == label][['source', 'target']].values])
        intersection = comb1.intersection(comb2)
        print("\n{} edges:\nin iteration {}: {}\nin iteration {}: {}\nshared: {}/{}={:.2f}%".format(
              label.upper(), iteration1, len(comb1), iteration2, len(comb2),
              len(intersection), len(comb1), 100*len(intersection)/len(comb1)))


def analyze_doc2vec(model, output_single_word, output_word_pairs):
    # cui_mapper = CUIMapper("cui_table.tsv")
    # go over word pairs
    # compute closest neighbor to 1. w1-w2+M 2. w1-w2+F
    male_vector = model.docvecs['male']
    female_vector = model.docvecs['female']
    word_list = list(model.wv.vocab.keys())
    if output_single_word is not None:
        out_single = open(output_single_word, "w")
        out_single.write("word\tcount\tneighbor_diff\tneighbors_female\tneighbors_male\n")
    if output_word_pairs is not None:
        out_pairs = open(output_word_pairs, "w")
        out_pairs.write("w1\tw2\tw1_count\tw2_count\tfirst_neighbor_female\tfirst_neighbor_male\n")

    for i1, w1 in enumerate(word_list):

        w1_name = cui_mapper.cui_to_name(w1)
        v1 = model.wv[w1]
        if output_single_word is not None:
            # single word neighbors
            male_neighbors_single = model.wv.similar_by_vector(vector=v1+male_vector)
            female_neighbors_single = model.wv.similar_by_vector(vector=v1+female_vector)
            out_single.write("{}\n".format("\t".join([
                w1_name,
                str(model.wv.vocab[w1].count),
                str(count_diff(male_neighbors_single, female_neighbors_single)),
                prettify_neighbors(female_neighbors_single),
                prettify_neighbors(male_neighbors_single)
            ])))
        if model.wv.vocab[w1].count < 10:
            continue
        if output_word_pairs is not None:
            for i2, w2 in enumerate(word_list):
                if i2 <= i1:
                    continue
                if model.wv.vocab[w2].count < 10:
                    continue
                w2_name = cui_mapper.cui_to_name(w2)
                v2 = model.wv[w2]
                male_neighbors = model.wv.similar_by_vector(vector=v1+male_vector-v2)
                female_neighbors = model.wv.similar_by_vector(vector=v1+female_vector-v2)
                out_pairs.write("{}\n".format("\t".join([
                    w1_name, w2_name, str(model.wv.vocab[w1].count), str(model.wv.vocab[w2].count),
                    prettify_neighbors(female_neighbors),
                    prettify_neighbors(male_neighbors)
                ])))


def find_interesting_cui_pairs(
        edges_filename="embedding_graphs\cuis_filtered_glove_merged5_edges.tsv",
        nodes_filename="embedding_graphs\cuis_filtered_glove_merged5_nodes.tsv"):
    edges = pd.read_csv(edges_filename, sep="\t", index_col=0)
    edges1 = edges[edges['weight'] != 1]
    edges_female = edges1[edges1['label'] == "female"][['source', 'target', 'weight']]
    edges_male = edges1[edges1['label'] == "male"][['source', 'target', 'weight']]
    edges_neutral = edges1[edges1['label'] == "neutral"][['source', 'target', 'weight']]
    merged = edges_female.merge(edges_male, on=['source', 'target'], how="inner", suffixes=["_female", "_male"])
    merged = merged.merge(edges_neutral, on=['source', 'target'], how="inner")
    merged = merged.rename(mapper={"weight": "weight_neutral"}, axis=1)
    merged['abs_diff'] = np.abs(merged['weight_female'] - merged['weight_male'])

    nodes = pd.read_csv(nodes_filename, sep="\t", index_col=None)
    nodes = nodes[['cui', 'name', 'label']]
    merged2 = merged.merge(nodes, left_on='source', right_on='name')
    merged2.rename(mapper={"cui": "source_cui", "label": "source_label"},
                   axis='columns', inplace=True)
    merged3 = merged2.merge(nodes, left_on='target', right_on='name')
    merged3.rename(mapper={"cui": "target_cui", "label": "target_label"},
                   axis='columns', inplace=True)
    filtered = merged3[(merged3.target_label == 'both') & (merged3.source_label == 'both')].copy()
    print("Before filter: {}, only 'both' label: {}".format(len(merged3), len(filtered)))
    filtered.sort_values(by='abs_diff', ascending=False, inplace=True)
    output = edges_filename.replace("edges", "pairs").replace("tsv", "csv")
    filtered.to_csv(output)


def map_to_group(word):
    subject_to_words = {1: ['heart', 'cardio', 'cardiac', 'angina', 'ventric', 'valve',
                            'aortic', 'coronary'],  # heart
                        2: ['kidney', 'renal', 'pyelonephritis'],  # kidney
                        3: ['cerebr', 'brain'],  # brain
                        4: ['liver', 'hepati'],  # liver
                        5: ['diabet'],  # diabetic
                        6: ['retin', 'eye', 'ocul'],  # eye

                        }
    for num in subject_to_words:
        word_list = subject_to_words[num]
        for keyword in word_list:
            if keyword in word.lower():
                return num
    return max(subject_to_words.keys())+1


def map_cuis_to_category_numbers(cui_list):
    cat_df = pd.read_csv(os.path.join('embedding_graphs','cuis_filtered_glove_single1_nodes_with_categories.csv'))
    categories = []
    for c in cat_df['category'].dropna().unique():
        categories.append(c.split("/")[0])
    categories = list(set(categories))
    categories.sort()
    categories_to_index = {categories[i]: i for i in range(len(categories))}
    unknown_cat = len(categories)
    print("num categories: {}".format(unknown_cat+1))
    res = []
    cui_to_cat = {x: y for x, y in cat_df.dropna(subset=['category'])[['cui', 'category']].values}
    for cui in cui_list:
        if cui in cui_to_cat:
            cat = cui_to_cat[cui].split('/')[0]
            res.append(categories_to_index[cat])
        else:
            res.append(unknown_cat)
    categories.append("unknown")
    return res, categories


def show_model_in_2d(model, model_name, word_counter, word_subset=None, top_words=None, filter_by_icd9=True, use_tsne=True, title=""):
    """model - an instance of EmbedModelWrapper"""
    cuis = model.get_vocab_list()
    if filter_by_icd9:
        gm = GraphMaker(models=[model], descriptions=[model_name])
        # only keep the concepts who have a matching icd9 code.
        cuis = gm.filter_cuis_by_icd9(cuis)
    if word_subset is not None:
        print("using words from given word subset, before filter: {}".format(len(cuis)))
        cuis = [c for c in cuis if c in word_subset]
        print("after filter: {}".format(len(cuis)))
    if top_words is not None:
        min_count = sorted(word_counter.values(), reverse=True)[top_words]
        filtered_idx = [i for i, cui in enumerate(cuis) if word_counter[cui] > min_count]
    else:
        filtered_idx = list(range(len(cuis)))
    # filtered = [cui for cui in cuis if word_counter[cui] > min_count]
    print("Showing {}/{} points.".format(len(filtered_idx), len(cuis)))
    word_labels = [cui_mapper.cui_to_name(cuis[i]) for i in filtered_idx]
    groups, categories_list = map_cuis_to_category_numbers([cuis[i] for i in filtered_idx])
    
    #groups = [map_to_group(word) for word in word_labels]
    arr = model.get_word_vectors_as_matrix_rows(cuis)
    if use_tsne:
        print("using TSNE")
        tsne = TSNE(n_components=2, random_state=0)
        points = tsne.fit_transform(arr)
    else:
        print("using PCA")
        pca = PCA(n_components=2, random_state=0)
        points = pca.fit_transform(arr)

    x_coords = points[filtered_idx, 0]
    y_coords = points[filtered_idx, 1]
    # display scatter plot
    #scatter_with_hover_annotation(x_coords, y_coords, word_labels, groups)
    plt.figure(figsize=(20, 10))
    plt.scatter(x_coords, y_coords, c=groups, s=100)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x+0.005, y), xytext=(0, 0), textcoords='offset points', alpha=0.6, 
                     #rotation=-20, rotation_mode='anchor',
                     fontsize=8)
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    #plt.colorbar(drawedges=True)
    plt.title(title)
    plt.annotate("\n".join(categories_list), xy =(15, 0))
    plt.savefig(model_name + '.png')
    plt.show()
    return cuis


def pretrained_vectors():
    concepts = pd.read_csv("embedding_graphs/cuis_filtered_glove_single1_min40_all_nodes.tsv", sep="\t")
    pretrained = PubmedW2V()
    def get_emb(row):
        return pretrained.get_embedding_for_token_seq(row['name'].split(" ")).tolist()
    concepts['emb'] = concepts.apply(get_emb, axis=1)
    pairs = pd.read_csv("embedding_graphs/cuis_filtered_glove_single1_min40_all_pairs.csv",
                        index_col=0)
    pairs = pairs.merge(concepts[['cui', 'emb']], how='inner',
                        left_on='source_cui', right_on='cui')
    pairs = pairs.rename({'emb': 'emb_source'}, axis=1)
    pairs = pairs.drop('cui', axis=1)
    pairs = pairs.merge(concepts[['cui', 'emb']], how='inner',
                        left_on='target_cui', right_on='cui')
    pairs = pairs.rename({'emb': 'emb_target'}, axis=1)
    pairs = pairs.drop('cui', axis=1)
    def calc_sim_by_pretrained(row):
        return 1 - cos_dist(np.array(row['emb_source']),
                            np.array(row['emb_target']))

    pairs['weight_pretrained'] = pairs.apply(calc_sim_by_pretrained,
                                             axis=1)
    out = pairs.drop(['emb_source', 'emb_target'], axis=1)
    return out

def extract_cui_embeddings_from_plain_w2v(emb_file, output_file):
    emb = pd.read_csv(emb_file, sep='\t', index_col=0, header=None,
                      names=['word', 'vector'])
    emb['vector'] = emb['vector'].apply(lambda st: np.array([float(x) for x in st.split(",")]))
    emb = emb.to_dict(orient='dict')['vector']

    def process_cui(name):
        if name in emb:
            return emb[name]
        if name.lower() in emb:
            return emb[name.lower()]
        vectors = []
        tokens = name.split()
        for token in tokens:
            if token in emb:
                vectors.append(emb[token])
            elif token.lower() in emb:
                vectors.append(emb[token.lower()])
        if len(vectors) > 1:
            return np.mean(np.array(vectors), axis=0)
        if len(vectors) == 0:
            print("Couldn't find {}".format(name))
            return None
        return vectors[0]  # Exactly one vector

    cuis = pd.read_csv('cui_to_icd9_manual.csv', index_col=0)
    cuis['emb'] = cuis['CUI name'].apply(process_cui)
    cuis = cuis.dropna(axis=0, subset=['emb'])
    cuis['emb'] = cuis['emb'].apply(lambda vec: ",".join([str(x) for x in vec]))
    cuis = cuis[['CUI', 'emb']]
    cuis.to_csv(output_file, sep='\t', index=False, header=False)


def extract_cui_embeddings_from_cui2vec_style_w2v(emb_file, output_file):
    emb = pd.read_csv(emb_file, sep='\t', index_col=0, header=None,
                      names=['word', 'vector'])
    emb['vector'] = emb['vector'].apply(lambda st: np.array([float(x) for x in st.split(",")]))
    emb = emb.to_dict(orient='dict')['vector']

    def process_cui(cui):
        if cui in emb:
            return emb[cui]
    cuis = pd.read_csv('code_mappings/all_cuis_with_icd9.txt', names=['CUI'])
    cuis['emb'] = cuis['CUI'].apply(process_cui)
    before = len(cuis)
    cuis = cuis.dropna(axis=0, subset=['emb'])
    print("found embedding for {}/{} CUIs from code_mappings/all_cuis_with_icd9.txt".format(len(cuis), before))
    cuis['emb'] = cuis['emb'].apply(lambda vec: ",".join([str(x) for x in vec]))
    cuis = cuis[['CUI', 'emb']]
    cuis.to_csv(output_file, sep='\t', index=False, header=False)


def extract_only_cuis_from_cui2vec_style_w2v(emb_file, output_file):
    emb = pd.read_csv(emb_file, sep='\t', index_col=False, header=None,
                      names=['word', 'vector'])
    emb = emb.dropna(subset=['word'])
    emb['vector'] = emb['vector'].apply(lambda st: np.array([float(x) for x in st.split(",")]))

    def is_cui(word):
        return word.startswith("C") and len(word) == 8 and word[1:].isdigit()

    before = len(emb)
    emb = emb[emb['word'].apply(is_cui)]
    print("{}/{} words look like CUIs".format(len(emb), before))
    emb = emb.rename({'word': 'CUI', 'vector': 'emb'}, axis=1)
    emb['emb'] = emb['emb'].apply(lambda vec: ",".join([str(x) for x in vec]))
    emb.to_csv(output_file, sep='\t', index=False, header=False)


def main():
    READ_MODELS = False
    TEST_GRAPH_STABILITY = False
    MAKE_GRAPH = False
    ANALYZE_NEIGHBOR_DIFF = False
    SHOW_2D_MODEL = False
    SAVE_MODELS = False

    ONLY_CONCEPTS = False
    MODE = "cui2vec_style"  # 'plain_text', 'cui2vec_style' or 'only_concepts'
    USE_GLOVE = False
    MIN_COUNT = 0
    MIN_COUNT_BEFORE_REP = 0 # 40
    VECTOR_SIZE = 40  # 50, 70, 100, 300
    TRAINING_ITERATIONS = 15  # 35
    WINDOW = 10 # 20
    TOPN = 10
    # create_pmid_to_participants() # writes the file: pmids_with_participants.tsv
    # merge_with_abstracts() # writes the file: abstracts_and_population.csv
    # tokenize_title_and_abstract(ONLY_CONCEPTS)

    fem_model_file_template = "embedding_models/fem{}.model"
    male_model_file_template = "embedding_models/male{}.model"
    neutral_model_file_template = "embedding_models/neutral{}.model"
    neutral_variation_model_file_template = "embedding_models/neutral_variation{}.model"

    graph_edges_template = "embedding_graphs/cuis_filtered_glove{}_edges.tsv"
    graph_nodes_template = "embedding_graphs/cuis_filtered_glove{}_nodes.tsv"

    df = read_tokenized_df(MODE)
    word_counter = word_count(df['tokenized'])
    num_words = len([w for w in word_counter.keys() if (word_counter[w] > MIN_COUNT_BEFORE_REP) and w.startswith('C') and len(w) == 8])
    print("mode: {}, unique words: {}, unique CUIs: {} ".format(MODE, len(word_counter), num_words))
    df['all_participants'] = df['male']+df['female']
    # print("df all participants:")
    # print(df['all_participants'].head())

    # iterations = [1, 2, 3, 4, 5]
    # iterations = [6, 7, 8, 9, 10]
    iterations = ['_cui2vec_style_w2v_copyrightfix_40']
    neutral_models, fem_models, male_models, neutral_var_models = [], [], [], []
    for iteration in iterations:
        fem_model_file = fem_model_file_template.format(iteration)
        male_model_file = male_model_file_template.format(iteration)
        neutral_model_file = neutral_model_file_template.format(iteration)
        neutral_variation_model_file = neutral_variation_model_file_template.format(iteration)

        fm = EmbedModelWrapper(use_glove=USE_GLOVE, min_count=MIN_COUNT,
                               vector_size=VECTOR_SIZE, iterations=TRAINING_ITERATIONS,
                               window=WINDOW)
        mm = EmbedModelWrapper(use_glove=USE_GLOVE, min_count=MIN_COUNT,
                               vector_size=VECTOR_SIZE, iterations=TRAINING_ITERATIONS,
                               window=WINDOW)
        neutral = EmbedModelWrapper(use_glove=USE_GLOVE, min_count=MIN_COUNT,
                                    vector_size=VECTOR_SIZE, iterations=TRAINING_ITERATIONS,
                                    window=WINDOW)
        neutral_var = EmbedModelWrapper(
            use_glove=USE_GLOVE, min_count=MIN_COUNT,
            vector_size=VECTOR_SIZE, iterations=TRAINING_ITERATIONS,
            window=WINDOW)

        if READ_MODELS:
            fm.load(fem_model_file)
            mm.load(male_model_file)
            neutral.load(neutral_model_file)
            neutral_var.load(neutral_variation_model_file)
        else:
            print("Training fem")
            fm.train(repeat_by_participants(df, 'female', word_counter, MIN_COUNT_BEFORE_REP))
            if SAVE_MODELS:
                fm.save(fem_model_file)

            print("Training male")
            mm.train(repeat_by_participants(df, 'male', word_counter, MIN_COUNT_BEFORE_REP))
            if SAVE_MODELS:
                mm.save(male_model_file)

            print("Training neutral")
            neutral.train(df['tokenized'])
            if SAVE_MODELS:
                neutral.save(neutral_model_file)

            #print("Training neutral with variations")
            #neutral_tokenized = repeat_by_participants(
            #    df, 'all_participants', word_counter, MIN_COUNT_BEFORE_REP)
            #with open("debug_neutral_with_repetition.txt", "w") as deb:
            #    deb.write("\n\n".join([str(x) for x in neutral_tokenized]))

            #neutral_var.train(neutral_tokenized)
            #if SAVE_MODELS:
            #    neutral_var.save(neutral_variation_model_file)

            fm.export_embeddings('embedding_models/female{}_emb.tsv'.format(iteration))
            mm.export_embeddings('embedding_models/male{}_emb.tsv'.format(iteration))
            neutral.export_embeddings('embedding_models/neutral{}_emb.tsv'.format(iteration))
            #neutral_var.export_embeddings('embedding_models/neutral_var{}_emb.tsv'.format(iteration))

        # to get neighbors of a specific cui
        #crohn_cui = 'C0010346'
        #fem_neighbors = prettify_neighbors(fm.most_similar(crohn_cui))
        #male_neighbors = prettify_neighbors(mm.most_similar(crohn_cui))
        #neutral_neighbors = prettify_neighbors(neutral_var.most_similar(crohn_cui))
        #print("female: {}, male: {}, neutral: {}".format(fem_neighbors, male_neighbors, neutral_neighbors))
        
        neutral_models.append(neutral)
        fem_models.append(fm)
        male_models.append(mm)
        #neutral_var_models.append(neutral_var)

        if ANALYZE_NEIGHBOR_DIFF:
            compare_models_by_neighbor_lists(
                fm, mm, neutral, neutral_var,
                output_file="differences_in_neighbors_with_neutral_variation.tsv")
        if TEST_GRAPH_STABILITY:
            graph_maker = GraphMaker(models=[fm, mm, neutral], descriptions=['female', 'male', 'neutral'])
            graph_maker.make_graph(edges_output_file=graph_edges_template.format(iteration),
                                   nodes_output_file=graph_nodes_template.format(iteration),
                                   similarity_threshold=None,  # don't filter edges by similarity
                                   topn=TOPN, choose_edges_by_neighbors=False,
                                   filter_nodes_by_icd9=True)
        if MAKE_GRAPH and iteration == iterations[0]:
            print("making a graph from a single iteration of word embeddings.")
            graph_maker = GraphMaker(models=[fm, mm, neutral_var],
                                     descriptions=['female', 'male', 'neutral'])
            s = "_single{}".format(iteration)
            graph_maker.make_graph(edges_output_file=graph_edges_template.format(s),
                                   nodes_output_file=graph_nodes_template.format(s),
                                   similarity_threshold=None,  # don't filter edges by similarity
                                   topn=TOPN, choose_edges_by_neighbors=False,
                                   filter_nodes_by_icd9=True)
            find_interesting_cui_pairs(edges_filename=graph_edges_template.format(s),
                                       nodes_filename=graph_nodes_template.format(s))
        if SHOW_2D_MODEL and iteration == iterations[0]:
            print("showing a single iteration on a 2D scatter.")
            cuis_fm = show_model_in_2d(fm, "embedding_models/female{}".format(iteration),
                             word_counter, use_tsne=False, title="female")
            cuis_mm = show_model_in_2d(mm, "embedding_models/male{}".format(iteration), 
                             word_counter, use_tsne=False, title="male")
            print("Are the concepts the same in the male and female model? {}".format((set(cuis_fm) == set(cuis_mm))))
            #show_model_in_2d(neutral_var,
            #                 "embedding_models/neutral_var{}".format(iteration),
            #                 word_counter,
            #                 word_subset=cuis_fm,
            #                 use_tsne=False, title="neutral_var")
            show_model_in_2d(neutral, "embedding_models/neutral{}".format(iteration),
                             word_counter, word_subset=cuis_fm, use_tsne=False, title="neutral")
    print("finishing with sys.exit")
    sys.exit()


    if TEST_GRAPH_STABILITY:
        how_many_shared_edges(1, 2)

    merged_neutral = ModelMerger(neutral_models)
    merged_fem = ModelMerger(fem_models)
    merged_male = ModelMerger(male_models)
    merged_neutral_var = ModelMerger(neutral_var_models)
    if ANALYZE_NEIGHBOR_DIFF:
        compare_models_by_neighbor_lists(
                merged_fem, merged_male, merged_neutral, merged_neutral_var,
                output_file="differences_in_neighbors_merged_with_neutral_variation.tsv")
    # if MAKE_GRAPH:
    #     print("making a graph from the merged models.")
    #     graph_maker = GraphMaker(models=[merged_fem, merged_male, merged_neutral_var],
    #                              descriptions=['female', 'male', 'neutral'])
    #     s = "_merged{}".format(len(iterations))
    #     graph_maker.make_graph(edges_output_file=graph_edges_template.format(s),
    #                            nodes_output_file=graph_nodes_template.format(s),
    #                            similarity_threshold=None,  # don't filter edges by similarity
    #                            topn=TOPN, choose_edges_by_neighbors=False,
    #                            filter_nodes_by_icd9=True)
    return merged_neutral, merged_fem, merged_male, merged_neutral_var


    #summarize_iterations(iterations=[1, 2, 3, 4, 5], output_file="summarize_5_models.tsv")

    # # compare_models_arithmetic(fm, "female", mm, "male", neutral, None, "word_diffs_arithmetic.tsv")


    # compare_iterations_by_order_preservation(1, 2)
    # compare_iterations_by_order_preservation(1, 3)
    # compare_iterations_by_order_preservation(2, 3)

    # output_file = "differences_in_models_cuis_3_filtered.tsv" if ONLY_CONCEPTS else "differences_in_models_all_words.tsv"
    # compare_models_by_neigbor_lists(fm, mm, neutral, output_file)

    # fm1 = train_embeddings_female("embedding_models/fem1_w2v.model")
    # fm2 = train_embeddings_female("embedding_models/fem2_w2v.model")
    # compare_models_by_neigbor_lists(fm1, fm2, neutral_model=None, output_file="diff_female_vs_female_10.tsv")

    # m1 = train_embeddings_neutral("embedding_models/neutral1_w2v.model")
    # m2 = train_embeddings_neutral("embedding_models/neutral2_w2v.model")
    # compare_models_by_neigbor_lists(m1, m2, neutral_model=None, output_file="diff_neutral_vs_neutral.tsv")

    # if READ_MODELS:
    #     d2v = Doc2Vec.load("embedding_models/first_doc2vec.model")
    # else:
    #     d2v = train_doc2vec(sample_times=5, description="first")
    # analyze_doc2vec(d2v, "d2v_single_words.tsv", None)
    #                 # "d2v_word_pairs.tsv")




if __name__ == "__main__":
    main()
