import re
import string
import time
import pandas as pd

from metamap_wrapper import extract_metamap_concepts, replace_words_with_concepts_in_text
import text_utils as tu


def should_keep_sentence(sentence):
    blacklist = ['http', 'https', 'url', 'www', 'clinicaltrials.gov', 'copyright', 'funded by', 'published by', 'subsidiary', '©', 'all rights reserved']
    s = sentence.lower()
    for w in blacklist:
        if w in s:
            return False
    # re, find NCTs
    if len(re.findall('nct[0-9]+', s)) > 0:
        return False
    if len(s) < 10:
        return False
    return True


def clean_abstracts(df):
    # filter sentences
    df['sentences'] = df['title_and_abstract'].apply(tu.split_abstract_to_sentences)
    d = {'total': 0, 'remaining': 0}

    def pick_sentences(sentences):
        new_sents = [sent for sent in sentences if should_keep_sentence(sent)]
        d['total'] += len(sentences)
        d['remaining'] += len(new_sents)
        return new_sents

    def join_to_abstract(sentences):
        return ' '.join(sentences)

    df['sentences'] = df['sentences'].apply(pick_sentences)
    df['title_and_abstract'] = df['sentences'].apply(join_to_abstract)
    print(f"kept {d['remaining']}/{d['total']} sentences without blacklisted words")
    return df


def read_abstracts_file(fname):
    df = pd.read_csv(fname, index_col=0)
    df['title'] = df['title'].fillna('')
    df['title'] = df['title'].apply(lambda x: x.strip('[]'))
    df['title_and_abstract'] = df['title'] + ' ' + df['abstract']
    df = df.dropna(axis=0, subset=['title_and_abstract'])
    df = clean_abstracts(df)
    return df


def tokenize_title_and_abstract(
        mode='only_concepts',
        output_file="abstracts_and_population_output.csv"):
    if mode == 'cui2vec_style':
        print("better use tokenize_abstracts_for_cui2vec")
        return
    def preprocess_row(row):
        text = " ".join([row['title'], row['abstract']])
        text = text.replace(";", " ")
        if mode == 'only_concepts':
            f = open("cui_table.tsv", "a")
            concepts = extract_metamap_concepts(text)
            cuis = [concept['cui'] for concept in concepts]
            for concept in concepts:
                f.write("{}\t{}\t{}\n".format(concept['cui'], concept['semtypes'], concept['preferred_name']))
            return cuis
        return tu.break_and_lemmatize(text)
    f = open("cui_table.tsv", "w")
    f.write("{}\t{}\t{}\n".format('cui', 'semtypes', 'preferred_name'))
    df = read_abstracts_file("abstracts_and_population.csv")
    print("tokenizing")
    df['tokenized'] = df.progress_apply(preprocess_row, axis=1)
    df.to_csv(output_file)


def tokenize_abstracts_for_cui2vec(csv_path):
    f = open("cui_table_for_cui2vec.tsv", "w")
    tracking = {'start': time.time(), 'done': 0, 'total': 0}

    def process(sentences):
        new_sentences = []
        for text in sentences:
            # lowercase, replace punctuation with spaces.
            text = text.lower().translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
            # remove unicode characters.
            text = "".join([x if ord(x) < 128 else ' ' for x in text])
            # remove too many spaces in a row
            text = " ".join(text.split())
            # replace concepts with their identifier
            new_text, concepts = replace_words_with_concepts_in_text(text)
            new_sentences.append(new_text)
            for concept in concepts:
                f.write("{}\t{}\t{}\n".format(
                    concept['cui'], concept['semtypes'], concept['preferred_name']))
            tracking['done'] += 1
            if tracking['done'] % 100 == 0:
                print("***********************************************")
                print("Done: {}/{} in {:.2f} minutes".format(
                    tracking['done'], tracking['total'],
                    (time.time()-tracking['start'])/60))
        return new_sentences
    abstracts = read_abstracts_file(csv_path)
    tracking['total'] = len(abstracts)
    abstracts['tokenized_sents'] = abstracts['sentences'].apply(process)
    abstracts.to_csv('abstracts_and_population_tokenized_for_cui2vec_copyrightfix_sent_sep.csv')
    f.close()
    df = pd.read_csv("cui_table_for_cui2vec.tsv", sep="\t", names=['cui', 'semtypes', 'name'])
    df.drop_duplicates(subset='cui', inplace=True)
    df.to_csv("cui_table_for_cui2vec.tsv", sep="\t")


def clean_cui_table():
    df = pd.read_csv("cui_table.tsv", sep="\t")
    print("read {} CUI records.".format(len(df)))
    df.drop_duplicates(subset='cui', inplace=True, ignore_index=True)
    print("remaining {} CUI records without duplicates.".format(len(df)))
    df.to_csv("cui_table_no_dups.tsv", sep="\t")

if __name__ == '__main__':
    tokenize_abstracts_for_cui2vec('abstracts_and_population.csv')
