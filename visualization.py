from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def scatter_with_hover_annotation(x, y, names, color_indices):
    fig, ax = plt.subplots()

    sc = plt.scatter(x, y, s=50, c=color_indices)

    annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "\n".join([names[n] for n in ind["ind"]])
        # text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
        #                        " ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

def read_abstracts():
    df = pd.read_csv('abstracts_and_population_tokenized_for_cui2vec_copyrightfix_sent_sep.csv', index_col=0)
    df = df.dropna(subset=['tokenized_sents'])
    df['tokenized_sents'] = df['tokenized_sents'].apply(literal_eval)
    df['tokenized'] = df['tokenized_sents'].apply(lambda x: (" ".join(x)).split())

    df2 = pd.read_csv('abstracts_population_date_topics.csv', index_col=0)
    merged = df.merge(df2[['date', 'mesh_headings']], left_index=True, right_index=True)
    merged['year'] = merged['date'].apply(lambda x: int(x[:4]))
    merged['female_participant_ratio'] = merged['female'] / (merged['female'] + merged['male'])
    merged['female_participant_ratio'] = merged['female'] / (merged['female'] + merged['male'])
    return merged

def build_cuis_and_times(abstracts_df):
    cuis_and_times = defaultdict(lambda: defaultdict(list))
    for i, r in abstracts_df.iterrows():
        for word in set(r['tokenized']):
            cuis_and_times[word][r['year']].append(r['female_participant_ratio'])
    return cuis_and_times


def barplot_for_cui(cui=None, year_to_ratios=None, disease_name=None, prevalence=None):
    if cui is None and year_to_ratios is None:
        print("can't make graph without cui and year_to_ratios")
        return
    if year_to_ratios is None:
        merged = read_abstracts()
        cuis_and_times = build_cuis_and_times(merged)
        year_to_ratios = cuis_and_times[cui]
    # now! to make ze graph!
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    year_range = range(2008, 2019)
    xticks = list(range(1, len(year_range)+1))
    ax.boxplot([year_to_ratios[year] for year in year_range], showfliers=False, showmeans=True)
    ax.yaxis.grid(True)
    ax.set_xticks(xticks)
    #if disease_name is not None:
    #    ax.set_xlabel(disease_name)
    if prevalence is not None:
        ax.axhline(y=prevalence, color='b', linestyle='--')
    plt.ylim(0, 1)
    plt.xticks(xticks, year_range)
    plt.rcParams.update({'font.size': 22})
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()


def csv_to_latex_table(csv_file, columns=None):
    df = pd.read_csv(csv_file)
    if columns is not None:
        df = df[columns]
    else:
        columns = df.columns
    print(df.head())
    print(r"\begin{table}[h]")
    print(r"\centering")
    centered_columns = "|"
    for i in range(len(columns)-1):
        centered_columns += " c |"
    print(r"\begin{tabular}{| l " + centered_columns + "}")
    print(r"\hline")
    print(" & ".join(columns) + r" \\")
    for i, r in df.iterrows():
        print(" & ".join([str(r[c]) for c in columns]) + r" \\")
        print(r"\hline")
    print(
        """
        \end{tabular}
        \caption{Your caption here}
        \label{your label here}
        \end{table}
        """)

