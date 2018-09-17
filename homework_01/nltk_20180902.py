
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import PlaintextCorpusReader

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.rc('xtick', labelsize=20)     
plt.rc('ytick', labelsize=20)


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... lexical diversity score - as given in nltk site
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def lexical_diversity(my_text_data):
    tokens = len(my_text_data)
    types = len(set(my_text_data))
    diversity_score = types / tokens
    return diversity_score

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... some directory and file name definitions
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

files = ".*\.txt"
home_dir = "/home/mcdevitt/_ds/_smu/msds_7337_nlp/homework_01/"
corpus_root = "./texts"
plot_dir = "./plots/"

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... read in texts / assemble corpus for evaluation
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

readers = PlaintextCorpusReader(corpus_root, files)

readers.fileids()

corpus = nltk.Text(readers.words())

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... create table to accumulate summary data
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

results_tbl = pd.DataFrame(columns =
    ['text_name',
     'num_chars',
     'num_words',
     'num_sents',
     'num_vocab',
     'tokens',
     'types',
     'lex_div'])

i_index = []
i_index = 0

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... loop thru each text to assemble metrics
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

print("Some basic statistics\n")

for fileid in readers.fileids():
    num_chars = len(readers.raw(fileid))
    num_words = len(readers.words(fileid))
    num_sents = len(readers.sents(fileid))
    tokens = len(readers.words(fileid))
    types = len(set(readers.words(fileid)))
    num_vocab = len(set(w.lower() for w in readers.words(fileid)))
#    print(round(num_chars/num_words, 2),
#          round(num_words/num_sents, 2),
#          round(num_words/num_vocab, 2), fileid)    
    rtxt = readers.words(fileid)
    ldiv = lexical_diversity(rtxt)
    print(round(ldiv, 4), fileid)

    table_data = {
     'text_name' : fileid,
     'num_chars' : num_chars,
     'num_words' : num_words,
     'num_sents' : num_sents,
     'num_vocab' : num_vocab,
     'tokens' : tokens,
     'types' : types,
     'lex_div' : ldiv
    } 

    df_tbl = pd.DataFrame(table_data,
        columns = ['text_name',
             'num_chars',
             'num_words',
             'num_sents',
             'num_vocab',
             'tokens',
             'types',
             'lex_div'],
    index = [i_index + 1])
    i_index += 1
    results_tbl = results_tbl.append(df_tbl)

results_tbl = results_tbl.sort_values(results_tbl.columns[7])
print('Results_tbl - sorted by col 8')
results_tbl

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... normalize each column by max column value
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

results = results_tbl.copy()
results = results.sort_values(results.columns[7], ascending = False)

df = results.iloc[:, 1:8]
df_nrml = df / df.max()

df_labels = results.iloc[:, 0]
df_labels

results = pd.concat([df_labels, df_nrml], axis = 1)

results['vocab_ldiv'] = results.apply(lambda x: x.lex_div * (x.num_words), axis=1)
results['vocab_ldiv'] = results['vocab_ldiv'] / results['vocab_ldiv'].max()

print('Results - ')
results

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... plot - lexical diversity scores - sorted in ascending TTR
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

N = 9
ind = np.arange(N) 
width = 0.5

_ = plt.figure(figsize = (18, 10))
offset = 0
plt.bar(ind + offset, results_tbl['lex_div'], width, label='Lex_Div', color = 'tomato')

plt.xticks(ind + width / 2, results_tbl['text_name'])
plt.xticks(rotation=90)
plt.legend(loc='upper left')
plt.title('Lexical Diversity - Selected Readers', fontsize = '30')

axes = plt.gca()
axes.set_ylim([0, 0.2])

plt.savefig(plot_dir + 'nltk_readers_ttr.png')
plt.show()

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... plot - vocabulary size - sorted in ascending TTR
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

N = 9
ind = np.arange(N) 
width = 0.4

_ = plt.figure(figsize = (18, 10))
offset = width / 2
plt.bar(ind + offset, results_tbl['num_vocab'], width, label='Vocab', color = 'orchid')
plt.bar(ind + offset + width, results_tbl['types'], width, label='Types', color = 'cornflowerblue')

plt.xticks(ind + width / 2, results_tbl['text_name'])
plt.xticks(rotation=90)
plt.legend(loc='upper left')
plt.title('Vocabulary Size - Selected Readers', fontsize = '30')

axes = plt.gca()

plt.savefig(plot_dir + 'nltk_readers_vocab.png')
plt.show()

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... plot - vocabulary size - vs other stats - scatter plots
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

N = 9
ind = np.arange(N) 
width = 0.4

_ = plt.figure(figsize = (18, 10))
offset = width / 2
plt.scatter(results_tbl['num_vocab'], np.log10(results_tbl['types']), color = 'orchid', s = 100)
plt.scatter(results_tbl['num_vocab'], np.log10(results_tbl['tokens']), color = 'slateblue', s = 100)
plt.scatter(results_tbl['num_vocab'], np.log10(results_tbl['num_sents']), color = 'cornflowerblue', s = 100)
plt.scatter(results_tbl['num_vocab'], np.log10(results_tbl['num_chars']), color = 'darkcyan', s = 100)

plt.legend(loc='upper left', fontsize = '25')
plt.title('Vocabulary Size - Selected Readers', fontsize = '30')
plt.xlabel('Number of Vocabulary Words', fontsize = '25')
plt.ylabel('Corresponding Statistics (log10 scale)', fontsize = '25')

axes = plt.gca()
axes.set_ylim([1, 7])

plt.savefig(plot_dir + 'nltk_metrics_vs_vocab.png')
plt.show()

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... plot comparison of all (normalized) metrics - sorted in descending TTR
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
results = results.sort_values(results_tbl.columns[7], ascending = False)
N = 9
ind = np.arange(N) 
width = 0.143

_ = plt.figure(figsize = (18, 8))
offset = width / 2
plt.bar(ind + offset, results['lex_div'], width, label='Lex_Div', color = 'tomato')
plt.bar(ind + offset + width, results['num_chars'], width, label='Chars', color = 'dodgerblue')
plt.bar(ind + offset + width*2, results['num_words'], width, label='Words', color = 'slateblue')
plt.bar(ind + offset + width*3, results['num_sents'], width, label='Sentences', color = 'cornflowerblue')
plt.bar(ind + offset + width*4, results['num_vocab'], width, label='Vocab', color = 'orchid')
plt.bar(ind + offset + width*5, results['vocab_ldiv'], width, label='Vocab_LDiv', color = 'darkcyan')

plt.xticks(ind + width / 2, results['text_name'])
plt.xticks(rotation=90)
plt.legend(loc='upper left')
plt.title('Normalized Characteristics Comparison (Lex Div Sorted)', fontsize = '30')


axes = plt.gca()
axes.set_ylim([-0.1, 1.1])

plt.savefig(plot_dir + 'nltk_readers_ttr_normalized.png')
plt.show()

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... plot comparison of all (normalized) metrics - 
# ... sorted in descending TTR*num_tokens
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

results = results.sort_values(results_tbl.columns[8], ascending = False)
N = 9
ind = np.arange(N) 
width = 0.143

_ = plt.figure(figsize = (18, 8))

offset = width / 2
plt.bar(ind + offset, results['lex_div'], width, label='Lex_Div', color = 'tomato')
plt.bar(ind + offset + width, results['num_chars'], width, label='Chars', color = 'dodgerblue', alpha = 0.9)
plt.bar(ind + offset + width*2, results['num_words'], width, label='Words', color = 'slateblue', alpha = 0.9)
plt.bar(ind + offset + width*3, results['num_sents'], width, label='Sentences', color = 'cornflowerblue', alpha = 0.9)
plt.bar(ind + offset + width*4, results['num_vocab'], width, label='Vocab', color = 'orchid', alpha = 0.9)
plt.bar(ind + offset + width*5, results['vocab_ldiv'], width, label='Vocab_LDiv', color = 'c')

plt.xticks(ind + width / 2, results['text_name'])
plt.xticks(rotation=90)
plt.legend(loc='upper right')
plt.title('Normalized Characteristics Comparison (Lex Div * Vocab)', fontsize = '30')

axes = plt.gca()
axes.set_ylim([-0.1, 1.1])

plt.savefig(plot_dir + 'nltk_readers_ttrxtokens_normalized.png')
plt.show()

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... end_of_file
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



