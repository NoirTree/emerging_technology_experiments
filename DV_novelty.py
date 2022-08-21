import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import string
import nltk
import re
import sys

base_path = r"DV_novelty"
abstr = pd.read_csv(os.path.join(base_path, "pat_abstract.csv"))
index = abstr.APPLN_ID

# <editor-fold desc="preprocess">
#①转小写
text = abstr['APPLN_ABSTRACT']
text = text.apply(str.lower)

#②去掉标点符号:
def rmpunctuation(text):
    return re.sub(r"[%s]+" %string.punctuation, " ", text)
foo = np.frompyfunc(rmpunctuation, 1, 1)
text = foo(text)
#仍然有\\n这样的
def removehuanhang(t):
    return t.replace('\\n', ' ')
foo2 = np.frompyfunc(removehuanhang, 1, 1)
text = foo2(text)

#③去停
stopword = pd.read_csv(os.path.join(base_path, "stopword.csv")) #加载停用词表
data_stopWord = stopword.iloc[:, 0]
data_stopWord = list(data_stopWord)
data_stopWord = list(map(str.lower, data_stopWord))#停词表也转小写（因为abstr转过小写了）
text_wth_stop = []
for each in text:
    a = ' '.join([w for w in each.split() if w not in data_stopWord])
    text_wth_stop.append(a)

#④tokenization (Copy from web)
# <editor-fold desc="创建词表">
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens
totalvocab_tokenized = []
totalvocab_stemmed = []
for i in text_wth_stop:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
vocab_frame = dict(zip(totalvocab_tokenized, totalvocab_stemmed))
# </editor-fold>

# <editor-fold desc="按照词表，把每句话各个单词词干化">
def tokenize_a_line(text):
    test_wd = ''
    for x in text.split():
        if x.isnumeric() is True:
            # 如果x是纯数字，那么直接去掉（但, 如"happy197"这种需要保留）
            pass
        elif x in vocab_frame:
            # 如果在vocab_frame中，就保存对应的stemmed结果
            test_wd = test_wd + vocab_frame[x] + ' '
        else:
            # 如果x不在vocab_frame中--保留
            test_wd = test_wd + x + ' '
    return test_wd.strip() # 两侧多余的空白去掉
foo3 = np.frompyfunc(tokenize_a_line, 1, 1)
# </editor-fold>

text_token = []
for line in text_wth_stop:
    text_token.append(foo3(line))
# </editor-fold>

# <editor-fold desc="LDA">
# <editor-fold desc="vectorizer">
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df =0.02, token_pattern='[a-zA-Z0-9]{3,}') #限制长度>3
data_vectorized = vectorizer.fit_transform(text_token)
# </editor-fold>

# <editor-fold desc="train model">
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
lda_model = LatentDirichletAllocation(n_components=20,               # Number of topics (try different)
                                      max_iter=100,               # Max learning iterations
                                      learning_method='online',
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_vectorized)
# </editor-fold>

# <editor-fold desc="evaluation by visualization">
# 作图评估：①圆圈大②不重叠
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.display(panel)
# </editor-fold>
# </editor-fold>
