import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import string
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
import gc

base_path = r"IV_R&D"
base_path_abstract = r"DV_novelty"
abstr = pd.read_csv(os.path.join(base_path_abstract, "pat_abstract.csv"))
index = abstr.APPLN_ID

# <editor-fold desc="preprocessing">
#①lower case
text = abstr['APPLN_ABSTRACT']
text = text.apply(str.lower)

#②remove puctuations
def rmpunctuation(text):
    return re.sub(r"[%s]+" %string.punctuation, " ", text)
foo = np.frompyfunc(rmpunctuation, 1, 1)
text = foo(text)
def removehuanhang(t):
    return t.replace('\\n', ' ')
foo2 = np.frompyfunc(removehuanhang, 1, 1)
text = foo2(text)

#③remove stop words
stopword = pd.read_csv(os.path.join(base_path_abstract, "stopword.csv"))
data_stopWord = stopword.iloc[:, 0]
data_stopWord = list(data_stopWord)
data_stopWord = list(map(str.lower, data_stopWord))
text_wth_stop = []
for each in text:
    a = ' '.join([w for w in each.split() if w not in data_stopWord])
    text_wth_stop.append(a)
# </editor-fold>

# <editor-fold desc="classify R&D innovations based on keywords">
r_dwordlst = ['Transaction inclusion', 'lightweight', 'updating period',
              'Side chains', 'transaction completion', 'operation efficiency',
              'Blockchain size', 'data storage', 'Auditability', 'visibility',
              'transparency', 'mitigating risk', 'prevent hacking',
              'Confidentiality', 'personal information',
              'User interface', 'user experience', 'customized',
              'Power consumption', 'quantity of electricity', 'resource waste']
r_dwordlst = list(map(str.lower, r_dwordlst))

# patent abstract directly has keyword(s)
counted_rd = 0
for tx in text_wth_stop:
    for wd in r_dwordlst:
        if wd in tx:
            counted_rd +=1
            break
print(f'#R&D: {counted_rd}')
# </editor-fold>

# <editor-fold desc="classify application innovations based on keywords">
app_wordlst = ['Cryptocurrency transaction', 'bitcoin transfer',
               'cryptocurrency payment', 'Tradin', 'prediction markets',
               'settlement', 'asset management', 'Blockchain-based identity',
               'KYC', 'IoT', 'internet-connected devices',
               'Supply chain', 'logistics', 'verification of goods',
               'Digital content', 'permission of content',
               'Healthcare', 'medical records',
               'Certificate authentication', 'certificate authority',
               'education certificate', 'Energy management',
               'photovoltaic power transaction', 'Tokens',
               'tokenization of assets', 'Voting', 'voting data']
app_wordlst = list(map(str.lower, app_wordlst))
sum(['supply chain' in x for x in text_wth_stop])

counted_app = 0
for tx in text_wth_stop:
    for wd in app_wordlst:
        if wd in tx:
            counted_app +=1
            break
print(f'#application: {counted_app}')
# </editor-fold>

# <editor-fold desc="get roBERTa embeddings">
# first remove keywords before put into model
def removeKeyword(text_wth_stop_lst):
    abstr_lst = []
    for sect in text_wth_stop_lst:
        lst = []
        for wd in sect.split():
            if (wd not in r_dwordlst) and (wd not in app_wordlst):
                lst.append(wd)
        abstr_lst.append(' '.join(lst))
    return abstr_lst
abstr_lst = removeKeyword(text_wth_stop)
len(abstr_lst) == len(text_wth_stop)

import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
def toSentenceEmbedding(atitle):
    tokens = roberta.encode(atitle)
    last_layer_features = roberta.extract_features(tokens)
    vec = last_layer_features.detach().numpy()
    vec = vec.squeeze()  # 去掉冗余维度
    # word level-->sentence level:加权平均
    sent = (np.sum(vec, axis=0) / vec.shape[0]).reshape((1, 768))  # (1, 768)
    return sent
toSentVect = np.frompyfunc(toSentenceEmbedding, 1, 1)

def toSentenceEmbedding_trunc(atitle):
    '''truncate to <=512 tokens'''
    tokens = roberta.encode(atitle)
    tokens = tokens[1:-1] # 首尾token去掉 (标识符，不代表特征)
    tokens = tokens if len(tokens) <= 512 else tokens[:512]  # truncate
    last_layer_features = roberta.extract_features(tokens)
    vec = last_layer_features.detach().numpy()
    vec = vec.squeeze()  # 去掉冗余维度
    # word level-->sentence level:加权平均
    sent = (np.sum(vec, axis=0) / vec.shape[0]).reshape((1, 768))  # (1, 768)
    return sent
toSentVect2 = np.frompyfunc(toSentenceEmbedding_trunc, 1, 1)

ttembeddings = toSentVect2(abstr_lst)
df = pd.DataFrame(np.array(ttembeddings_lst).squeeze())
df.shape
df.to_csv(r"roberta_wthkeyword.csv", index = False)
# </editor-fold>

df = pd.read_csv(os.path.join(base_path, "roberta_wthkeyword.csv"))
df["APPLN_ID"] = index

# <editor-fold desc="patents that can be classifed using keywords method">
APPLN_TYPE = []
for i in range(len(text_wth_stop)):
    tx = text_wth_stop[i]
    for wd in app_wordlst: #检查application keyword
        if wd in tx:
            APPLN_TYPE.append(True)
            break
    if len(APPLN_TYPE) == i+1: #说明找到application keyword
        continue
    for wd in r_dwordlst: #否则
        if wd in tx:
            APPLN_TYPE.append(True)
            break
    if len(APPLN_TYPE) == i+1: #说明找到R&D keyword
        continue
    APPLN_TYPE.append(False) #否则，先标记为未知
APPLN_TYPE.count(True)
type_df = pd.DataFrame({"APPLN_ID": index,
                        "APPLN_TYPE": APPLN_TYPE})
# </editor-fold>

# <editor-fold desc="remaining patents">
test_df = type_df[type_df['APPLN_TYPE'] == False]
train_df = type_df[type_df['APPLN_TYPE'] == True]
test_emb_df = pd.merge(test_df, df, how = 'left', on="APPLN_ID")
abstr_df = pd.DataFrame({"APPLN_ID": index,
                        "ABSTRACT": text_wth_stop})
test_abstr_df = pd.merge(test_df, abstr_df, how = 'left', on="APPLN_ID")
train_abstr_df = pd.merge(train_df, abstr_df, how = 'left', on="APPLN_ID")
# </editor-fold>

# <editor-fold desc="R&D model">
APPLN_R_D = []
for tx in train_abstr_df['ABSTRACT']:
    for wd in r_dwordlst: #检查R&D keyword
        if wd in tx:
            APPLN_R_D.append(True)
            break
    else:
        APPLN_R_D.append(False) # 其他在train中，都赋为false

train_r_d_df = train_df
train_r_d_df['APPLN_R_D'] = APPLN_R_D
train_r_d_df = train_r_d_df.drop('APPLN_TYPE', axis = 1)
train_r_d_df_emb = pd.merge(train_r_d_df, df, how = 'left', on = 'APPLN_ID')

y = train_r_d_df_emb.APPLN_R_D
y = y.astype('int')
X = train_r_d_df_emb.iloc[:, 2:] #有768维

# <editor-fold desc="SVM+optimize param">
# define search space
params = dict()
params['C'] = (1e-6, 100.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['degree'] = (1,5)
params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the search
search = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X, y)
# report the best result
print(search.best_score_)
print(search.best_params_)
r_d_param_dict = dict([('C', 0.7020022367229055), ('degree', 4), ('gamma', 0.013600210488809198), ('kernel', 'rbf')])

# retrain model
clf_r_d_svm = SVC(random_state=1).set_params(**r_d_param_dict)
clf_r_d_svm.fit(X, y)
y_pred_svm = clf_r_d_svm.predict(X)
np.sum(y_pred_svm == y)/len(y) # acc

# <editor-fold desc="混淆矩阵">
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y, y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,)
disp.plot()
plt.title("R&D SVM")
plt.show()
# </editor-fold>
# </editor-fold>


# <editor-fold desc="prediction">
y_test_pred_r_d = clf_r_d_svm.predict_proba(test_emb_df.iloc[:, 2:])
y_test_pred_r_d = y_test_pred_r_d[:, 1] # 取出1的概率
test_df = test_df.drop('APPLN_TYPE', axis = 1)
test_df['APPLN_R_D']=y_test_pred_r_d
# train也做相同处理
train_df = train_df.drop('APPLN_TYPE', axis = 1).drop('APPLN_R_D', axis = 1)
train_df['APPLN_R_D'] = y_pred_svm # 把train的标签也换成概率
# 把test和train重新按行堆叠
df_r_d = pd.concat([train_df, test_df])
pd.value_counts(df_r_d.APPLN_R_D)
df_r_d.to_csv(r'R_D_prob_classify.csv', index = False)
# </editor-fold>
# </editor-fold>

# <editor-fold desc="app model">
APPLN_app = []
for tx in train_abstr_df['ABSTRACT']:
    for wd in app_wordlst: #检查R&D keyword
        if wd in tx:
            APPLN_app.append(True)
            break
    else:
        APPLN_app.append(False) # 其他在train中，都赋为false

APPLN_app.count(True)
train_app_df = train_df
train_app_df['APPLN_app'] = APPLN_app
train_app_df = train_app_df.drop(['APPLN_TYPE', 'APPLN_R_D'], axis = 1)
train_app_df_emb = pd.merge(train_app_df, df, how = 'left', on = 'APPLN_ID')

# <editor-fold desc="SVM+optimize param">
# define search space
params = dict()
params['C'] = (1e-6, 100.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['degree'] = (1,5)
params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the search
search = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv)
# perform the search
search.fit(X, y)
# report the best result
print(search.best_score_)
print(search.best_params_)
# </editor-fold>

app_param_dict = dict([('C', 0.04804694302538957), ('degree', 5), ('gamma', 100.0), ('kernel', 'linear')])

# retrain model
clf_app_svm = SVC(random_state=1).set_params(**app_param_dict)
clf_app_svm.fit(X, y)
y_pred_svm = clf_app_svm.predict(X)
np.sum(y_pred_svm == y)/len(y) # acc

# <editor-fold desc="混淆矩阵">
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y, y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,)
disp.plot()
plt.title("application SVM")
plt.show()
# </editor-fold>

# <editor-fold desc="prediction">
y_test_pred_app = clf_app_svm.predict_proba(test_emb_df.iloc[:, 2:])
y_test_pred_app = y_test_pred_app[:, 1] # 取出1的概率
test_df = test_df.drop('APPLN_TYPE', axis = 1)
test_df['APPLN_app']=y_test_pred_app
# train也做相同处理
train_df = train_df.drop('APPLN_TYPE', axis = 1).drop('APPLN_app', axis = 1)
train_df['APPLN_app'] = y_pred_svm #把train的标签也换成概率
# 把test和train重新按行堆叠
df_app = pd.concat([train_df, test_df])
pd.value_counts(df_app.APPLN_R_D)
df_app.to_csv(r'APP_prob_classify.csv', index = False)
# </editor-fold>
# </editor-fold>