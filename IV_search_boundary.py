import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

cited = pd.read_csv('focal引用IPC.csv')
focal = pd.read_csv('focal IPC.csv')
focal_classify_df = focal.merge(df[["CITING_APPLN_ID", "IV_APPLN_app", "IV_APPLN_R_D"]], on="CITING_APPLN_ID", how="right")

#把IPC切割成前3
def slicing_3(x):
    return x[:3]
foo_3 = np.frompyfunc(slicing_3, 1, 1)
foo_3(cited.CITED_APPLN_IPC)

#将cited处理成向量
cited_IPC = list(cited.CITED_APPLN_IPC.value_counts().index)
vector = list(focal.CITING_APPLN_IPC.value_counts().index)
cited_IPC.sort()
vector.sort()

#长转宽
cited_wide = cited.pivot_table(index = 'CITING_APPLN_ID', columns = 'CITED_APPLN_IPC', values = 'CITED_APPLN_IPC_WEIGHT', fill_value=0)
focal_wide = focal.pivot_table(index = 'CITING_APPLN_ID', columns = 'CITING_APPLN_IPC', values = 'CITING_APPLN_IPC_WEIGHT', fill_value=0)

#为cited添加列，使之与focal同宽
col = list(set(vector)-set(cited_IPC))
df = pd.DataFrame(columns = col, index = cited_wide.index)
df = df.fillna(0)
cited_exp_wide = pd.concat([df, cited_wide], axis = 1)
focal_wide.shape[1] == cited_exp_wide.shape[1] #已经一致

#计算focal_wide
union_citing_appln_id = list(set(focal.CITING_APPLN_ID.values) & set(cited.CITING_APPLN_ID.values))
cited_union = cited_exp_wide.loc[union_citing_appln_id]
focal_union = focal_wide.loc[union_citing_appln_id]

#求距离
def calCosDist(x, y):
    return 1 - np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

dist_df = pd.DataFrame(index = union_citing_appln_id, columns = ['dist'])
for id in union_citing_appln_id:
    x = cited_union.loc[id]
    y = focal_union.loc[id]
    dist_df.loc[id] = calCosDist(x, y)
dist_df['CITING_APPLN_ID'] = dist_df.index