#!/usr/bin/env python
# coding: utf-8

# # Import des librairies

# In[1]:


import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager


# In[2]:


from lightgbm import LGBMClassifier


# In[3]:


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[4]:


import re


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_auc_score, fbeta_score, make_scorer


# In[6]:


import mlflow
import mlflow.lightgbm


# In[7]:


# Pour l'API et le dashboard
import pickle
import requests
import json


# ---
# ---
# ---
# # Fonctions

# In[8]:


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# In[9]:


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= False)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# In[10]:


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv(
        'C:/Users/valev/OneDrive/Documents/Formation DS/Projet 7/Données/Projet+Mise+en+prod+-+home-credit-default-risk/application_train.csv',
        nrows=num_rows)
    test_df = pd.read_csv(
        'C:/Users/valev/OneDrive/Documents/Formation DS/Projet 7/Données/Projet+Mise+en+prod+-+home-credit-default-risk/application_test.csv',
        nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Below code gives percentage of null in every column
    na_percentage = df.isna().sum() / df.shape[0] * 100
    na_percentage_test = test_df.isna().sum() / test_df.shape[0] * 100
    # Below code gives list of columns having more than 50% missing
    col_to_drop = na_percentage[na_percentage > 50].keys()
    col_to_drop_test = na_percentage_test[na_percentage_test > 50].keys()
    df = df.drop(col_to_drop, axis=1)
    test_df = test_df.drop(col_to_drop_test, axis=1)
    # Replace missing values
    for col in df.select_dtypes(['float', 'int64']):  # ---Applying only on numerical features
        df[col].fillna(df[col].median(), inplace=True)
    for col in test_df.select_dtypes(['float', 'int64']):  # ---Applying only on numerical features
        test_df[col].fillna(test_df[col].median(), inplace=True)

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df


# In[11]:


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv(
        'C:/Users/valev/OneDrive/Documents/Formation DS/Projet 7/Données/Projet+Mise+en+prod+-+home-credit-default-risk/bureau.csv',
        nrows=num_rows)
    bb = pd.read_csv(
        'C:/Users/valev/OneDrive/Documents/Formation DS/Projet 7/Données/Projet+Mise+en+prod+-+home-credit-default-risk/bureau_balance.csv',
        nrows=num_rows)

    # Below code gives percentage of na in every column
    na_percentage = bureau.isna().sum() / bureau.shape[0] * 100
    na_percentage_bb = bb.isna().sum() / bb.shape[0] * 100
    # Below code gives list of columns having more than 50% missing
    col_to_drop = na_percentage[na_percentage > 50].keys()
    col_to_drop_bb = na_percentage_bb[na_percentage_bb > 20].keys()
    bureau = bureau.drop(col_to_drop, axis=1)
    bb = bb.drop(col_to_drop_bb, axis=1)
    # Replace missing values
    for col in bureau.select_dtypes(['float', 'int64']):  # ---Applying only on numerical features
        bureau[col].fillna(bureau[col].median(), inplace=True)
    for col in bb.select_dtypes(['float', 'int64']):  # ---Applying only on numerical features
        bb[col].fillna(bb[col].median(), inplace=True)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        # 'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        # 'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# In[12]:


# Preprocess previous_applications.csv
def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv(
        'C:/Users/valev/OneDrive/Documents/Formation DS/Projet 7/Données/Projet+Mise+en+prod+-+home-credit-default-risk/previous_application.csv',
        nrows=num_rows)

    # Below code gives percentage of na in every column
    na_percentage = prev.isna().sum() / prev.shape[0] * 100
    # Below code gives list of columns having more than 50% missing
    col_to_drop = na_percentage[na_percentage > 50].keys()
    prev = prev.drop(col_to_drop, axis=1)
    # Replace missing values
    for col in prev.select_dtypes(['float', 'int64']):  # ---Applying only on numerical features
        prev[col].fillna(prev[col].median(), inplace=True)

    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        # 'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        # 'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# In[13]:


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv(
        'C:/Users/valev/OneDrive/Documents/Formation DS/Projet 7/Données/Projet+Mise+en+prod+-+home-credit-default-risk/POS_CASH_balance.csv',
        nrows=num_rows)

    # Below code gives percentage of na in every column
    na_percentage = pos.isna().sum() / pos.shape[0] * 100
    # Below code gives list of columns having more than 50% missing
    col_to_drop = na_percentage[na_percentage > 50].keys()
    pos = pos.drop(col_to_drop, axis=1)
    # Replace missing values
    for col in pos.select_dtypes(['float', 'int64']):  # ---Applying only on numerical features
        pos[col].fillna(pos[col].median(), inplace=True)

    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# In[14]:


# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv(
        'C:/Users/valev/OneDrive/Documents/Formation DS/Projet 7/Données/Projet+Mise+en+prod+-+home-credit-default-risk/installments_payments.csv',
        nrows=num_rows)

    # Below code gives percentage of na in every column
    na_percentage = ins.isna().sum() / ins.shape[0] * 100
    # Below code gives list of columns having more than 50% missing
    col_to_drop = na_percentage[na_percentage > 50].keys()
    ins = ins.drop(col_to_drop, axis=1)
    # Replace missing values
    for col in ins.select_dtypes(['float', 'int64']):  # ---Applying only on numerical features
        ins[col].fillna(ins[col].median(), inplace=True)

    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# In[15]:


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv(
        'C:/Users/valev/OneDrive/Documents/Formation DS/Projet 7/Données/Projet+Mise+en+prod+-+home-credit-default-risk/credit_card_balance.csv',
        nrows=num_rows)

    # Below code gives percentage of na in every column
    na_percentage = cc.isna().sum() / cc.shape[0] * 100
    # Below code gives list of columns having more than 50% missing
    col_to_drop = na_percentage[na_percentage > 50].keys()
    cc = cc.drop(col_to_drop, axis=1)
    # Replace missing values
    for col in cc.select_dtypes(['float', 'int64']):  # ---Applying only on numerical features
        cc[col].fillna(cc[col].median(), inplace=True)

    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


# In[16]:


def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
        return df


# In[17]:


#Fonction pour calculer les scores métier pour chaque algorithme:
def score_metier(y_test,ypred):
    cm=confusion_matrix(y_test,ypred)
    tn, fp, fn, tp = cm.ravel()
    #Score métier: 10* le taux de faux négatifs plus le taux de faux positifs, car taux de faux négatifs à pénalyser
    score=10*fn/(fn+tp) + fp/(fp+tn)
    return score


# In[18]:


if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()


# ### Jointure

# In[19]:


merge=main()


# In[20]:


merge.head()


# In[178]:


merge.info()


# In[22]:


plt.figure(figsize=(7,6))
plt.rc('ytick', labelsize=11)
plt.rc('xtick', labelsize=11)
sns.barplot(x=merge.isna().mean(), y=merge.columns)
plt.ylabel("Taux de valeurs manquantes",size=12)
plt.xlabel('Colonnes',size=12)
plt.title("Taux de valeurs manquantes par colonne",size=15)
plt.show()


# Beaucoup de valeurs manquantes

# In[23]:


merge.describe()


# In[24]:


plt.figure(figsize=(7,5))
merge.dtypes.value_counts().plot.pie(figsize=(12, 6),autopct="%.1f%%",explode=[0.05]*3).set_title('Types des données')


# In[25]:


merge.columns


# # Preprocessing

# In[21]:


merge = merge.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


# In[22]:


merge.shape


# In[324]:


# separate training data
target = merge['TARGET']
features = merge.drop(columns=['TARGET'], axis = 1)
print('x_train data shape: ', features.shape)
print('y_train data shape: ', target.shape)


# In[365]:


y = target
y.shape


# In[380]:


X= features
X.shape


# In[26]:


merge[merge['TARGET'].notnull()]


# In[27]:


from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier


# In[28]:


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot


# In[29]:


from sklearn.dummy import DummyClassifier


# In[30]:


import imblearn
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, RepeatedStratifiedKFold


# In[381]:


X.dropna(axis='columns',inplace=True)


# In[323]:


# Train test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)


# In[33]:


#Export de X_train pour une future utilisation de l'API
X.to_csv('data.csv', index=False)


# In[34]:


#Export de X_test pour le dashboard
X_test.to_csv('X_test.csv', index=False)


# In[35]:


# Export des 1000 premières lignes de X_test pour le dashboard
X_test.sort_index().head(1000).to_csv('X_test_sampled.csv', index=False)


# In[36]:


sampled=X.head(1000)
sampled.shape


# In[37]:


# Export de 1% des données pour une future démonstration de l'API
sampled.to_csv('sampled.csv', index=False)


# In[38]:


sampled['SK_ID_CURR'].to_csv('liste_sampled.csv', index=False)


# In[39]:


sampled.set_index("SK_ID_CURR", drop=False, inplace=True)


# In[40]:


#Export des IDs pour une future utilisation du dashboard
X['SK_ID_CURR'].to_csv('liste.csv', index=False)


# # Modèles
# ## DummyClassifier

# In[41]:


sampler = SMOTE(random_state=0)


# In[228]:


fbeta_scorer = make_scorer(fbeta_score, beta=2)


# In[229]:


dum_pipeline =  Pipeline([
    ('sampler', sampler),
    ('model', DummyClassifier())
])


# In[230]:


grid_dum = GridSearchCV(estimator=dum_pipeline,
                    param_grid={'model__strategy': ['stratified', 'most_frequent', 'uniform']},
                    cv=5,
                    scoring=fbeta_scorer,
                    refit='acc',
                    return_train_score=True)


# In[231]:


grid_dum.fit(X_train, y_train)


# In[232]:


grid_dum.scorer_


# In[233]:


grid_dum.best_score_


# In[234]:


bestimator_dum = grid_dum.best_estimator_


# In[235]:


bestimator_dum


# In[236]:


ypred = bestimator_dum.predict(X_test)


# In[237]:


print('Training accuracy {:.4f}'.format(bestimator_dum.score(X_train,y_train)))
print('Testing accuracy {:.4f}'.format(bestimator_dum.score(X_test,y_test)))


# In[238]:


print('Mean ROC AUC: %.3f' % roc_auc_score(y_test, ypred))


# In[239]:


auc_dum=roc_auc_score(y_test, ypred)


# ### Score métier

# In[240]:


score_dum=score_metier(y_test,ypred)


# In[282]:


score_dum


# ### Courbe ROC

# In[242]:


# predict probabilities
yhat = bestimator_dum.predict_proba(X_test)


# In[243]:


# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]


# In[244]:


# plot no skill roc curve
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(y_test, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
pyplot.title("Courbe ROC de DummyClassifier")
# show the plot
pyplot.show()


# ## RandomForest Classifier

# In[245]:


from sklearn.ensemble import RandomForestClassifier


# In[246]:


param_grid = { 
    'model__n_estimators': [200, 500],
    'model__max_features': ['sqrt', 'log2'],
    'model__max_depth' : [4,5,6,7,8],
    'model__criterion' :['gini', 'entropy']
}


# In[247]:


rfc_pipeline =  Pipeline([
    ('sampler', sampler),
    ('model', RandomForestClassifier())
])


# In[248]:


# enable autologging
mlflow.sklearn.autolog()


# In[ ]:


with mlflow.start_run() as run:
    CV_rfc = GridSearchCV(estimator=rfc_pipeline, 
                        param_grid=param_grid, 
                        cv= 5,
                        scoring=fbeta_scorer,
                        n_jobs=-1
                     )
    CV_rfc.fit(X_train, y_train)


# In[ ]:


CV_rfc.best_score_


# In[ ]:


bestimator_rf = CV_rfc.best_estimator_


# In[ ]:


ypred = bestimator_rf.predict(X_test)


# In[ ]:


print('Training accuracy {:.4f}'.format(bestimator_rf.score(X_train,y_train)))
print('Testing accuracy {:.4f}'.format(bestimator_rf.score(X_test,y_test)))


# In[ ]:


print('Mean ROC AUC: %.3f' % roc_auc_score(y_test, ypred))


# In[ ]:


auc_rf=roc_auc_score(y_test, ypred)


# ### Score métier

# In[ ]:


score_rf=score_metier(y_test,ypred)


# In[281]:


score_rf


# ### Courbe ROC

# In[289]:


# predict probabilities
yhat = bestimator_rf.predict_proba(X_test)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# plot no skill roc curve
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(y_test, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
pyplot.title("Courbe ROC de Random Forest Classifier")
# show the plot
pyplot.show()


# ## LGBMClassifier

# In[ ]:


#kf = KFold(n_splits=10, random_state=0, shuffle=True)
sampler = SMOTE(random_state=0)
rs_parameters = {
    'model__learning_rate': [0.005,0.01,0.001,0.05],
    'model__n_estimators': [20,40,60,80,100],
    'model__num_leaves': [6,8,12,16]
    }
smp_pipeline = Pipeline([
    ('sampler', sampler),
    ('model', LGBMClassifier())
])


# In[ ]:


print(mlflow.__version__)


# In[ ]:


mlflow.lightgbm.autolog()  # Enable auto logging.


# In[ ]:


with mlflow.start_run():
    grid_imba = GridSearchCV(smp_pipeline,
                         param_grid=rs_parameters,
                         cv=5,
                         scoring=fbeta_scorer,
                         return_train_score=True,
                         n_jobs=-1,
                         verbose=True
                        )
    grid_imba.fit(X_train, y_train)


# In[ ]:


grid_imba.best_score_


# In[ ]:


bestimator_lgbm = grid_imba.best_estimator_
ypred = bestimator_lgbm.predict(X_test)


# In[ ]:


print('Training accuracy {:.4f}'.format(bestimator_lgbm.score(X_train,y_train)))
print('Testing accuracy {:.4f}'.format(bestimator_lgbm.score(X_test,y_test)))


# In[ ]:


print('Mean ROC AUC: %.3f' % roc_auc_score(y_test, ypred)) #Résultat au-dessus de 0.82, donc problème d'overfitting


# In[ ]:


auc_lgbm=roc_auc_score(y_test, ypred)


# ### Score métier

# In[ ]:


score_lgbm=score_metier(y_test,ypred)


# In[283]:


score_lgbm


# ### Courbe ROC

# In[288]:


# predict probabilities
yhat = bestimator_lgbm.predict_proba(X_test)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# plot no skill roc curve
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(y_test, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
pyplot.title("Courbe ROC de LGBM Classifier")
# show the plot
pyplot.show()


# # XGBoost

# In[ ]:


import xgboost as xgb


# In[ ]:


sampler = SMOTE(random_state=0)
xgb_parameters = {
              'model__nthread':[4], #when use hyperthread, xgboost may become slower
              'model__objective':['binary:logistic'],
              'model__learning_rate': [0.05], #so called `eta` value
              'model__max_depth': [6],
              'model__min_child_weight': [11],
              'model__silent': [1],
              'model__subsample': [0.8],
              'model__colsample_bytree': [0.7],
              'model__n_estimators': [5], #number of trees, change it to 1000 for better results
              'model__missing':[-999],
              'model__seed': [1337]
                }
xgb_pipeline = Pipeline([
    ('sampler', sampler),
    ('model', xgb.XGBClassifier())
])


# In[ ]:


with mlflow.start_run():
    grid_xgb = GridSearchCV(xgb_pipeline,
                             param_grid=xgb_parameters,
                             cv=5,
                             scoring=fbeta_scorer,
                             return_train_score=True,
                             n_jobs=-1,
                             verbose=True
                            )
    grid_xgb.fit(X_train, y_train)


# In[ ]:


bestimator_xgb = grid_xgb.best_estimator_
ypred = bestimator_xgb.predict(X_test)


# In[ ]:


print('Training accuracy {:.4f}'.format(bestimator_xgb.score(X_train,y_train)))
print('Testing accuracy {:.4f}'.format(bestimator_xgb.score(X_test,y_test)))


# In[ ]:


print('Mean ROC AUC: %.3f' % roc_auc_score(y_test, ypred)) #Résultat au-dessus de 0.82, donc problème d'overfitting


# In[ ]:


auc_xgb=roc_auc_score(y_test, ypred)


# ### Score métier

# In[ ]:


score_xgb=score_metier(y_test,ypred)


# In[284]:


score_xgb


# ### Courbe ROC

# In[287]:


# predict probabilities
yhat = bestimator_xgb.predict_proba(X_test)
# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]
# plot no skill roc curve
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(y_test, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
pyplot.title("Courbe ROC de LGBM Classifier")
# show the plot
pyplot.show()


# In[54]:


bestimator_xgb['model']


# # Tableau des performances des différents modèles

# In[285]:


perf=pd.DataFrame({'AUC':[
                            auc_dum,
                            auc_rf,
                            auc_lgbm,
                            auc_xgb],
                    'Training Accuracy':[
                            bestimator_dum.score(X_train,y_train),
                            bestimator_rf.score(X_train,y_train),
                            bestimator_lgbm.score(X_train,y_train),
                            bestimator_xgb.score(X_train,y_train)],
                    'Test Accuracy':[
                            bestimator_dum.score(X_test,y_test),
                            bestimator_rf.score(X_test,y_test),
                            bestimator_lgbm.score(X_test,y_test),
                            bestimator_xgb.score(X_test,y_test)],
                    "Score métier":[
                          score_dum,
                          score_rf,
                          score_lgbm,
                          score_xgb
                      ]},
                       index=['Dummy Classifier','Random Forest Classifier','LGBM Classifier', 'XGBoost'])


# In[286]:


perf


# In[290]:


# Enregistrement du modèle le plus performant pour l'API
pickle.dump(bestimator_lgbm['model'], open('model.pkl','wb'))


# In[291]:


model = pickle.load(open('model.pkl','rb'))


# ### Feature importance

# Choix de XBoost pour le choix du modèle de classification => Meilleur AUC

# In[292]:


# Récupération de l'importance des variables
importance = bestimator_lgbm['model'].feature_importances_


# In[298]:


features = X.columns
indices = np.argsort(importance)[-7 : ]

#Graphique des feature importance
plt.figure(figsize=(8,5))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### SHAP values

# In[299]:


import shap
print("SHAP Version : {}".format(shap.__version__))


# In[300]:


shap.initjs()


# In[301]:


get_ipython().run_line_magic('time', "shap_values = shap.TreeExplainer(bestimator_lgbm['model']).shap_values(X_test)")


# In[303]:


shap.summary_plot(shap_values, X_test,plot_size=(14,6),max_display=7)


# ### SHAP values locales

# In[304]:


#TreeExplainer comme modèle du pipeline
explainer = shap.TreeExplainer(bestimator_lgbm['model'])


# In[305]:


#Shap values des données
shap_values = explainer(X_train)


# In[306]:


#Waterfall plot pour un client
shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],shap_values[0].values[:,0], feature_names=X_train.columns)


# # Export de données pour le dashboard et l'API

# In[73]:


#Export des noms de colonnes de X_train pour le dashboard
feature_names=X_train.columns.values


# In[314]:


feat_names=pd.read_csv("C:/Users/valev/Projet-7/feature_names.csv")


# In[329]:


feat_names.head(1).to_csv('colonnes.csv', index=False)


# In[77]:


X_train.to_csv('feature_names.csv', index=False)


# In[312]:


from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2


# In[377]:


X, y = load_digits(return_X_y=True)
X.shape


# In[378]:


KBest = SelectKBest(chi2, k=10).fit(X, y) 


# In[379]:


f = KBest.get_support(1)


# In[382]:


X_new = X[X.columns[f]] # final features`


# In[384]:


X_new.head()


# In[169]:


graphique=pd.concat([X['SK_ID_CURR'], X_new],axis=1)


# In[106]:


graphique=pd.read_csv('graphique.csv',delimiter= ',')


# In[55]:


model = pickle.load(open('model.pkl','rb'))


# In[90]:


X_sampled=X.sort_index()


# In[93]:


predictions_probas = model.predict_proba(X_sampled)[:,1]


# In[94]:


predictions_probas=pd.DataFrame(predictions_probas)


# In[110]:


graphique.sort_values(by=['SK_ID_CURR'],inplace=True)


# In[112]:


graphique.insert(11, "Score", predictions_probas)


# In[96]:


predictions_probas.reset_index(drop=True, inplace=True)


# In[97]:


graphique=pd.concat([X_sampled,predictions_probas],axis=1,ignore_index=True)


# In[82]:


predictions_model=predictions_model.rename(columns={0: 'ID', 1: 'Prediction'})


# In[140]:


#Ajout de catégories "Crédit accepté" ou "Crédit non accepté" pour un kdeplot:
graphique['categorie'] = ['Crédit accepté' if x > 0.5 else 'Crédit non accepté' for x in graphique['Score']]


# In[145]:


graphique.rename(columns={"categorie": "Catégories de client"},inplace=True)


# In[147]:


#Export de graphique pour le dashboard
graphique.to_csv('graphique.csv', index=False)


# In[84]:


predictions_model.to_csv('predictions_model.csv', index=False)


# # Data drift

# Evidently is an open-source Python library for data scientists and ML engineers. It helps evaluate, test, and monitor the performance of ML models from validation to production. It works with tabular and text data.
# 
# Data Drift: Run statistical tests to compare the input feature distributions, and visually explore the drift.
# 
# Test des modèles sur X_test:

# In[34]:


pip install evidently


# In[35]:


from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_preset import DataQualityTestPreset


# In[45]:


#To run the Data Stability test suite and display the reports in the notebook:
data_stability= TestSuite(tests=[
    DataStabilityTestPreset(),
])
data_stability.run(current_data=X_train, reference_data=X_test, column_mapping=None)
data_stability 


# In[46]:


#Save file as html
data_stability.save_html("evidently_test.html")


# In[47]:


from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# In[48]:


#To generate data drift reports
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(current_data=X_train, reference_data=X_test, column_mapping=None)
data_drift_report


# In[49]:


data_drift_report.save_html("evidently_report.html")

