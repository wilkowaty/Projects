import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt # a plotting library
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
import time
import scipy.sparse
from scipy import sparse
from scipy.sparse import hstack
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.glmm import GLMMEncoder
from sklearn.preprocessing import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import cross_validate
import nltk
from nltk.stem.porter import *
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import FunctionTransformer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

train = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Warsztaty projektowe\\Python\\train.tsv', sep='\t')
test = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Warsztaty projektowe\\Python\\test_noy.tsv', sep='\t')
y_train = 1*(train["label"] == "pants-fire")
nlp = spacy.load("en_core_web_lg")

def lemmatize(x):
    l = []
    for d in x:
        lem = nlp(d)
        l.append(" ".join(t.lemma_ for t in lem))
    return l
Lemmatizer = FunctionTransformer(lemmatize)

def pos_f(x):
    l = []
    for d in x:
        lem = nlp(d)
        l.append(" ".join(t.pos_ for t in lem))
    return l
PartOfSpeech = FunctionTransformer(pos_f)

def shaper(x):
    new_x = pd.DataFrame(columns=["avg_sentence_l","nos","now","avg_word_l"])
    i = 0
    for d in x:
        text = nlp(d)
        new_x.loc[i,"avg_sentence_l"] = float(np.mean(list(len(token) for token in text.sents)))
        new_x.loc[i,"nos"] = float(len(list(text.sents)))
        new_x.loc[i,"now"] = float(len([token.text for token in text]))
        new_x.loc[i,"avg_word_l"] = float(np.mean([len(token.text) for token in text]))
        i = i + 1
    return new_x.astype(float)
TextFeatures = FunctionTransformer(shaper)


DimRed = FeatureUnion([
            ("KPCA",KernelPCA(n_components=50, kernel='poly')),
            ("Isomap",Isomap(n_components=50)),
            ("TSVD",TruncatedSVD(n_components=50))
        ])

lem = Pipeline([("lemmatizer",Lemmatizer),("vectorizer", TfidfVectorizer()),("DimRed",DimRed)]) #
pos = Pipeline([("pos",PartOfSpeech),("vectorizer", TfidfVectorizer())])

clf1 = LogisticRegression(random_state=0)
clf2 = AdaBoostClassifier(n_estimators=100, random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=30,weights='distance')
clf4 = BernoulliNB()
clf5 = RandomForestClassifier(random_state=0,max_depth=4)
clf = VotingClassifier(estimators=[
        ('lr', clf1), ('ada', clf2), ('knn', clf3),('NB',clf4),('rf',clf5)], voting='soft')

transformer1 = ColumnTransformer([
    ('subject',Pipeline([('vect',CountVectorizer())]),'subject'),
    ('statement1',Pipeline([
        ('FU',FeatureUnion([
            ("lem",lem),
            ("pos",pos),
            ("text_f",TextFeatures)
        ]))
    ]),
     'statement'),
    ('speaker',TargetEncoder(),'speaker'),
    ('party',OneHotEncoder(handle_unknown='ignore',drop=['none']),['party']),
    ('state',TargetEncoder(),'state')
])

p_main1 = Pipeline([("transformer",transformer1),('scaler',MaxAbsScaler()),("clf", clf)])

cv_results = cross_validate(p_main1, train, y_train, cv=5,scoring="roc_auc",n_jobs=None)

np.mean(cv_results['test_score'])

#0.715 kPCA n=1000, CountVectorizer, log_reg
#0.718 kPCA n=1000, TfidVectorizer, log_reg
#0.704 - , TfidfVectorizer, naivebayes


p_main1.fit(train,y_train)
test_pred = p_main1.predict_proba(test)
np.save("C:\\Users\\Lenovo\\Desktop\\Warsztaty projektowe\\Python\\test_pred.npy",test_pred)
np.savetxt("C:\\Users\\Lenovo\\Desktop\\Warsztaty projektowe\\Python\\test_pred.txt",test_pred[:,1])

import seaborn as sns
sns.histplot(test_pred[:,1])
plt.show()

np.sum(1*(test_pred[:,1]>0.45))