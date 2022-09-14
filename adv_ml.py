#MARCIN WILK 
#Project for advanced machine learning course 
#KDD Cup 2012 - task and data is from this kaggle contest

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder
import re
from datetime import datetime
from category_encoders.count import CountEncoder
import pydot
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Embedding, Masking
from tensorflow.keras.layers import Conv2D, Dropout,Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import LSTM, SimpleRNN,GRU
import random
from sklearn import metrics
import sklearn
import imblearn
from imblearn.under_sampling import RandomUnderSampler

os.chdir("C:\\Users\\Lenovo\\Desktop\\Zaawansowany ML\\Projekt\\")

df = pd.read_csv('D80M.tsv',sep='\t')
df.head()
df.info()

y = df["Click"]
y.value_counts()

#balance the samples, we want to have the same number of failures and successes

idx_0 = np.where(y==0)[0]
idx_1 = np.where(y==1)[0]

idx_0 = random.sample(set(idx_0), len(idx_1))

idx = list(idx_1) + list(idx_0)
del idx_0
del idx_1
idx = np.array(idx)

X = df.iloc[idx,1:].reset_index(drop=True)
y = y[idx].reset_index(drop=True)

del df

X.info()
# Click - target variable
# 1 - 8 numerical variables
# 9 - 12 text variables

X_numerical = X.iloc[:,0:8].reset_index(drop=True)
y = y.reset_index(drop=True)

#split into train and validation samples

train_idx = random.sample(range(7492850), 5994279)
train_idx.sort()
train_idx = np.array(train_idx)
test_idx = np.setdiff1d(range(7492849), train_idx, assume_unique=True)

X_numerical_train = X_numerical.iloc[train_idx,:].reset_index(drop=True)
X_numerical_test = X_numerical.iloc[test_idx,:].reset_index(drop=True)

y_train = np.array(y[train_idx])
y_test = np.array(y[test_idx])

np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)

X_numerical["DisplayURL"].value_counts() #count encoder
X_numerical["AdId"].value_counts() #count encoder
X_numerical["UserID"].value_counts() #count encoder
X_numerical["AdvertiserId"].value_counts() #count encoder
X_numerical["Depth"].value_counts() #no encoding
X_numerical["Position"].value_counts() #no encoding
X_numerical["Age"].value_counts() #no encoding
X_numerical["Gender"].value_counts() #one hot encoder


count_enc = CountEncoder(cols=["DisplayURL","AdId","UserID","AdvertiserId"],handle_unknown=0,normalize=True)
one_hot = OneHotEncoder(sparse=False,handle_unknown='infrequent_if_exist',max_categories=1000).fit(np.array(X_numerical_train["Gender"]).reshape(-1, 1))

X_numerical_train = count_enc.fit_transform(X_numerical_train)
X_numerical_test = count_enc.transform(X_numerical_test)

X_numerical_train["Gender_0"] = one_hot.transform(np.array(X_numerical_train["Gender"]).reshape(-1, 1))[:,0]
X_numerical_train["Gender_1"] = one_hot.transform(np.array(X_numerical_train["Gender"]).reshape(-1, 1))[:,1]
X_numerical_train["Gender_2"] = one_hot.transform(np.array(X_numerical_train["Gender"]).reshape(-1, 1))[:,2]

X_numerical_test["Gender_0"] = one_hot.transform(np.array(X_numerical_test["Gender"]).reshape(-1, 1))[:,0]
X_numerical_test["Gender_1"] = one_hot.transform(np.array(X_numerical_test["Gender"]).reshape(-1, 1))[:,1]
X_numerical_test["Gender_2"] = one_hot.transform(np.array(X_numerical_test["Gender"]).reshape(-1, 1))[:,2]

X_numerical_train.drop("Gender",axis=1,inplace=True)
X_numerical_test.drop("Gender",axis=1,inplace=True)

# logistic regression on numerical variables

clf = LogisticRegression().fit(X_numerical_train, y_train)
y_pred_proba = clf.predict_proba(X_numerical_test)
metrics.roc_auc_score(y_test, y_pred_proba[:,1])
clf.coef_
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba[:,1])

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# model on numerical variables

input = tf.keras.layers.Input(shape=(10,))
x = Dense(5,'linear')(input)
x = Dense(2,'linear')(x)
output = Dense(1,'sigmoid')(input)

model1 = tf.keras.models.Model(input, output)

model1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.AUC()])

model1.summary()

model1.fit(tf.constant(X_numerical_train),tf.constant(y_train),epochs=50, verbose=2, validation_data=(tf.constant(X_numerical_test),tf.constant(y_test)),batch_size=1000,shuffle=True)

y_pred_proba = model1.predict(X_numerical_test)
metrics.roc_auc_score(y_test, y_pred_proba)
np.unique(y_pred_proba, return_counts=True)

for layerr in model1.layers:
    print(layerr.get_weights())

# model with text variables

def clean_desc(x):
    return list(map(int,x.split('|')))

X_text = X.iloc[:,8:].reset_index(drop=True)

X_text = X_text.applymap(clean_desc)
X_text.applymap(np.max).apply(np.max)

AdKeyword_tokens = tf.keras.preprocessing.sequence.pad_sequences(X_text["AdKeyword_tokens"], maxlen=10, padding='post', truncating="post")
AdTitle_tokens = tf.keras.preprocessing.sequence.pad_sequences(X_text["AdTitle_tokens"], maxlen=10, padding='post', truncating="post")
AdDescription_tokens = tf.keras.preprocessing.sequence.pad_sequences(X_text["AdDescription_tokens"], maxlen=10, padding='post', truncating="post")
Query_tokens = tf.keras.preprocessing.sequence.pad_sequences(X_text["Query_tokens"], maxlen=10, padding='post', truncating="post")

del X
del X_text

# model 2

input_AdKeyword = tf.keras.layers.Input(shape=(10,))
input_AdTitle = tf.keras.layers.Input(shape=(10,))
input_AdDescription = tf.keras.layers.Input(shape=(10,))
input_Query = tf.keras.layers.Input(shape=(10,))

expand_AdKeyword = tf.expand_dims(input_AdKeyword, axis=2, name=None)
expand_AdTitle = tf.expand_dims(input_AdTitle, axis=2, name=None)
expand_AdDescription = tf.expand_dims(input_AdDescription, axis=2, name=None)
expand_Query = tf.expand_dims(input_Query, axis=2, name=None)

#embed_AdKeyword = Embedding(1079182,10,input_length=16)(input_AdKeyword)
#embed_AdTitle = Embedding(1079182,10,input_length=23)(input_AdTitle)
#embed_AdDescription = Embedding(1079182,10,input_length=43)(input_AdDescription)
#embed_Query = Embedding(1079182,10,input_length=116)(input_Query)

mask_AdKeyword = Masking()(expand_AdKeyword)
mask_AdTitle = Masking()(expand_AdTitle)
mask_AdDescription = Masking()(expand_AdDescription)
mask_Query = Masking()(expand_Query)

lstm_AdKeyword = LSTM(1)(mask_AdKeyword)
lstm_AdTitle = LSTM(1)(mask_AdTitle)
lstm_AdDescription = LSTM(1)(mask_AdDescription)
lstm_Query = LSTM(1)(mask_Query)

merged = tf.keras.layers.concatenate([lstm_AdKeyword,lstm_AdTitle,lstm_AdDescription,lstm_Query])

layer = Dense(8,'relu')(merged)
layer = Dense(4,'relu')(layer)
layer = Dense(2,'relu')(layer)
output = Dense(1,'sigmoid')(layer)

model2 = tf.keras.models.Model([input_AdKeyword,input_AdTitle,input_AdDescription,input_Query], output)

model2.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model2.summary()

model2.fit([AdKeyword_tokens[train_idx,:],AdTitle_tokens[train_idx,:],AdDescription_tokens[train_idx,:],Query_tokens[train_idx,:]],
           y_train,epochs=5, verbose=2, 
           validation_data = ([AdKeyword_tokens[test_idx,:],AdTitle_tokens[test_idx,:],AdDescription_tokens[test_idx,:],Query_tokens[test_idx,:]],
                              y_test), batch_size=1000,shuffle=False)

y_pred = model2.predict([AdKeyword_tokens[test_idx,:],AdTitle_tokens[test_idx,:],AdDescription_tokens[test_idx,:],Query_tokens[test_idx,:]])
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# model 3

input_AdKeyword = tf.keras.layers.Input(shape=(10,1))
input_AdTitle = tf.keras.layers.Input(shape=(10,1))
input_AdDescription = tf.keras.layers.Input(shape=(10,1))
input_Query = tf.keras.layers.Input(shape=(10,1))

#expand_AdKeyword = tf.expand_dims(input_AdKeyword, axis=1, name=None)
#expand_AdTitle = tf.expand_dims(input_AdTitle, axis=1, name=None)
#expand_AdDescription = tf.expand_dims(input_AdDescription, axis=1, name=None)
#expand_Query = tf.expand_dims(input_Query, axis=1, name=None)

conv1d_AdKeyword = Conv1D(9,2)(input_AdKeyword)
conv1d_AdTitle = Conv1D(9,2)(input_AdTitle)
conv1d_AdDescription = Conv1D(9,2)(input_AdDescription)
conv1d_Query = Conv1D(9,2)(input_Query)

flat_AdKeyword = Flatten()(conv1d_AdKeyword)
flat_AdTitle = Flatten()(conv1d_AdTitle)
flat_AdDescription = Flatten()(conv1d_AdDescription)
flat_Query = Flatten()(conv1d_Query)

merged = tf.keras.layers.concatenate([flat_AdKeyword,flat_AdTitle,flat_AdDescription,flat_Query])

layer = Dense(81,'relu')(merged)
layer = Dense(27,'relu')(layer)
layer = Dense(9,'relu')(layer)
output = Dense(1,'sigmoid')(layer)

model3 = tf.keras.models.Model([input_AdKeyword,input_AdTitle,input_AdDescription,input_Query], output)

model3.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['AUC'])

model3.summary()

model3.fit([AdKeyword_tokens[train_idx,:],AdTitle_tokens[train_idx,:],AdDescription_tokens[train_idx,:],Query_tokens[train_idx,:]],
           y_train,epochs=5, verbose=2, 
           validation_data = ([AdKeyword_tokens[test_idx,:],AdTitle_tokens[test_idx,:],AdDescription_tokens[test_idx,:],Query_tokens[test_idx,:]],
                              y_test), batch_size=1000,shuffle=False)

y_pred_proba = model3.predict([AdKeyword_tokens[test_idx,:],AdTitle_tokens[test_idx,:],AdDescription_tokens[test_idx,:],Query_tokens[test_idx,:]])
metrics.roc_auc_score(y_test, y_pred_proba)
np.unique(y_pred_proba, return_counts=True)

# model numerical + text

input_AdKeyword = tf.keras.layers.Input(shape=(10))
input_AdTitle = tf.keras.layers.Input(shape=(10))
input_AdDescription = tf.keras.layers.Input(shape=(10))
input_Query = tf.keras.layers.Input(shape=(10))
input_numerical = tf.keras.layers.Input(shape=(10,))

emb = Embedding(1079182,7,input_length=10,mask_zero=True)
embed_AdKeyword = emb(input_AdKeyword)
embed_AdTitle = emb(input_AdTitle)
embed_AdDescription = emb(input_AdDescription)
embed_Query = emb(input_Query)

expand_AdKeyword = tf.expand_dims(embed_AdKeyword, axis=-1, name=None)
expand_AdTitle = tf.expand_dims(embed_AdTitle, axis=-1, name=None)
expand_AdDescription = tf.expand_dims(embed_AdDescription, axis=-1, name=None)
expand_Query = tf.expand_dims(embed_Query, axis=-1, name=None)

conv1d_AdKeyword = Conv1D(32,2,activation='relu')(embed_AdKeyword)
conv1d_AdTitle = Conv1D(32,2,activation='relu')(embed_AdTitle)
conv1d_AdDescription = Conv1D(32,2,activation='relu')(embed_AdDescription)
conv1d_Query = Conv1D(32,2,activation='relu')(embed_Query)

conv2d_AdKeyword = Conv2D(32,(2,7),activation='relu')(expand_AdKeyword)
conv2d_AdTitle = Conv2D(32,(2,7),activation='relu')(expand_AdTitle)
conv2d_AdDescription = Conv2D(32,(2,7),activation='relu')(expand_AdDescription)
conv2d_Query = Conv2D(32,(2,7),activation='relu')(expand_Query)

flat_AdKeyword = Flatten()(conv1d_AdKeyword)
flat_AdTitle = Flatten()(conv1d_AdTitle)
flat_AdDescription = Flatten()(conv1d_AdDescription)
flat_Query = Flatten()(conv1d_Query)

flat_AdKeyword_2 = Flatten()(conv2d_AdKeyword)
flat_AdTitle_2 = Flatten()(conv2d_AdTitle)
flat_AdDescription_2 = Flatten()(conv2d_AdDescription)
flat_Query_2 = Flatten()(conv2d_Query)

merged = tf.keras.layers.concatenate([flat_AdKeyword,flat_AdTitle,flat_AdDescription,flat_Query,
                                      input_numerical,
                                      flat_AdKeyword_2,flat_AdTitle_2,flat_AdDescription_2,flat_Query_2])

layer = Dense(230,activation='relu')(merged)
layer = Dense(23,activation='relu')(layer)
layer = Dense(10,activation='relu')(layer)
output = Dense(1,activation='sigmoid')(layer)

model4 = tf.keras.models.Model([input_AdKeyword,input_AdTitle,input_AdDescription,input_Query,input_numerical], output)

model4.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.AUC()])

model4.summary()

model4.fit([AdKeyword_tokens[train_idx,:],AdTitle_tokens[train_idx,:],AdDescription_tokens[train_idx,:],Query_tokens[train_idx,:],X_numerical_train],
           y_train,epochs=5, verbose=2, 
           validation_data = ([AdKeyword_tokens[test_idx,:],AdTitle_tokens[test_idx,:],AdDescription_tokens[test_idx,:],Query_tokens[test_idx,:],X_numerical_test],
                              y_test), batch_size=1000,shuffle=True)


y_pred_proba = model4.predict([AdKeyword_tokens[test_idx,:],AdTitle_tokens[test_idx,:],AdDescription_tokens[test_idx,:],Query_tokens[test_idx,:],X_numerical_test])
metrics.roc_auc_score(y_test, y_pred_proba)
np.unique(y_pred_proba, return_counts=True)
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#model 5

input_AdKeyword_m5 = tf.keras.layers.Input(shape=(10))
input_AdTitle_m5 = tf.keras.layers.Input(shape=(10))
input_AdDescription_m5 = tf.keras.layers.Input(shape=(10))
input_Query_m5 = tf.keras.layers.Input(shape=(10))
input_numerical_m5 = tf.keras.layers.Input(shape=(10,))

emb_m5 = Embedding(1079182,10,input_length=10,mask_zero=True)
embed_AdKeyword_m5 = emb_m5(input_AdKeyword_m5)
embed_AdTitle_m5 = emb_m5(input_AdTitle_m5)
embed_AdDescription_m5 = emb_m5(input_AdDescription_m5)
embed_Query_m5 = emb_m5(input_Query_m5)

lstm_AdKeyword_m5 = LSTM(10)(embed_AdKeyword_m5)
lstm_AdTitle_m5 = LSTM(10)(embed_AdTitle_m5)
lstm_AdDescription_m5 = LSTM(10)(embed_AdDescription_m5)
lstm_Query_m5 = LSTM(10)(embed_Query_m5)

merged_m5 = tf.keras.layers.concatenate([lstm_AdKeyword_m5,lstm_AdTitle_m5,lstm_AdDescription_m5,lstm_Query_m5,
                                      input_numerical_m5])
                                      #input_AdKeyword_m5,input_AdTitle_m5,input_AdDescription_m5,input_Query_m5])

layer_m5 = Dense(25,'relu')(merged_m5)
layer_m5 = Dense(5,'relu')(layer_m5)
output_m5 = Dense(1,'sigmoid')(layer_m5)

model5 = tf.keras.models.Model([input_AdKeyword_m5,input_AdTitle_m5,input_AdDescription_m5,input_Query_m5,input_numerical_m5], output_m5)

model5.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.AUC()])

model5.summary()

model5.fit([AdKeyword_tokens[train_idx,:],AdTitle_tokens[train_idx,:],AdDescription_tokens[train_idx,:],Query_tokens[train_idx,:],X_numerical_train],
           y_train,epochs=1, verbose=2, 
           validation_data = ([AdKeyword_tokens[test_idx,:],AdTitle_tokens[test_idx,:],AdDescription_tokens[test_idx,:],Query_tokens[test_idx,:],X_numerical_test],
                              y_test), batch_size=1000,shuffle=True)


y_pred_proba = model5.predict([AdKeyword_tokens[test_idx,:],AdTitle_tokens[test_idx,:],
                               AdDescription_tokens[test_idx,:],Query_tokens[test_idx,:],
                               X_numerical_test],batch_size=1000)
plt.hist(y_pred_proba, bins=100)
plt.show()
metrics.roc_auc_score(y_test, y_pred_proba)
np.unique(y_pred_proba, return_counts=True)
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# model 6 = 4 +5

input_AdKeyword_m6 = tf.keras.layers.Input(shape=(10))
input_AdTitle_m6 = tf.keras.layers.Input(shape=(10))
input_AdDescription_m6 = tf.keras.layers.Input(shape=(10))
input_Query_m6 = tf.keras.layers.Input(shape=(10))
input_numerical_m6 = tf.keras.layers.Input(shape=(10,))

emb_m6 = Embedding(1079182,10,input_length=10,mask_zero=True)
embed_AdKeyword_m6 = emb_m6(input_AdKeyword_m6)
embed_AdTitle_m6 = emb_m6(input_AdTitle_m6)
embed_AdDescription_m6 = emb_m6(input_AdDescription_m6)
embed_Query_m6 = emb_m6(input_Query_m6)

lstm_AdKeyword_m6 = LSTM(10)(embed_AdKeyword_m6)
lstm_AdTitle_m6 = LSTM(10)(embed_AdTitle_m6)
lstm_AdDescription_m6 = LSTM(10)(embed_AdDescription_m6)
lstm_Query_m6 = LSTM(10)(embed_Query_m6)

conv1d_AdKeyword_m6 = Conv1D(9,2)(embed_AdKeyword_m6)
conv1d_AdTitle_m6 = Conv1D(9,2)(embed_AdTitle_m6)
conv1d_AdDescription_m6 = Conv1D(9,2)(embed_AdDescription_m6)
conv1d_Query_m6 = Conv1D(9,2)(embed_Query_m6)

flat_AdKeyword_m6 = Flatten()(conv1d_AdKeyword_m6)
flat_AdTitle_m6 = Flatten()(conv1d_AdTitle_m6)
flat_AdDescription_m6 = Flatten()(conv1d_AdDescription_m6)
flat_Query_m6 = Flatten()(conv1d_Query_m6)

merged_m6 = tf.keras.layers.concatenate([lstm_AdKeyword_m6,lstm_AdTitle_m6,lstm_AdDescription_m6,lstm_Query_m6,
                                      input_numerical_m6,
                                      #input_AdKeyword_m6,input_AdTitle_m6,input_AdDescription_m6,input_Query_m6,
                                      flat_AdKeyword_m6,flat_AdTitle_m6,flat_AdDescription_m6,flat_Query_m6])

layer_m6 = Dense(180,'relu')(merged_m6)
layer_m6 = Dense(60,'relu')(layer_m6)
layer_m6 = Dense(30,'relu')(layer_m6)
output_m6 = Dense(1,'sigmoid')(layer_m6)


model6 = tf.keras.models.Model([input_AdKeyword_m6,input_AdTitle_m6,input_AdDescription_m6,input_Query_m6,input_numerical_m6], output_m6)

model6.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.AUC()])

model6.summary()

model6.fit([AdKeyword_tokens[train_idx,:],AdTitle_tokens[train_idx,:],AdDescription_tokens[train_idx,:],Query_tokens[train_idx,:],X_numerical_train],
           y_train,epochs=5, verbose=2, 
           validation_data = ([AdKeyword_tokens[test_idx,:],AdTitle_tokens[test_idx,:],AdDescription_tokens[test_idx,:],Query_tokens[test_idx,:],X_numerical_test],
                              y_test), batch_size=1000,shuffle=True)


# model 7

input_AdKeyword_m7 = tf.keras.layers.Input(shape=(10))
input_AdTitle_m7 = tf.keras.layers.Input(shape=(10))
input_AdDescription_m7 = tf.keras.layers.Input(shape=(10))
input_Query_m7 = tf.keras.layers.Input(shape=(10))
input_numerical_m7 = tf.keras.layers.Input(shape=(10,))

emb_m7 = Embedding(1079182,5,input_length=10,mask_zero=True)
embed_AdKeyword_m7 = emb_m7(input_AdKeyword_m7)
embed_AdTitle_m7 = emb_m7(input_AdTitle_m7)
embed_AdDescription_m7 = emb_m7(input_AdDescription_m7)
embed_Query_m7 = emb_m7(input_Query_m7)

lstm_AdKeyword_m7 = LSTM(5)(embed_AdKeyword_m7)
lstm_AdTitle_m7 = LSTM(5)(embed_AdTitle_m7)
lstm_AdDescription_m7 = LSTM(5)(embed_AdDescription_m7)
lstm_Query_m7 = LSTM(5)(embed_Query_m7)

merged_m7 = tf.keras.layers.concatenate([lstm_AdKeyword_m7,lstm_AdTitle_m7,lstm_AdDescription_m7,lstm_Query_m7,
                                      input_numerical_m7])
                                      #input_AdKeyword_m5,input_AdTitle_m5,input_AdDescription_m5,input_Query_m5])

layer_m7 = Dense(15,'relu')(merged_m7)
output_m7 = Dense(1,'sigmoid')(layer_m7)

model7 = tf.keras.models.Model([input_AdKeyword_m7,input_AdTitle_m7,input_AdDescription_m7,input_Query_m7,input_numerical_m7], output_m7)

model7.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.AUC()])

model7.summary()

model7.fit([AdKeyword_tokens[train_idx,:],AdTitle_tokens[train_idx,:],AdDescription_tokens[train_idx,:],Query_tokens[train_idx,:],X_numerical_train],
           y_train,epochs=5, verbose=2, 
           validation_data = ([AdKeyword_tokens[test_idx,:],AdTitle_tokens[test_idx,:],AdDescription_tokens[test_idx,:],Query_tokens[test_idx,:],X_numerical_test],
                              y_test), batch_size=1000,shuffle=True)


y_pred_proba = model7.predict([AdKeyword_tokens[test_idx,:],AdTitle_tokens[test_idx,:],
                               AdDescription_tokens[test_idx,:],Query_tokens[test_idx,:],
                               X_numerical_test],batch_size=1000)

# predict on test set for best performing models after sufficient training

df_test = pd.read_csv('D5M_test_x.tsv',sep='\t')
df_test.info()

X_numerical_test_5M = df_test.iloc[:,1:9]
X_numerical_test_5M = count_enc.transform(X_numerical_test_5M)
X_text_test_5M = df_test.iloc[:,9:]
del df_test

X_numerical_test_5M["Gender_0"] = one_hot.transform(np.array(X_numerical_test_5M["Gender"]).reshape(-1, 1))[:,0]
X_numerical_test_5M["Gender_1"] = one_hot.transform(np.array(X_numerical_test_5M["Gender"]).reshape(-1, 1))[:,1]
X_numerical_test_5M["Gender_2"] = one_hot.transform(np.array(X_numerical_test_5M["Gender"]).reshape(-1, 1))[:,2]

X_numerical_test_5M.drop("Gender",axis=1,inplace=True)

X_text_test_5M = X_text_test_5M.applymap(clean_desc)

AdKeyword_tokens_test_5M = tf.keras.preprocessing.sequence.pad_sequences(X_text_test_5M["AdKeyword_tokens"], maxlen=10, padding='post', truncating="post")
AdTitle_tokens_test_5M = tf.keras.preprocessing.sequence.pad_sequences(X_text_test_5M["AdTitle_tokens"], maxlen=10, padding='post', truncating="post")
AdDescription_tokens_test_5M = tf.keras.preprocessing.sequence.pad_sequences(X_text_test_5M["AdDescription_tokens"], maxlen=10, padding='post', truncating="post")
Query_tokens_test_5M = tf.keras.preprocessing.sequence.pad_sequences(X_text_test_5M["Query_tokens"], maxlen=10, padding='post', truncating="post")

del X_text_test_5M

y_pred_test_5M = model5.predict([AdKeyword_tokens_test_5M,AdTitle_tokens_test_5M,AdDescription_tokens_test_5M,Query_tokens_test_5M,X_numerical_test_5M])
y_pred_test_m4 = model4.predict([AdKeyword_tokens_test_5M,AdTitle_tokens_test_5M,
                                 AdDescription_tokens_test_5M,Query_tokens_test_5M,X_numerical_test_5M],batch_size=1000)


np.savetxt("C:\\Users\\Lenovo\\Desktop\\Zaawansowany ML\\Projekt\\y_pred_test_5M.txt",y_pred_test_5M)
np.savetxt("C:\\Users\\Lenovo\\Desktop\\Zaawansowany ML\\Projekt\\y_pred_test_m4.txt",y_pred_test_m4)

ones = np.where(y_pred_test_m4>0.5)
y_pred_test_m4.shape
np.histogram(y_pred_test_5M,bins=100)
plt.hist(y_pred_test_m4, bins=100)
plt.show()

