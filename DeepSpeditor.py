#Marcin Wilk
#team name DeepSpeditor
#Project for FedCSIS 2022 Challenge

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
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import LSTM, SimpleRNN,GRU
import random

os.chdir("C:\\Users\\Lenovo\\Desktop\\DeepSpeditor\\")

#preparation of the datasets

#FUEL--------------------------------

fuel_prices = pd.read_csv("fuel_prices.csv",sep=";")

fuel_prices["date"] = fuel_prices["date"].apply(lambda x:datetime.strptime(x, '%Y-%m-%d')).dt.date

#MAIN--------------------------------

css_main_training = pd.read_csv("css_main_training.csv",sep=";")
css_main_test = pd.read_csv("css_main_test.csv",sep=";")
css_main_training.head()
css_main_training.describe()
css_main_training.info()

css_main_training.id_currency.value_counts() #smart encoder
css_main_training.direction.value_counts() #onehot
css_main_training.id_payer.value_counts() #normalized count encoder
css_main_training.temperature.value_counts() #regex and average, nulls and >30 set to 30
css_main_training.load_size_type.value_counts() #onehot
css_main_training.contract_type.value_counts() #onehot
css_main_training.id_service_type.value_counts() #fill NaN with 0
css_main_training.route_start_datetime.value_counts() #change to datetime
css_main_training.route_start_country.value_counts() #normalized count encoder
css_main_training.prim_train_line.value_counts() #fill NaN with 0
css_main_training.prim_ferry_line.value_counts() #fill NaN with 0
css_main_training.ferry_intervals.value_counts() #fill NaN with 0

css_main_training.id_service_type.fillna(0,inplace=True)
css_main_training.prim_train_line.fillna(0,inplace=True)
css_main_training.prim_ferry_line.fillna(0,inplace=True)
css_main_training.ferry_intervals.fillna(0,inplace=True)

css_main_test.id_service_type.fillna(0,inplace=True)
css_main_test.prim_train_line.fillna(0,inplace=True)
css_main_test.prim_ferry_line.fillna(0,inplace=True)
css_main_test.ferry_intervals.fillna(0,inplace=True)


direction_enc = OneHotEncoder(drop='if_binary',handle_unknown='ignore',sparse=False,dtype='int').fit(np.array(css_main_training.direction).reshape(-1, 1))
direction_enc_df = pd.DataFrame(direction_enc.transform(np.array(css_main_training.direction).reshape(-1, 1)),
                                columns=["direction_"+cat for cat in direction_enc.categories_[0]])
css_main_training = css_main_training.join(direction_enc_df)
css_main_training.drop('direction', axis=1, inplace=True)

direction_test = pd.DataFrame(direction_enc.transform(np.array(css_main_test.direction).reshape(-1, 1)),
                                columns=["direction_"+cat for cat in direction_enc.categories_[0]])
css_main_test = css_main_test.join(direction_test)
css_main_test.drop('direction', axis=1, inplace=True)

load_size_type_enc = OneHotEncoder(drop='if_binary',handle_unknown='ignore',sparse=False,dtype='int').fit(np.array(css_main_training.load_size_type).reshape(-1, 1))
css_main_training.load_size_type = load_size_type_enc.transform(np.array(css_main_training.load_size_type).reshape(-1, 1))

css_main_test.load_size_type = load_size_type_enc.transform(np.array(css_main_test.load_size_type).reshape(-1, 1))

contract_type_enc = OneHotEncoder(drop='if_binary',handle_unknown='ignore',sparse=False,dtype='int').fit(np.array(css_main_training.contract_type).reshape(-1, 1))
css_main_training.contract_type = contract_type_enc.transform(np.array(css_main_training.contract_type).reshape(-1, 1))

css_main_test.contract_type = contract_type_enc.transform(np.array(css_main_test.contract_type).reshape(-1, 1))


def read_temperature(x):
    if pd.isnull(x):
        return(30)
    list_of_temp = re.findall('[+-]?[^\S]?\d+',x)
    if not list_of_temp:
        return(30)
    list_of_floats = []
    for el in list_of_temp:
        if '-' in el:
            list_of_floats.append(-1*float(re.findall('\d+',el)[0]))
        else:
            list_of_floats.append(float(re.findall('\d+',el)[0]))
    result = np.mean(list_of_floats)
    if result>30: return(30)
    else: return(result)

css_main_training.temperature = css_main_training.temperature.apply(read_temperature)

css_main_test.temperature = css_main_test.temperature.apply(read_temperature)

def bucket_currency(x):
    if x=='PLN' or x=='EUR':
        return(x)
    else:
        return('other')

css_main_training.id_currency = css_main_training.id_currency.apply(bucket_currency)
id_currency_enc = OneHotEncoder(drop='if_binary',handle_unknown='ignore',sparse=False,dtype='int').fit(np.array(css_main_training.id_currency).reshape(-1, 1))
id_currency_enc_df = pd.DataFrame(id_currency_enc.transform(np.array(css_main_training.id_currency).reshape(-1, 1)),
                                columns=["id_currency_"+cat for cat in id_currency_enc.categories_[0]])
css_main_training = css_main_training.join(id_currency_enc_df)
css_main_training.drop('id_currency', axis=1, inplace=True)

css_main_test.id_currency = css_main_test.id_currency.apply(bucket_currency)
id_currency_test = pd.DataFrame(id_currency_enc.transform(np.array(css_main_test.id_currency).reshape(-1, 1)),
                                columns=["id_currency_"+cat for cat in id_currency_enc.categories_[0]])
css_main_test = css_main_test.join(id_currency_test)
css_main_test.drop('id_currency', axis=1, inplace=True)


css_main_training.route_start_datetime = css_main_training.route_start_datetime.apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
css_main_training.route_end_datetime = css_main_training.route_end_datetime.apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

css_main_test.route_start_datetime = css_main_test.route_start_datetime.apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
css_main_test.route_end_datetime = css_main_test.route_end_datetime.apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

country_enc = CountEncoder(normalize=True,cols=['id_payer','first_load_country','last_unload_country','route_start_country','route_end_country'],handle_unknown=0, handle_missing='return_nan').fit(css_main_training)
css_main_training = country_enc.transform(css_main_training)
css_main_test = country_enc.transform(css_main_test)

#css_main_training.info()

css_main_training["route_length_hours"] = (css_main_training.route_end_datetime-css_main_training.route_start_datetime)/np.timedelta64(1, 'h')
css_main_training["route_start_month"] = css_main_training.route_start_datetime.dt.month
css_main_training["route_end_month"] = css_main_training.route_end_datetime.dt.month
css_main_training["route_start_year"] = css_main_training.route_start_datetime.dt.year
css_main_training["route_end_year"] = css_main_training.route_end_datetime.dt.year

css_main_test["route_length_hours"] = (css_main_test.route_end_datetime-css_main_test.route_start_datetime)/np.timedelta64(1, 'h')
css_main_test["route_start_month"] = css_main_test.route_start_datetime.dt.month
css_main_test["route_end_month"] = css_main_test.route_end_datetime.dt.month
css_main_test["route_start_year"] = css_main_test.route_start_datetime.dt.year
css_main_test["route_end_year"] = css_main_test.route_end_datetime.dt.year

#22 and 23 to delete later (object type)

css_main_training.route_start_datetime = css_main_training.route_start_datetime.dt.date
css_main_test.route_start_datetime = css_main_test.route_start_datetime.dt.date
css_main_training_fuel = pd.merge(css_main_training,fuel_prices,how="left",left_on="route_start_datetime",right_on="date")
css_main_training_fuel.drop(["route_start_datetime","route_end_datetime","date"],axis=1,inplace=True)
css_main_training.drop(["route_start_datetime","route_end_datetime"],axis=1,inplace=True)
css_main_test_fuel = pd.merge(css_main_test,fuel_prices,how="left",left_on="route_start_datetime",right_on="date")
css_main_test_fuel.drop(["route_start_datetime","route_end_datetime","date"],axis=1,inplace=True)


#ROUTES-----------------------------------------------------------------------

css_routes_training = pd.read_csv("css_routes_training.csv",sep=";")
css_routes_test = pd.read_csv("css_routes_test.csv",sep=";")
css_routes_training.head()
css_routes_training.describe()
css_routes_training.info()

css_routes_training.id_contractor.value_counts() #fill NaN with 0
css_routes_training.step_type.value_counts() #normalized count encoder
css_routes_training.ferry.value_counts()
css_routes_training.ferry_line.value_counts() #fill NaN with 0
css_routes_training.train_line.value_counts() #fill NaN with 0
css_routes_training.city.value_counts() #absolutly of no value
css_routes_training.address.value_counts() #absolutly of no value
css_routes_training = css_routes_training.drop(["city","address"],axis=1) #drop it
css_routes_test = css_routes_test.drop(["city","address"],axis=1) #drop it
css_routes_training.country_code.value_counts() #normalized count encoder
css_routes_training.estimated_time.value_counts() #change to datetime
css_routes_training.vehicle_type.value_counts() #one hot encoder
css_routes_training.vehicle_capacity_type.value_counts() #one hot encoder
css_routes_training.trailer_generator.value_counts() #one hot encoder

css_routes_training.id_contractor.fillna(0,inplace=True)
css_routes_training.ferry_duration.fillna(0,inplace=True)
css_routes_training.ferry_line.fillna(0,inplace=True)
css_routes_training.train_line.fillna(0,inplace=True)

css_routes_test.id_contractor.fillna(0,inplace=True)
css_routes_test.ferry_duration.fillna(0,inplace=True)
css_routes_test.ferry_line.fillna(0,inplace=True)
css_routes_test.train_line.fillna(0,inplace=True)

count_enc = CountEncoder(normalize=True,cols=['step_type','country_code'],handle_unknown=0, handle_missing='return_nan').fit(css_routes_training)
css_routes_training = count_enc.transform(css_routes_training)
css_routes_test = count_enc.transform(css_routes_test)

css_routes_training.estimated_time = css_routes_training.estimated_time.apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
css_routes_test.estimated_time = css_routes_test.estimated_time.apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

vehicle_type_enc = OneHotEncoder(drop='if_binary',handle_unknown='ignore',sparse=False,dtype='int').fit(np.array(css_routes_training.vehicle_type).reshape(-1, 1))
css_routes_training.vehicle_type = vehicle_type_enc.transform(np.array(css_routes_training.vehicle_type).reshape(-1, 1))
css_routes_test.vehicle_type = vehicle_type_enc.transform(np.array(css_routes_test.vehicle_type).reshape(-1, 1))

vehicle_capacity_type_enc = OneHotEncoder(drop='if_binary',handle_unknown='ignore',sparse=False,dtype='int').fit(np.array(css_routes_training.vehicle_capacity_type).reshape(-1, 1))
css_routes_training.vehicle_capacity_type = vehicle_capacity_type_enc.transform(np.array(css_routes_training.vehicle_capacity_type).reshape(-1, 1))
css_routes_test.vehicle_capacity_type = vehicle_capacity_type_enc.transform(np.array(css_routes_test.vehicle_capacity_type).reshape(-1, 1))

trailer_generator_enc = OneHotEncoder(drop='if_binary',handle_unknown='ignore',sparse=False,dtype='int').fit(np.array(css_routes_training.trailer_generator).reshape(-1, 1))
css_routes_training.trailer_generator = trailer_generator_enc.transform(np.array(css_routes_training.trailer_generator).reshape(-1, 1))
css_routes_test.trailer_generator = trailer_generator_enc.transform(np.array(css_routes_test.trailer_generator).reshape(-1, 1))

#to concider
#step_duration = css_routes_training.groupby("id_contract").estimated_time.diff(periods=1).shift(periods=-1)/np.timedelta64(1, 'h')
#step_duration.fillna(0,inplace=True)
#css_routes_training["step_duration_hours"] = step_duration
#for now

css_routes_training["estimated_time_hour"] = css_routes_training.estimated_time.dt.hour
css_routes_training.drop("estimated_time",inplace=True,axis=1)
css_routes_test["estimated_time_hour"] = css_routes_test.estimated_time.dt.hour
css_routes_test.drop("estimated_time",inplace=True,axis=1)

#css_routes_training.info()

#2-25,57 - general info, no nulls
#26-42 - vehicle info lots of nulls
#43-56 - trailer info lots of nulls

css_routes_training["has_vehicle_info"] = 1*(~pd.isnull(css_routes_training.id_vehicle_model))
css_routes_training["has_trailer_info"] = 1*(~pd.isnull(css_routes_training.id_trailer_model))
css_routes_test["has_vehicle_info"] = 1*(~pd.isnull(css_routes_test.id_vehicle_model))
css_routes_test["has_trailer_info"] = 1*(~pd.isnull(css_routes_test.id_trailer_model))

#delete step number

css_routes_training.drop("step",axis=1,inplace=True)
css_routes_test.drop("step",axis=1,inplace=True)

#fill all nulls with zeros

css_routes_training.fillna(0,inplace=True)
css_routes_test.fillna(0,inplace=True)

#merging both dataframes

css_routes_training.id_contract.value_counts(sort=False)
css_main_training["number_of_steps"] = css_routes_training.id_contract.value_counts(sort=False).reset_index(drop=True)

#we are getting rid of JAAR contract - too long

css_main_training = css_main_training.loc[css_main_training.id_contract != "JAAR",:].reset_index(drop=True)
css_main_training_fuel = css_main_training_fuel.loc[css_main_training_fuel.id_contract != "JAAR",:].reset_index(drop=True)
css_routes_training = css_routes_training.loc[css_routes_training.id_contract != "JAAR",:].reset_index(drop=True)

#we add each step to main dataframe as new set of columns

css_merged = css_main_training.copy()

for step_num in range(0,20):
    css_routes_training_step = css_routes_training.groupby('id_contract', as_index=False, sort=False).nth(step_num).reset_index(drop=True)
    css_merged = pd.merge(css_merged,css_routes_training_step,how="left",on="id_contract",suffixes=('','_'+str(step_num)))


css_merged.shape
css_merged.info()
y = css_main_training.expenses
css_merged.drop(["id_contract","expenses"],axis=1,inplace=True)
css_merged.fillna(0,inplace=True)
X = css_merged.copy()

#predicting expenses without neural network

X_train, X_test, y_train, y_test = train_test_split(css_main_training_fuel.drop(["id_contract","expenses"],axis=1), y, test_size=0.33, random_state=0)

clf = LinearRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

clf = LinearRegression().fit(css_main_training.drop(["id_contract"],axis=1), y)
y_pred = clf.predict(css_main_training.drop(["id_contract"],axis=1))
np.sqrt(mean_squared_error(y,y_pred))

clf = KNeighborsRegressor(n_neighbors=15, weights='distance').fit(X_train, y_train)
y_pred = clf.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#prediction using only main and fuel

css_main_training_fuel.info()
css_main_test_fuel.info()


model2_general_info_input = tf.keras.layers.Input(shape=(None,45))
model2_general_info = Dense(30, activation='relu')(model2_general_info_input)
model2_general_info = Dense(15, activation='relu')(model2_general_info)
model2_general_info = Dense(1, activation='linear')(model2_general_info)

model2 = tf.keras.models.Model(model2_general_info_input, model2_general_info)

#optimizer = "rmsprop"

model2.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

model2.fit(X_train,y_train,epochs=150, verbose=2, validation_data = (X_test,y_test), batch_size=128)
y_pred = model2.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#prediction using only routes info

css_routes_training.info()
css_routes_test.info()


routes_list = []



def get_ragged_constants(data):
    return tf.RaggedTensor.from_row_lengths(
        values=data.iloc[:,1:].values,
        row_lengths=data.groupby('id_contract').size())




css_routes_tensor = get_ragged_constants(css_routes_training)
css_routes_tensor_test_contest = get_ragged_constants(css_routes_test)

    
css_routes_tensor.shape
y_tensor = tf.ragged.constant(y)


model3_route_info_input = tf.keras.layers.Input(shape=(None,58))
model3_route_info = LSTM(58)(model3_route_info_input)
model3_route_info = Dense(40,activation='relu')(model3_route_info)
model3_route_info = Dense(20,activation='relu')(model3_route_info)
model3_route_info = Dense(1,activation='linear')(model3_route_info)

model3 = tf.keras.models.Model(model3_route_info_input, model3_route_info)

#optimizer = "rmsprop"

model3.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])


model3.fit(css_routes_tensor,y_tensor,epochs=50, verbose=2, batch_size=128)



#prepare data for new models

X_tensor = tf.ragged.constant(css_main_training_fuel.drop(["id_contract","expenses"],axis=1))
X_tensor_test_contest = tf.ragged.constant(css_main_test_fuel.drop(["id_contract","expenses"],axis=1))

train_idx = random.sample(range(330054), 264000)
train_idx.sort()
train_idx = np.array(train_idx)
test_idx = np.setdiff1d(range(330054), train_idx, assume_unique=True)

X_tensor_train = tf.gather(X_tensor, train_idx, validate_indices=None, axis=0, batch_dims=0, name=None)
X_tensor_test = tf.gather(X_tensor, test_idx, validate_indices=None, axis=0, batch_dims=0, name=None)

css_routes_tensor_train = tf.gather(css_routes_tensor, train_idx, validate_indices=None, axis=0, batch_dims=0, name=None)
css_routes_tensor_test = tf.gather(css_routes_tensor, test_idx, validate_indices=None, axis=0, batch_dims=0, name=None)

y_tensor_train = tf.gather(y_tensor, train_idx, validate_indices=None, axis=0, batch_dims=0, name=None)
y_tensor_test = tf.gather(y_tensor, test_idx, validate_indices=None, axis=0, batch_dims=0, name=None)

#merging two models

model4_general_info_input = tf.keras.layers.Input(shape=(45,))
model4_general_info = Dense(30, activation='relu')(model4_general_info_input)
model4_general_info = Dense(15, activation='relu')(model4_general_info)

model4_route_info_input = tf.keras.layers.Input(shape=(None,58))
model4_route_info = LSTM(58)(model4_route_info_input)
model4_route_info = Dense(40,activation='relu')(model4_route_info)
model4_route_info = Dense(20,activation='relu')(model4_route_info)


model4_merged = tf.keras.layers.concatenate([model4_general_info,model4_route_info])
model4_merged = Dense(20,activation='relu')(model4_merged)
model4_merged = Dense(10,activation='relu')(model4_merged)
model4_merged = Dense(1,activation='linear')(model4_merged)

model4 = tf.keras.models.Model([model4_general_info_input,model4_route_info_input], model4_merged)

model4.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])



model4.fit([X_tensor_train,css_routes_tensor_train],y_tensor_train,epochs=150, verbose=2, batch_size=128,validation_data = ([X_tensor_test,css_routes_tensor_test],y_tensor_test))

y_pred = model4.predict([X_tensor_test,css_routes_tensor_test])
np.sqrt(mean_squared_error(y_tensor_test,y_pred))

y_pred_contest = model4.predict([X_tensor_test_contest,css_routes_tensor_test_contest])

a_file = open("solution1.txt", "w")
for row in y_pred_contest:
    np.savetxt(a_file, row)

a_file.close()

#model 5

model5_general_info_input = tf.keras.layers.Input(shape=(45,))
model5_general_info = Dense(30, activation='relu')(model5_general_info_input)
model5_general_info = Dense(15, activation='relu')(model5_general_info)

model5_route_info_input = tf.keras.layers.Input(shape=(None,58))
model5_route_info = GRU(58)(model5_route_info_input)
model5_route_info = Dense(40,activation='relu')(model5_route_info)
model5_route_info = Dense(20,activation='relu')(model5_route_info)


model5_merged = tf.keras.layers.concatenate([model5_general_info,model5_route_info])
model5_merged = Dense(20,activation='relu')(model5_merged)
model5_merged = Dense(10,activation='relu')(model5_merged)
model5_merged = Dense(1,activation='linear')(model5_merged)

model5 = tf.keras.models.Model([model5_general_info_input,model5_route_info_input], model5_merged)

model5.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

model5.fit([X_tensor_train,css_routes_tensor_train],y_tensor_train,epochs=150, verbose=2, batch_size=128,validation_data = ([X_tensor_test,css_routes_tensor_test],y_tensor_test))

y_pred = model5.predict([X_tensor_test,css_routes_tensor_test])
model5rmse = np.sqrt(mean_squared_error(y_tensor_test,y_pred))

y_pred_contest_model5 = model5.predict([X_tensor_test_contest,css_routes_tensor_test_contest])

a_file = open("solution3.txt", "w")
for row in y_pred_contest_model5:
    np.savetxt(a_file, row)

a_file.close()

#model 6

model6_general_info_input = tf.keras.layers.Input(shape=(45,))
model6_general_info = Dense(30, activation='relu')(model6_general_info_input)
model6_general_info = Dense(15, activation='relu')(model6_general_info)

model6_route_info_input = tf.keras.layers.Input(shape=(None,58))
model6_route_info = SimpleRNN(58)(model6_route_info_input)
model6_route_info = Dense(40,activation='relu')(model6_route_info)
model6_route_info = Dense(20,activation='relu')(model6_route_info)


model6_merged = tf.keras.layers.concatenate([model6_general_info,model6_route_info])
model6_merged = Dense(20,activation='relu')(model6_merged)
model6_merged = Dense(10,activation='relu')(model6_merged)
model6_merged = Dense(1,activation='linear')(model6_merged)

model6 = tf.keras.models.Model([model6_general_info_input,model6_route_info_input], model6_merged)

model6.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

model6.fit([X_tensor_train,css_routes_tensor_train],y_tensor_train,epochs=150, verbose=2, batch_size=128,validation_data = ([X_tensor_test,css_routes_tensor_test],y_tensor_test))

y_pred = model6.predict([X_tensor_test,css_routes_tensor_test])
model6rmse = np.sqrt(mean_squared_error(y_tensor_test,y_pred))

y_pred_contest_model6 = model6.predict([X_tensor_test_contest,css_routes_tensor_test_contest])

a_file = open("solution2.txt", "w")
for row in y_pred_contest_model6:
    np.savetxt(a_file, row)

a_file.close()

#model 7

model7_general_info_input = tf.keras.layers.Input(shape=(45,))

model7_route_info_input = tf.keras.layers.Input(shape=(None,58))
model7_route_info = SimpleRNN(58)(model7_route_info_input)

model7_merged = tf.keras.layers.concatenate([model7_general_info_input,model7_route_info])
model7_merged = Dense(80,activation='relu')(model7_merged)
model7_merged = Dense(40,activation='relu')(model7_merged)
model7_merged = Dense(20,activation='relu')(model7_merged)
model7_merged = Dense(1,activation='linear')(model7_merged)

model7 = tf.keras.models.Model([model7_general_info_input,model7_route_info_input], model7_merged)

model7.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

model7.fit([X_tensor_train,css_routes_tensor_train],y_tensor_train,epochs=150, verbose=2, batch_size=128,validation_data = ([X_tensor_test,css_routes_tensor_test],y_tensor_test))

y_pred = model7.predict([X_tensor_test,css_routes_tensor_test])
model7rmse = np.sqrt(mean_squared_error(y_tensor_test,y_pred))

y_pred_contest_model7 = model7.predict([X_tensor_test_contest,css_routes_tensor_test_contest])

a_file = open("solution4.txt", "w")
for row in y_pred_contest_model7:
    np.savetxt(a_file, row)

a_file.close()

# model 6 more epochs

model8 = tf.keras.models.Model([model6_general_info_input,model6_route_info_input], model6_merged)

model8.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

model8.fit([X_tensor_train,css_routes_tensor_train],y_tensor_train,epochs=300, verbose=2, batch_size=128,validation_data = ([X_tensor_test,css_routes_tensor_test],y_tensor_test))

y_pred = model8.predict([X_tensor_test,css_routes_tensor_test])
model8rmse = np.sqrt(mean_squared_error(y_tensor_test,y_pred))

y_pred_contest_model8 = model8.predict([X_tensor_test_contest,css_routes_tensor_test_contest])
plt.show()

a_file = open("solution5.txt", "w")
for row in y_pred_contest_model8:
    np.savetxt(a_file, row)

a_file.close()


#predicting expenses new approach


def compile_and_run(model, epochs=5):
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    #print(model.layers)
    model.summary()
    batch_size = 128
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return tf.keras.utils.plot_model(model)

css_merged.info(verbose=True)

# 0-41 general info
# 42 number of steps
# indexes of info
has_vehicle_info_idx = np.arange(99,1203,58)
has_trailer_info_idx = np.arange(100,1203,58)
# 43-66 + 58 route info
# 67-83 + 58 vehicle info
# 84-97 + 58 trailer info
# 98 +58 estimated time

# general info

model1_general_info_input = tf.keras.layers.Input(shape=(None,42))
model1_general_info = Dense(21, activation='relu')(model1_general_info_input)
ones = tf.fill(tf.shape(model1_general_info), 1.0)
model1_general_info = multiply([model1_general_info,ones])
#model1_general_info = tf.reshape(model1_general_info,shape=(21,))

# rest

model1_number_of_steps_input = tf.keras.layers.Input(shape=(None,1))
model1_number_of_steps = tf.cast(model1_number_of_steps_input, dtype='int32', name=None)
batch_size1 = tf.reshape(tf.constant(1),shape=(1,))
a = np.array([1,0],dtype="float32")
b = tf.tile(tf.constant(a),batch_size1)
c = np.array([1,-1], dtype="int32")
d = tf.constant(c)
e = np.array([0,20], dtype="int32")
f = tf.tile(tf.constant(e),batch_size1)
g = tf.reshape(multiply([tf.tile(d,batch_size1),tf.repeat(model1_number_of_steps,2)]), shape=(2,), name=None)
h = tf.math.add(f,g)
model1_number_of_steps = tf.repeat(b, h, axis=None, name=None)
model1_number_of_steps = tf.reshape(model1_number_of_steps,(1,20))



model1_has_vehicle_info_input = tf.keras.layers.Input(shape=(None,20))
ones20 = tf.fill(tf.shape(model1_has_vehicle_info_input), 1.0)
model1_has_vehicle_info = multiply([model1_has_vehicle_info_input,ones20])
#model1_has_vehicle_info_input = tf.reshape(model1_has_vehicle_info_input,shape=(20,))
model1_has_trailer_info_input = tf.keras.layers.Input(shape=(None,20))
model1_has_trailer_info = multiply([model1_has_trailer_info_input,ones20])
#model1_has_trailer_info_input = tf.reshape(model1_has_trailer_info_input,shape=(20,))

# route info always availible

route_info_list = []

# input

for i in range(20):
    route_info_list.append(tf.keras.layers.Input(shape=(None,25)))


# layer 1

model1_route_info_layer1 = Dense(5, activation='relu')

model1_route_info_layer1_list = [model1_route_info_layer1(x) for x in route_info_list]


# layer 2

model1_route_info_layer2 = Dense(1, activation='relu')

model1_route_info_layer2_list = [model1_route_info_layer2(x) for x in model1_route_info_layer1_list]


# concatenate

model1_route_info = tf.keras.layers.concatenate(model1_route_info_layer2_list)

#model1_route_info = tf.reshape(model1_route_info,shape=(None,20))
model1_route_info = multiply([model1_route_info,model1_number_of_steps])

# vehicle info

vehicle_info_list = []

# input

for i in range(20):
    vehicle_info_list.append(tf.keras.layers.Input(shape=(None,17)))

# layer 1

model1_vehicle_info_layer1 = Dense(5, activation='relu')

model1_vehicle_info_layer1_list = [model1_vehicle_info_layer1(x) for x in vehicle_info_list]

# layer 2

model1_vehicle_info_layer2 = Dense(1, activation='relu')

model1_vehicle_info_layer2_list = [model1_vehicle_info_layer2(x) for x in model1_vehicle_info_layer1_list]


# concatenate

model1_vehicle_info = tf.keras.layers.concatenate(model1_vehicle_info_layer2_list)

#model1_vehicle_info = tf.reshape(model1_vehicle_info,shape=(20,))
model1_vehicle_info = multiply([model1_vehicle_info,model1_has_vehicle_info_input])
model1_vehicle_info = multiply([model1_vehicle_info,model1_number_of_steps])

# trailer info

trailer_info_list = []

# input

for i in range(20):
    trailer_info_list.append(tf.keras.layers.Input(shape=(None,14)))


# layer 1

model1_trailer_info_layer1 = Dense(5, activation='relu')

model1_trailer_info_layer1_list = [model1_trailer_info_layer1(x) for x in trailer_info_list]


# layer 2

model1_trailer_info_layer2 = Dense(1, activation='relu')

model1_trailer_info_layer2_list = [model1_trailer_info_layer2(x) for x in model1_trailer_info_layer1_list]


# concatenating

model1_trailer_info = tf.keras.layers.concatenate(model1_trailer_info_layer2_list)

#model1_trailer_info = tf.reshape(model1_trailer_info,shape=(20,))
model1_trailer_info = multiply([model1_trailer_info,model1_has_trailer_info_input])
model1_trailer_info = multiply([model1_trailer_info,model1_number_of_steps])

# concatenating everything

model1_concatenated = tf.keras.layers.concatenate([model1_general_info,model1_route_info,model1_vehicle_info,model1_trailer_info])


model1_concatenated_layer1 = Dense(50, activation='relu')(model1_concatenated)
model1_concatenated_layer2 = Dense(25, activation='relu')(model1_concatenated_layer1)
model1_output = Dense(1, activation='linear')(model1_concatenated_layer2)

# creating model

route_vehicle_trailer_list = []
for i in range(20):
    route_vehicle_trailer_list.append(route_info_list[i])
    route_vehicle_trailer_list.append(vehicle_info_list[i])
    route_vehicle_trailer_list.append(trailer_info_list[i])

input_list = [model1_general_info_input,model1_number_of_steps_input,model1_has_vehicle_info_input,model1_has_trailer_info_input]+route_vehicle_trailer_list

model1 = tf.keras.models.Model(input_list, model1_output)

#optimizer = "rmsprop"

model1.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

# 0-41 general info
# 42 number of steps
# indexes of info
has_vehicle_info_idx = np.arange(99,1203,58)
has_trailer_info_idx = np.arange(100,1203,58)
# 43-66 + 58 route info 
np.arange(43,67,1)
# 67-83 + 58 vehicle info
# 84-97 + 58 trailer info
# 98 +58 estimated time

X_list = [X.iloc[:,0:42],X.iloc[:,42],X.iloc[:,has_vehicle_info_idx],X.iloc[:,has_trailer_info_idx]]



for i in range(20):
    X_list.append(X.iloc[:,np.append(np.arange(43+i*58,67+i*58,1),98+i*58)])
    X_list.append(X.iloc[:,np.arange(67+i*58,84+i*58,1)])
    X_list.append(X.iloc[:,np.arange(84+i*58,98+i*58,1)])

model1.fit(X_list,y,epochs=5, verbose=False, validation_split=0.2, batch_size=1)

model1.summary()










