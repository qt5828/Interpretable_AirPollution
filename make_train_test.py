import torch
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os 
import random
import warnings

import torch
import datetime

from loss_utils import *
from models import *

import pickle

# 전역 변수 선언
SEQ_LENGTH = 7
DROP_ONEHOT = True
SELECTED_COLUMN = "median_no2" # "median_pm10", "median_no2"
all_train = []
all_test = []
cities_list = []

# featrue 개수 (실제 학습시에는 타겟까지 +1)
if DROP_ONEHOT:
    # INPUT_DIM = 8
    INPUT_DIM = 8 + 6
    # INPUT_DIM = 8 + 339 + 13
# one-hot 포함 개수 (요일정보, 달 정보, city/country정보)
else:
    INPUT_DIM = 8 + 19 + 339 + 13
HIDDEN_DIM = 32
LAYER_DIM = 3
normalization_type = 'mean_std' # 'max', mean_std
col_max = {}
col_mean = {}
col_mean2 = {}
col_std = {}

device = torch.device("cuda")







def get_train_test_data(df):
  # we'll mostly need median and variance values of features for most of our needs

  for col in df.columns:
    for x in ["min", "max", "count", "past_week", "variance"]:
      if x in col:
        df.drop([col], axis=1, inplace=True)

  # df["Population Staying at Home"] = df["Population Staying at Home"].apply(lambda x: x.replace(",", ""))
  # df["Population Not Staying at Home"] = df["Population Not Staying at Home"].apply(lambda x: x.replace(",", ""))

  # Now we want 2 more features. Which day of week it is and which month it is.
  # Both of these will be one-hot and hence we'll add 7+12 = 19 more columns.
  # Getting month id is easy from the datetime column. 
  # For day of week, we'll use datetime library.
  
  df['weekday'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday())
  df['month'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").month - 1)

  # using one-hot on Country and City
  Country_onehot = pd.get_dummies(df['Country'])
  # City_onehot = pd.get_dummies(df['City'])
  df.drop('Country', axis=1, inplace=True)
  # df.drop('City', axis=1, inplace=True)
  df = df.join([Country_onehot])
  # df = df.join([Country_onehot, City_onehot])
  
  # using one-hot on month and weekday
  weekday_onehot = pd.get_dummies(df['weekday'])
  weekday_onehot.columns = ["day_"+str(x) for x in weekday_onehot]
  month_onehot = pd.get_dummies(df['month'])
  month_onehot.columns = ["month_"+str(x) for x in month_onehot]
  
  df.drop(['weekday', 'month'], axis=1, inplace=True)
  df = df.join([weekday_onehot, month_onehot])

 

  cities_list = list(set(df['City']))
  cities_list.sort()
  print(cities_list)
  print("total # of cities : ", len(cities_list))
  city_df = {}
  test_indices_of_cities = {}
  train_set = {}
  test_set = {}
  TEST_SET_SIZE = 60                                        

  for city in cities_list:
    city_df[city] = df[df['City'] == city].sort_values('Date').reset_index()
    for col in city_df[city].columns:
      if col in ["median_pm25", "median_o3", "median_so2", "median_no2", "median_pm10", "median_co"]:
        continue
      try:  
        _mean = np.nanmean(city_df[city][col])
        if np.isnan(_mean) == True:
          _mean = 0
        city_df[city][col] = city_df[city][col].fillna(_mean)
      except:
        pass
    if city_df[city].shape[0] < 600 :
      print("City with less than 600 data : {} {}".format(city_df[city].shape[0], city))
      del city_df[city]
      continue
    
    test_index_start = random.randint(0, city_df[city].shape[0] - TEST_SET_SIZE)
    test_indices_of_cities[city] = [test_index_start, test_index_start + TEST_SET_SIZE]

    test_set[city] = city_df[city].iloc[test_index_start:test_index_start + TEST_SET_SIZE]
    train_set[city] = city_df[city].drop(index=list(range(test_index_start, test_index_start + TEST_SET_SIZE)))

  return train_set, test_set

class CityDataForecast(torch.utils.data.Dataset):
  def __init__(self, selected_column, split, train_set, test_set, all_train, all_test):
    self.split = split
    if split == "train":
      self.dataset = train_set
    else:
      self.dataset = test_set

    self.valid_city_idx = 0
    self.valid_day_idx = 1
    self.selected_column = selected_column

  def __getitem__(self, idx):
    # TEST
    if self.split != "train":
      # getting all data out of the validation set
      out, city = self.get_idx_data(idx)
      # print(out.columns)
      
    # TRAIN
    else:
      # getting data randomly for train split
      # Train for only one city
      city = random.choice(cities_list)
      _df = self.dataset[city]
      # arbitrary sequence
      start_idx = random.randint(1,_df.shape[0]-SEQ_LENGTH)
      # -1~6까지(총 seq_length+1만큼)
      out =  _df.iloc[start_idx-1:start_idx+SEQ_LENGTH]

    out = out.drop(['index', 'Date', 'City'], axis=1)

    Y = pd.DataFrame({})
    Y_true = pd.DataFrame({})

    for col in out.columns:
      # target을 Y값으로.
      if col == self.selected_column:
        Y_true[col] = out[col]
        Y[col] = out[col].fillna(col_mean[city][col])
      
      # target을 input features에서 제외.
      if col in ["median_pm25", "median_o3", "median_so2", "median_no2", "median_pm10", "median_co"]:
        out.drop([col], axis=1, inplace=True)
      else:
        out[col] = out[col].astype("float")
      
    out = np.concatenate((np.array(out)[1:,:], np.array(Y)[:-1,:]), axis=1)
    Y = np.array(Y)[1:]
    Y_true = np.array(Y_true)[1:]
    
    return out, Y, Y_true

  def get_idx_data(self, idx):
    city_idx = self.valid_city_idx % len(cities_list)
    city = cities_list[city_idx]
    print(city_idx)
    # city = cities_list[self.valid_city_idx]
    _df = self.dataset[city]

    # -1~seq-1(seq+1길이만큼!)
    out =  _df.iloc[self.valid_day_idx-1:self.valid_day_idx+SEQ_LENGTH]
    # print(out)
    # raise Exception("FINISH")
    
    if self.valid_day_idx+SEQ_LENGTH >= _df.shape[0]-1:
      self.valid_day_idx = 1
      self.valid_city_idx += 1
    else:
      self.valid_day_idx += 1

    return out, city

  def __len__(self):
    if self.split != "train":
      return (60-SEQ_LENGTH)*len(cities_list)
    return len(all_train) - (SEQ_LENGTH - 1)*len(cities_list)


class CityDataEstimatePollutant(torch.utils.data.Dataset):
  def __init__(self, selected_column, split, train_set, test_set, all_train, all_test):
    self.split = split
    if split == "train":
      self.dataset = train_set
    else:
      self.dataset = test_set

    self.valid_day_idx = 0
    self.selected_column = selected_column
    
    self.valid_city_idx = 0

  def __getitem__(self, idx):
    # TEST
    if self.split != "train":
      out, city = self.get_idx_data(idx)
      
    # TRAIN
    else:
      city = random.choice(cities_list)
      _df = self.dataset[city]
      start_idx = random.randint(0,_df.shape[0]-1)
      out = _df.iloc[start_idx:start_idx+1]
      # print(start_idx)

    out = out.drop(['index', 'Date', 'City'], axis=1)

    Y = pd.DataFrame({})
    Y_true = pd.DataFrame({})

    for col in out.columns:
      if col == self.selected_column:
        Y_true[col] = out[col]
        #print(out[col])
        Y[col] = out[col].fillna(col_mean[city][col])
      
      if col in ["median_pm25", "median_o3", "median_so2", "median_no2", "median_pm10", "median_co"]:
        out.drop([col], axis=1, inplace=True)
      else:
        out[col] = out[col].astype("float")

    out = np.array(out)
    Y = np.array(Y)
    Y_true = np.array(Y_true)
    
    # print(Y_true)
    return out, Y, Y_true

  def get_idx_data(self, idx):
    city = cities_list[self.valid_city_idx]
    _df = self.dataset[city]

    out =  _df.iloc[self.valid_day_idx:self.valid_day_idx+1]
    
    # _df.shape[0]-1
    if self.valid_day_idx >= _df.shape[0]-1:
      self.valid_day_idx = 0
      self.valid_city_idx += 1
    else:
      self.valid_day_idx += 1

    # print(self.valid_day_idx)
    return out, city

  def __len__(self):
    if self.split != "train":
      return 60*len(cities_list)
    return len(all_train)


import lime
from lime.lime_tabular import LimeTabularExplainer

# from AirPollution_integ_ours import *


# 데이터 로드
dfs = []
# dfs.append(pd.read_csv("./data/US2019to2023.csv"))
# dfs.append(pd.read_csv("./data/CA2019to2023.csv"))
dfs.append(pd.read_csv("./data/CN2019to2023.csv"))
# dfs.append(pd.read_csv("./data/DE2019to2023.csv"))
# dfs.append(pd.read_csv("./data/ES2019to2023.csv"))
# dfs.append(pd.read_csv("./data/FR2019to2023.csv"))
# dfs.append(pd.read_csv("./data/GB2019to2023.csv"))
dfs.append(pd.read_csv("./data/IN2019to2023.csv"))
dfs.append(pd.read_csv("./data/IR2019to2023.csv"))
dfs.append(pd.read_csv("./data/JP2019to2023.csv"))
dfs.append(pd.read_csv("./data/KR2019to2023.csv"))
# dfs.append(pd.read_csv("./data/RO2019to2023.csv"))
dfs.append(pd.read_csv("./data/TR2019to2023.csv"))

df = pd.concat(dfs, ignore_index=True)
df.to_csv("./data/asia2019to2023.csv")
# df = pd.read_csv("./data/integ2019to2023.csv")
# df.drop('Unnamed: 0', axis=1, inplace=True)

# Data Preprocessing
train_set, test_set = get_train_test_data(df)
print("Finish Datza Preprocessing")

cities_list = list(train_set.keys())


# print("train data")
all_train = pd.DataFrame()
for city in cities_list:
    all_train = all_train.append(train_set[city], ignore_index=True)

# print("test data")
all_test = pd.DataFrame({})
for city in test_set:
    all_test = all_test.append(test_set[city], ignore_index=True)

# print("concat data")
concat_df = pd.concat([all_train,all_test],axis=0)

# print("city")
# i=1
for city in cities_list:
    col_mean[city] = {}
    for col in train_set[city]:
        if col in ["index", "Date", "City", "Country"]:
            continue

        train_set[city][col] = train_set[city][col].astype("float")
        test_set[city][col] = test_set[city][col].astype("float")

        if col in ["median_pm25", "median_o3", "median_so2", "median_no2", "median_pm10", "median_co"]:
            _mean = np.nanmean(train_set[city][col])
            if np.isnan(_mean) == True:
                _mean = 0
        
            col_mean[city][col] = _mean
            train_set[city][col] = train_set[city][col].fillna(_mean)

        if normalization_type == 'mean_std':
            col_mean2[col] = np.nanmean(concat_df[col].astype("float"))
            col_std[col] = np.nanstd(concat_df[col].astype("float"))
            train_set[city][col] = (train_set[city][col] - col_mean2[col]) / (col_std[col] + 0.001)
            test_set[city][col] = (test_set[city][col] - col_mean2[col]) / (col_std[col] + 0.001)

        else:
            col_max[col] = concat_df[col].astype("float").max()
            train_set[city][col] = train_set[city][col] / (col_max[col] + 0.001)
            test_set[city][col] = test_set[city][col] / (col_max[col] + 0.001)

    # print("drop onehot or not", i)
    # i += 1
    if DROP_ONEHOT:
        # print(len(train_set[city].columns.to_list()))
        train_set[city].drop(train_set[city].columns[-(19):], axis=1, inplace=True)
        test_set[city].drop(test_set[city].columns[-(19):], axis=1, inplace=True)
        # train_set[city].drop(train_set[city].columns[-(19 + 339 + 13):], axis=1, inplace=True)
        # test_set[city].drop(test_set[city].columns[-(19 + 339 + 13):], axis=1, inplace=True)
        # print(len(train_set[city].columns.to_list()))

# number of data per city
print("num of cities : ", len(cities_list))
for city in cities_list:
    print(city+"({})".format(len(train_set[city])), end=", ")
    
print(len(train_set[city].columns.to_list()))
print(train_set[city].columns.to_list())
print(train_set[city].columns)
    
# Fix Seed for DataLoader
g = torch.Generator()
g.manual_seed(0)


train_data = CityDataEstimatePollutant(SELECTED_COLUMN, "train", train_set, test_set, all_train, all_test)
val_data = CityDataEstimatePollutant(SELECTED_COLUMN, "test", train_set, test_set, all_train, all_test)

x_train = []
y_train = []
for idx in range(len(train_data)):
    data, target, _ = train_data[idx]
    x_train.append(data)
    y_train.append(target)
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []
y_test = []
for idx in range(len(val_data)):
    data, target, _ = val_data[idx]
    x_test.append(data)
    y_test.append(target)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train)
print(y_train)

import pickle
with open('x_train_'+SELECTED_COLUMN+'.pkl', 'wb') as f:
    pickle.dump(x_train, f)
    
with open('y_train_'+SELECTED_COLUMN+'.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('x_test_'+SELECTED_COLUMN+'.pkl', 'wb') as f:
    pickle.dump(x_test, f)

with open('y_test_'+SELECTED_COLUMN+'.pkl', 'wb') as f:
    pickle.dump(y_test, f)
    