import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.autograd import Variable
from datetime import datetime
from tqdm import tqdm
import sklearn
from copy import deepcopy
from loss_utils import *
from models import *
import torch.nn as nn
import math
import os


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
seed_everything(100)
device = 'cuda'

import warnings
warnings.filterwarnings('ignore')


# Data Pre-processing

df = pd.read_csv("./data/city_pollution_data.csv")

DROP_ONEHOT = True
SEQ_LENGTH = 7

if DROP_ONEHOT:
  INPUT_DIM = 10 
else:
  INPUT_DIM = 29

HIDDEN_DIM = 32
LAYER_DIM = 3

normalization_type = 'mean_std' # 'max', mean_std

def get_train_test_data(df):
  # we'll mostly need median and variance values of features for most of our needs

  for col in df.columns:
    for x in ["min", "max", "count", "County", "past_week", "latitude", "longitude", "State", "variance"]:
      if x in col:
        df.drop([col], axis=1, inplace=True)

  df["Population Staying at Home"] = df["Population Staying at Home"].apply(lambda x: x.replace(",", ""))
  df["Population Not Staying at Home"] = df["Population Not Staying at Home"].apply(lambda x: x.replace(",", ""))

  # Now we want 2 more features. Which day of week it is and which month it is.
  # Both of these will be one-hot and hence we'll add 7+12 = 19 more columns.
  # Getting month id is easy from the datetime column. 
  # For day of week, we'll use datetime library.
  
  df['weekday'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday())
  df['month'] = df['Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").month - 1)

  # using one-hot on month and weekday
  weekday_onehot = pd.get_dummies(df['weekday'])
  weekday_onehot.columns = ["day_"+str(x) for x in weekday_onehot]
  month_onehot = pd.get_dummies(df['month'])
  month_onehot.columns = ["month_"+str(x) for x in month_onehot]

  df.drop(['weekday', 'month'], axis=1, inplace=True)
  df = df.join([weekday_onehot, month_onehot])

  cities_list = list(set(df['City']))
  city_df = {}
  test_indices_of_cities = {}
  train_set = {}
  test_set = {}
  TEST_SET_SIZE = 60

  for city in cities_list:
    city_df[city] = df[df['City'] == city].sort_values('Date').reset_index()
    for col in city_df[city].columns:
      if col in ["pm25_median", "o3_median", "so2_median", "no2_median", "pm10_median", "co_median"]:
        continue
      try:  
        _mean = np.nanmean(city_df[city][col])
        if np.isnan(_mean) == True:
          _mean = 0
        city_df[city][col] = city_df[city][col].fillna(_mean)
      except:
        pass
    
    random.seed(0)
    test_index_start = random.randint(0, city_df[city].shape[0] - TEST_SET_SIZE)
    test_indices_of_cities[city] = [test_index_start, test_index_start + TEST_SET_SIZE]

    test_set[city] = city_df[city].iloc[test_index_start:test_index_start + TEST_SET_SIZE]
    train_set[city] = city_df[city].drop(index=list(range(test_index_start, test_index_start + TEST_SET_SIZE)))

  return train_set, test_set

train_set, test_set = get_train_test_data(df)

cities_list = list(train_set.keys())

all_train = pd.DataFrame()
for city in cities_list:
  all_train = all_train.append(train_set[city], ignore_index=True)

all_test = pd.DataFrame({})
for city in test_set:
  all_test = all_test.append(test_set[city], ignore_index=True)

concat_df = pd.concat([all_train,all_test],axis=0)

# ---------------------------------------------------------------------------- #
col_max = {}
col_mean = {}
col_mean2 = {}
col_std = {}

for city in cities_list:
  col_mean[city] = {}
  for col in train_set[city]:
    if col in ["index", "Date", "City"]:
      continue

    train_set[city][col] = train_set[city][col].astype("float")
    test_set[city][col] = test_set[city][col].astype("float")

    if col in ["pm25_median", "o3_median", "so2_median", "no2_median", "pm10_median", "co_median"]:
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

  if DROP_ONEHOT:
    train_set[city].drop(train_set[city].columns[-19:], axis=1, inplace=True)
    test_set[city].drop(test_set[city].columns[-19:], axis=1, inplace=True)

class CityDataP(torch.utils.data.Dataset):
  def __init__(self, selected_column, split):
    self.split = split
    if split == "train":
      self.dataset = train_set
    else:
      self.dataset = test_set

    self.valid_city_idx = 0
    self.valid_day_idx = 0
    self.selected_column = selected_column

  def __getitem__(self, idx):
    if self.split != "train":
      # getting all data out of the validation set
      out, city = self.get_idx_data(idx)
    
    else:
      # getting data randomly for train split
      city = random.choice(cities_list)
      _df = self.dataset[city]
      start_idx = random.randint(1,_df.shape[0]-SEQ_LENGTH)
      out =  _df.iloc[start_idx-1:start_idx+SEQ_LENGTH]

    out = out.drop(['index', 'Date', 'City'], axis=1)

    Y = pd.DataFrame({})
    Y_true = pd.DataFrame({})

    for col in out.columns:
      if col == self.selected_column:
        Y_true[col] = out[col]
        Y[col] = out[col].fillna(col_mean[city][col])
      
      if col in ["pm25_median", "pm10_median", "o3_median", "so2_median", "no2_median", "co_median"]:
        out.drop([col], axis=1, inplace=True)
      else:
        out[col] = out[col].astype("float")

    out = np.concatenate((np.array(out)[1:,:], np.array(Y)[:-1,:]), axis=1)
    Y = np.array(Y)[1:]
    Y_true = np.array(Y_true)[1:]
    
    return out, Y, Y_true

  def get_idx_data(self, idx):
    city = cities_list[self.valid_city_idx]
    _df = self.dataset[city]

    out =  _df.iloc[self.valid_day_idx:self.valid_day_idx+SEQ_LENGTH]
    
    if self.valid_day_idx+SEQ_LENGTH >= _df.shape[0]:
      self.valid_day_idx = 0
      self.valid_city_idx += 1
    else:
      self.valid_day_idx += 1

    return out, city

  def __len__(self):
    if self.split != "train":
      return (61-SEQ_LENGTH)*len(cities_list)
    return len(all_train) - (SEQ_LENGTH - 1)*len(cities_list)

class CityDataForecast(torch.utils.data.Dataset):
  def __init__(self, selected_column, split):
    self.split = split
    if split == "train":
      self.dataset = train_set
    else:
      self.dataset = test_set

    self.valid_city_idx = 0
    self.valid_day_idx = 0
    self.selected_column = selected_column

  def __getitem__(self, idx):
    city = random.choice(cities_list)
    _df = self.dataset[city]
    start_idx = random.randint(1,_df.shape[0]-1)
    out =  _df.iloc[start_idx-1:start_idx+1]

    out = out.drop(['index', 'Date', 'City'], axis=1)

    Y = pd.DataFrame({})
    Y_true = pd.DataFrame({})

    for col in out.columns:
      if col == self.selected_column:
        Y_true[col] = out[col]
        #print(out[col])
        Y[col] = out[col].fillna(col_mean[city][col])
      
      if col in ["pm25_median", "pm10_median", "o3_median", "so2_median", "no2_median", "co_median"]:
        out.drop([col], axis=1, inplace=True)
      else:
        out[col] = out[col].astype("float")

    # print(out.shape)
    out = np.concatenate((np.array(out)[1:,:], np.array(Y)[:-1,:]), axis=1)
    Y = np.array(Y)[1:]
    Y_true = np.array(Y_true)[1:]
    

    return out, Y, Y_true

  def get_idx_data(self, idx):
    city = cities_list[self.valid_city_idx]
    _df = self.dataset[city]

    out =  _df.iloc[self.valid_day_idx:self.valid_day_idx+1]
    
    if self.valid_day_idx+1 >= _df.shape[0]:
      self.valid_day_idx = 0
      self.valid_city_idx += 1
    else:
      self.valid_day_idx += 1

    return out, city

  def __len__(self):
    if self.split != "train":
      return (61-1)*len(cities_list)
    return len(all_train) - (1 - 1)*len(cities_list) 

# function that implement the look_ahead mask for masking future time steps. 
def create_look_ahead_mask(size, device=device):
    mask = torch.ones((size, size), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask  # (size, size)

if __name__ == '__main__':

  for SELECTED_COLUMN in ["o3_median", "pm25_median", "so2_median", "pm10_median", "no2_median", "co_median"]:
      
      train_data = CityDataForecast(SELECTED_COLUMN, "train")
      val_data = CityDataForecast(SELECTED_COLUMN, "test")

      sampleLoader = DataLoader(train_data, 32, shuffle=True, num_workers=4)
      val_loader = DataLoader(val_data, 4096, shuffle=False, num_workers=4)

      for model_name in ['OLS', 'LASSO', 'Ridge', 'Elastic']:
          lr = 0.001
          n_epochs = 10
          
          print("set l1,l2 loss")
          l1_lmbda = 0.01
          l1_lmbda = torch.FloatTensor([l1_lmbda]).cuda()
          l1_reg = torch.tensor(0., requires_grad=True).to(device)
          l2_lmbda = 0.01
          l2_lmbda = torch.FloatTensor([l2_lmbda]).cuda()
          l2_reg = torch.tensor(0., requires_grad=True).to(device)

          criterion = nn.MSELoss()
          
          if model_name == 'OLS':
              model = LinearRegression(input_dim=11).to(device)
          elif model_name == 'LASSO':
              model = LinearRegression(input_dim=11).to(device)
          elif model_name == 'Ridge':
              model = LinearRegression(input_dim=11).to(device)
          elif model_name == 'Elastic':
              model = LinearRegression(input_dim=11).to(device)    
          else :
            raise Exception("Wrong Model")
          
          
          opt = torch.optim.Adam(model.parameters(), lr=lr)
          
          print('Start ' + model_name + 'model training')
          best_mse = 2000.0
          mape = 2000.0
          best_model = None

          for epoch in range(1, n_epochs + 1):
              epoch_loss = 0
              batch_idx = 0
              bar = tqdm(sampleLoader)

              model.train()
              for x_batch, y_batch, _ in bar:
                  model.train()
                  x_batch = x_batch.cuda().float()
                  y_batch = y_batch.cuda().float()

                  out = model(x_batch)
                  opt.zero_grad()

                  if model_name == 'OLS':
                      loss = criterion(out[:,-1,:], y_batch[:,-1,:])               
                  elif model_name == 'LASSO':
                      l1_reg = torch.norm(model.linear.weight, p=1)
                      loss = criterion(out[:,-1,:], y_batch[:,-1,:]) + l1_lmbda * l1_reg
                  elif model_name == 'Ridge':
                      l2_reg = torch.norm(model.linear.weight, p=2)
                      loss = criterion(out[:,-1,:], y_batch[:,-1,:]) + l2_lmbda * l2_reg
                  elif model_name == 'Elastic':
                      l1_reg = torch.norm(model.linear.weight, p=1)
                      l2_reg = torch.norm(model.linear.weight, p=2)
                      loss = criterion(out[:,-1,:], y_batch[:,-1,:]) + l1_lmbda * l1_reg + l2_lmbda * l2_reg
                  
                 
                  epoch_loss = (epoch_loss*batch_idx + loss.item())/(batch_idx+1)
                  loss.backward()
                  opt.step()

                  bar.set_description(str(epoch_loss))
                  batch_idx += 1

              # Evaluation
              model.eval()
              mse_list = []
              total_se = 0.0
              total_pe = 0.0
              total_valid = 0.0

              for x_val, _, y_val in val_loader:
                  x_val, y_val = [t.cuda().float() for t in (x_val, y_val)]
                  
                  out = model(x_val)

                  ytrue = y_val[:,-1,:].squeeze().cpu().numpy()
                  ypred = out[:,-1,:].squeeze().cpu().detach().numpy()
                  
                  true_valid = np.isnan(ytrue) != 1
                  ytrue = ytrue[true_valid] #np.nan_to_num(ytrue, 0)
                  ypred = ypred[true_valid]

                  if normalization_type == 'mean_std':
                    ytrue = (ytrue * col_std[SELECTED_COLUMN]) + col_mean2[SELECTED_COLUMN]
                    ypred = (ypred * col_std[SELECTED_COLUMN]) + col_mean2[SELECTED_COLUMN]
                  
                  else:
                    ytrue = (ytrue * col_max[SELECTED_COLUMN])
                    ypred = (ypred * col_max[SELECTED_COLUMN])

                  se = (ytrue - ypred)**2 # np.square(ytrue - ypred)
                  pe = np.abs((ytrue - ypred) / (ytrue + 0.0001))
                  total_se += np.sum(se)
                  total_pe += np.sum(pe)
                  total_valid += np.sum(true_valid)

              eval_mse = total_se / total_valid # np.mean(se) # 
              eval_mape = total_pe / total_valid # np.mean(pe) # 
              print('valid samples:', total_valid)
              print('Eval MSE: ', eval_mse)
              print('Eval RMSE: {}: '.format(SELECTED_COLUMN), np.sqrt(eval_mse))
              print('Eval MAPE: {}: '.format(SELECTED_COLUMN), eval_mape*100)

              if eval_mse < best_mse:
                best_model = deepcopy(model)
                best_mse = eval_mse
                mape = eval_mape*100
                torch.save(best_model.state_dict(), "./save/"+SELECTED_COLUMN+"/"+model_name+".pth")
                
          print(model_name)   
          print("Best MSE :", best_mse)
          print("RMSE :", np.sqrt(best_mse))
          print("MAPE :", mape)
          print()