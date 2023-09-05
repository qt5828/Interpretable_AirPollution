import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import os 
import random
import warnings

import torch
import datetime

from loss_utils import *
from models import *

from torch import nn
from torch import optim

# SET HERE
SELECTED_COLUMN = "median_so2" # ["median_pm25", "median_o3", "median_so2", "median_no2", "median_pm10", "median_co"]:
SEQ_LENGTH = 3


DROP_ONEHOT = True

if DROP_ONEHOT:
    INPUT_DIM = 8   
else:
    INPUT_DIM = 29
HIDDEN_DIM = 32
LAYER_DIM = 3
normalization_type = 'mean_std' # 'max', mean_std
col_max = {}
col_mean = {}
col_mean2 = {}
col_std = {}

# For model performance
MSEs = {}
RMSEs = {}
MAPEs = {}
models = ['RNN', 'LSTM', 'TransLSTM', 'Transformer', 'CosFormer', 'CosSquareFormer', 'OLS', 'LASSO', 'Ridge', 'Elastic']
    
device = torch.device("cuda")


def get_train_test_data(df):
  # we'll mostly need median and variance values of features for most of our needs

  for col in df.columns:
    for x in ["Country", "min", "max", "count", "County", "past_week", "latitude", "longitude", "State", "variance"]:
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

  # using one-hot on month and weekday
  weekday_onehot = pd.get_dummies(df['weekday'])
  weekday_onehot.columns = ["day_"+str(x) for x in weekday_onehot]
  month_onehot = pd.get_dummies(df['month'])
  month_onehot.columns = ["month_"+str(x) for x in month_onehot]

  df.drop(['weekday', 'month'], axis=1, inplace=True)
  df = df.join([weekday_onehot, month_onehot])

  cities_list = list(set(df['City']))
  print(cities_list)
  cities_list.sort()
  print(cities_list)
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
  def __init__(self, selected_column, split, city_to_use):
    self.split = split
    if split == "train":
      self.dataset = train_set
    else:
      self.dataset = test_set

    self.valid_day_idx = 1
    self.selected_column = selected_column
    self.city_to_use = city_to_use
    
    self.city_index = cities_list.index(self.city_to_use)

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
      city = self.city_to_use
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
    # print(out.columns) 
    
    # if self.split != 'train':
      # print(out.columns)
      # print(np.array(out)[:,:])
      # print(np.array(Y)[:,:])
      # print(np.array(out)[1:,:])
      # print(np.array(Y)[:-1,:])
      # out = np.concatenate((np.array(out)[1:,:], np.array(Y)[:-1,:]), axis=1)
      # print(out)
      
      # raise Exception("FINISH")
    out = np.concatenate((np.array(out)[1:,:], np.array(Y)[:-1,:]), axis=1)
    Y = np.array(Y)[1:]
    Y_true = np.array(Y_true)[1:]
    
    # print("out shape : ", out.shape)
    # print("Y shape : ", Y.shape)
    # print("Y_true shape : ", Y_true.shape)
    
    return out, Y, Y_true

  def get_idx_data(self, idx):
    # city_idx = self.valid_city_idx % len(cities_list)
    # city = cities_list[city_idx]
    city = self.city_to_use
    _df = self.dataset[city]

    # -1~seq-1(seq+1길이만큼!)
    out =  _df.iloc[self.valid_day_idx-1:self.valid_day_idx+SEQ_LENGTH]
    # print(out)
    # raise Exception("FINISH")
    
    if self.valid_day_idx+SEQ_LENGTH >= _df.shape[0]:
      self.valid_day_idx = 1
      # self.valid_city_idx += 1
    else:
      self.valid_day_idx += 1

    return out, city

  def __len__(self):
    if self.split != "train":
      return (61-SEQ_LENGTH)
    return len(train_set[self.city_to_use]) - (SEQ_LENGTH - 1)


class CityDataEstimatePollutant(torch.utils.data.Dataset):
  def __init__(self, selected_column, split, city_to_use):
    self.split = split
    if split == "train":
      self.dataset = train_set
    else:
      self.dataset = test_set

    self.valid_day_idx = 0
    self.selected_column = selected_column
    self.city_to_use = city_to_use
    
    self.city_index = cities_list.index(self.city_to_use)
    

  def __getitem__(self, idx):
    # TEST
    if self.split != "train":
      out, city = self.get_idx_data(idx)
      
    # TRAIN
    else:
      city = self.city_to_use
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
    city = city_to_use
    _df = self.dataset[city]

    out =  _df.iloc[self.valid_day_idx:self.valid_day_idx+1]
    
    # _df.shaoe[0]-1
    if self.valid_day_idx >= _df.shape[0]:
      self.valid_day_idx = 0
    else:
      self.valid_day_idx += 1

    # print(self.valid_day_idx)
    return out, city

  def __len__(self):
    return self.dataset[self.city_to_use].shape[0]


# function that implement the look_ahead mask for masking future time steps. 
def create_look_ahead_mask(size, device):
    mask = torch.ones((size, size), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask  # (size, size)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
# Evaluation
def evaluation(val_loader, model, model_name, LUR, SELECTED_COLUMN, mask=False):
    model.to(device)
    model.eval()
    mse_list = []
    total_se = 0.0
    total_pe = 0.0
    total_valid = 0.0

    for x_val, _, y_val in val_loader:
        x_val, y_val = [t.to(device).float() for t in (x_val, y_val)]
        
        if mask:
            masking = create_look_ahead_mask(x_val.shape[1], device)
            out, _ = model(x_val.to(device), masking)
        else:
            out = model(x_val.to(device))

        if LUR :
            ytrue = y_val[:,-1,:].squeeze().cpu().numpy()
            ypred = out[:,-1,:].squeeze().cpu().detach().numpy()
        else:
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
    
    return eval_mse, eval_mape*100


# Train
def train(sampleLoader, val_loader, model, model_name, SELECTED_COLUMN, mask=False, LUR=False, l1=False, l2=False):

    lr = 0.01
    n_epochs = 10   
    
    model.to(device)

    criterion = nn.MSELoss()
    
    # LUR
    if LUR:
        print("set l1,l2 loss")
        l1_lmbda = 0.01
        l1_lmbda = torch.FloatTensor([l1_lmbda]).to(device)
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
        l2_lmbda = 0.01
        l2_lmbda = torch.FloatTensor([l2_lmbda]).to(device)
        l2_reg = torch.tensor(0., requires_grad=True).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
            
    # DL
    else:
        print("set SoftDTW loss")
        lmbda = 0.5
        dtw_loss = SoftDTW(use_cuda=True, gamma=0.1)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    
      
    print('Start ' + model_name + ' training')
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
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            if mask==True:
                masking = create_look_ahead_mask(x_batch.shape[1], device)
                out, _ = model(x_batch.to(device), masking)
            else :
                out = model(x_batch.to(device))
            opt.zero_grad()
            if LUR:
                # LASSO
                if l1==True and l2==False:
                    l1_reg = torch.norm(model.linear.weight, p=1)
                    loss = criterion(out[:,-1,:], y_batch[:,-1,:]) + l1_lmbda * l1_reg
                # Ridge
                elif l1==False and l2==True:
                    l2_reg = torch.norm(model.linear.weight, p=2)
                    loss = criterion(out[:,-1,:], y_batch[:,-1,:]) + l2_lmbda * l2_reg
                # Elastic
                elif l1==True and l2==True:
                    l1_reg = torch.norm(model.linear.weight, p=1)
                    l2_reg = torch.norm(model.linear.weight, p=2)
                    loss = criterion(out[:,-1,:], y_batch[:,-1,:]) + l1_lmbda * l1_reg + l2_lmbda * l2_reg
                # OLS
                else:
                    # print(out.shape)
                    loss = criterion(out[:,-1,:], y_batch[:,-1,:])
            else:
                # print(out.shape)
                loss = criterion(out[:,-1,:], y_batch[:,-1,:]) + lmbda * dtw_loss(out.to(device),y_batch.to(device)).mean()
            epoch_loss = (epoch_loss*batch_idx + loss.item())/(batch_idx+1)
            loss.backward(retain_graph=True)
            opt.step()
            bar.set_description(str(epoch_loss))
        eval_mse, eval_mape = evaluation(val_loader, model, model_name, LUR, SELECTED_COLUMN, mask)
        
        city = sampleLoader.dataset.city_to_use

        if eval_mse < best_mse:
            best_model = deepcopy(model)
            best_mse = eval_mse
            mape = eval_mape
            torch.save(best_model.state_dict(), "./save_US/"+SELECTED_COLUMN+"/"+model_name+"_"+city+"{}.pth".format(SEQ_LENGTH))
      
    print(model_name)   
    print("Best MSE :", best_mse)
    print("RMSE :", np.sqrt(best_mse))
    print("MAPE :", mape)
    print()
    MSEs[model_name].append(best_mse)
    RMSEs[model_name].append(np.sqrt(best_mse))
    MAPEs[model_name].append(mape)
    
    
    


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    SEED_VALUE = 100
    os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    df = pd.read_csv("./data/US2019to2023.csv")
    
    
    # Data Preprocessing
    train_set, test_set = get_train_test_data(df)

    cities_list = list(train_set.keys())

    all_train = pd.DataFrame()
    for city in cities_list:
        all_train = all_train.append(train_set[city], ignore_index=True)

    all_test = pd.DataFrame({})
    for city in test_set:
        all_test = all_test.append(test_set[city], ignore_index=True)

    concat_df = pd.concat([all_train,all_test],axis=0)

    for city in cities_list:
        col_mean[city] = {}
        for col in train_set[city]:
            if col in ["index", "Date", "City"]:
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

        if DROP_ONEHOT:
            train_set[city].drop(train_set[city].columns[-19:], axis=1, inplace=True)
            test_set[city].drop(test_set[city].columns[-19:], axis=1, inplace=True)

    
    # number of data per city
    print("num of cities : ", len(cities_list))
    for city in cities_list:
        print(city+"({})".format(len(train_set[city])), end=", ")
        
    # Fix Seed for DataLoader
    g = torch.Generator()
    g.manual_seed(0)
    
    for model in models :
        MSEs[model] = []
        RMSEs[model] = []
        MAPEs[model] = []
    
    
    for city_to_use in cities_list:
        print(city_to_use)

        train_data = CityDataForecast(SELECTED_COLUMN, "train", city_to_use)
        val_data = CityDataForecast(SELECTED_COLUMN, "test", city_to_use)
        sampleLoader = DataLoader(train_data, 32, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_data, 4096, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

        # RNN
        RNNmodel = RNN(1, INPUT_DIM+1, HIDDEN_DIM, LAYER_DIM).to(device)
        train(sampleLoader, val_loader, RNNmodel, "RNN", SELECTED_COLUMN)
        # LSTM
        LSTMmodel = LSTM(1, INPUT_DIM+1, HIDDEN_DIM, LAYER_DIM).to(device)
        train(sampleLoader, val_loader, LSTMmodel, "LSTM", SELECTED_COLUMN)
        # BiLSTM
        # BiLSTMmodel = LSTM(1, INPUT_DIM+1, HIDDEN_DIM, LAYER_DIM, bidirectional=True).to(device)
        # train(sampleLoader, val_loader, BiLSTMmodel, "BiLSTM", SELECTED_COLUMN)
        # TransLSTM
        TransLSTMmodel = TransLSTM(num_layers=3, D=16, H=5, hidden_mlp_dim=32, inp_features=INPUT_DIM+1, out_features=1, dropout_rate=0.1, LSTM_module = LSTM(4, INPUT_DIM+1, HIDDEN_DIM, LAYER_DIM, bidirectional = False).to(device), attention_type='regular').to(device) # cosine_square, cosine, regular # 6L, 12H
        train(sampleLoader, val_loader, TransLSTMmodel, "TransLSTM", SELECTED_COLUMN, mask=True)
        # Transformer
        Transmodel = Transformer(num_layers=6, D=16, H=10, hidden_mlp_dim=32, inp_features=INPUT_DIM+1, out_features=1, dropout_rate=0.1, attention_type='regular', SL=SEQ_LENGTH).to(device) # cosine_square, cosine, regular # 6L, 12H
        train(sampleLoader, val_loader, Transmodel, "Transformer", SELECTED_COLUMN, mask=True)
        # CosFormer
        TransCosModel = Transformer(num_layers=6, D=16, H=10, hidden_mlp_dim=32, inp_features=INPUT_DIM+1, out_features=1, dropout_rate=0.1, attention_type='cosine', SL=SEQ_LENGTH).to(device) # cosine_square, cosine, regular # 6L, 12
        train(sampleLoader, val_loader, TransCosModel, "CosFormer", SELECTED_COLUMN, mask=True)
        # CosSquareFormer
        TransCosSquare = Transformer(num_layers=6, D=16, H=10, hidden_mlp_dim=32, inp_features=INPUT_DIM+1, out_features=1, dropout_rate=0.1, attention_type='cosine_square', SL=SEQ_LENGTH).to(device) # cosine_square, cosine, regular # 6L, 12H
        train(sampleLoader, val_loader, TransCosModel, "CosSquareFormer", SELECTED_COLUMN, mask=True)


    print("MSEs")
    for key in MSEs :
        for val in MSEs[key]:
            print(key, val, end=",")
        print()
    print()    

    print("RMSEs")
    for key in RMSEs :
        for val in RMSEs[key]:
            print(key, val, end=",")
        print()
    print() 

    print("MAPEs")
    for key in MAPEs :
        for val in MAPEs[key]:
            print(key, val, end=",")
        print()
    print() 
    
    for city_to_use in cities_list:
        print(city_to_use)
        
        train_data = CityDataEstimatePollutant(SELECTED_COLUMN, "train", city_to_use)
        val_data = CityDataEstimatePollutant(SELECTED_COLUMN, "test", city_to_use)
        sampleLoader = DataLoader(train_data, 32, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_data, 4096, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
        
        OLS = LinearRegression(input_dim=INPUT_DIM)
        train(sampleLoader, val_loader, OLS, "OLS", SELECTED_COLUMN, LUR=True)
        LASSO = LinearRegression(input_dim=INPUT_DIM)
        train(sampleLoader, val_loader, LASSO, "LASSO", SELECTED_COLUMN, LUR=True, l1=True)
        Ridge = LinearRegression(input_dim=INPUT_DIM)
        train(sampleLoader, val_loader, Ridge, "Ridge", SELECTED_COLUMN, LUR=True, l2=True)
        Elastic = LinearRegression(input_dim=INPUT_DIM)
        train(sampleLoader, val_loader, Elastic, "Elastic", SELECTED_COLUMN, LUR=True, l1=True, l2=True)
        
    print("MSEs")
    for key in MSEs :
        for val in MSEs[key]:
            print(key, val, end=",")
        print()
    print()    

    print("RMSEs")
    for key in RMSEs :
        for val in RMSEs[key]:
            print(key, val, end=",")
        print()
    print() 

    print("MAPEs")
    for key in MAPEs :
        for val in MAPEs[key]:
            print(key, val, end=",")
        print()
    print() 