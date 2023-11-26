import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

class LSTMSequenceDataset(Dataset):
    def __init__(self, dataframe, seq_len_src=56, seq_len_tgt=8, lag_len=8):
        self.seq_len_src = seq_len_src
        self.lag_len = lag_len
        self.seq_len_tgt = seq_len_tgt
        self.df = torch.tensor(dataframe.values).float()
        self.len = math.floor((len(self.df) - self.seq_len_src - self.seq_len_tgt) / self.lag_len) + 1

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        if type(i) == slice:
            raise Exception('Exception: Expected int, inputted slice!')

        X = self.df[i*self.lag_len:i*self.lag_len+self.seq_len_src]
        Y = self.df[i*self.lag_len+self.seq_len_src:i*self.lag_len+self.seq_len_src+self.seq_len_tgt]

        return X, Y

    def input_size(self):
        return self.df.shape[1]

# lstm - cat - fc -
class LSTM(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers, num_features_pred, device):
        super().__init__()
        self.num_features= num_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_features_pred = num_features_pred
        self.device = device
        
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_units,
                            batch_first=True,
                            num_layers=num_layers,
                            dropout = 0.3)
        self.fc1 = nn.Linear(in_features=hidden_units, out_features=num_features)
        self.fc2 = nn.Linear(in_features=num_features, out_features=num_features_pred)
        self.relu = nn.ReLU()  

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_criterion(self, criterion):
        self.criterion = criterion
        
    def forward(self, src, hidden_states):     
        lstm_out, (h0, c0) = self.lstm(src, hidden_states)  # output size = (batch, sequence_length, hidden_size)
        output = self.fc1(lstm_out)
        output = self.relu(output)
        output = self.fc2(output)
        
        return output[:, -1:, :self.num_features_pred], (h0, c0)  # output.shape = [batch_size, seq_len_src, num_features_pred]
    
def lstm_train(model, data_loader):
    total_losses = []
    model.train()
    for src, tgt in data_loader:
        src = src.to(model.device)
        tgt = tgt.to(model.device)
        
        # initialize hidden states
        hidden_states = None
        
        # # input for lstm
        # input_tgt = torch.cat((src[:,-1:,:], tgt[:, :-1, :]), dim=1)
        
        output, hidden = model(src, hidden_states)
        
        # compute the loss
        loss = model.criterion(output, tgt[:, :, :model.num_features_pred])

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        total_losses.append(loss.item())
    
    # change learning rate
    model.scheduler.step()

    return np.average(total_losses)

def lstm_predict(model, dataset):
    model.eval()
    with torch.no_grad():   
        all_outputs = torch.Tensor().to(model.device)
        for i in tqdm(range(len(dataset))):
            src = dataset[i][0].float().to(model.device)
            tgt = dataset[i][1].float().to(model.device)
            
            # add a batch_size 1
            src = src.unsqueeze(0)
            tgt = tgt.unsqueeze(0)

            # initialize tensor for predictions
            outputs = torch.Tensor().to(model.device)  # outputs.shape =  [seq_len, num_features_pred]
            
            # initialize hidden states
            hidden = None
            
            # input_tensor
            lstm_input = src
            
            # predict recursively
            for t in range(tgt.shape[1]):
                lstm_output, hidden = model(lstm_input, hidden)   # lstm_output.shape = [batch_size, 1, num_features_pred]
                outputs = torch.cat((outputs, lstm_output[:, -1, :]), dim=0)
                
                # input at the next period = (location output from prediction) + (acurate dummy input from test data)
                lstm_output = torch.cat((lstm_output, tgt[:, t:t+1, model.num_features_pred:]), dim=2)  # lstm_output.shape = [batch_size, 1, num_features]
                
                # outputの1期分を付け加えて、最初の1期を消す
                lstm_input = torch.cat([lstm_input[:, 1:, :], lstm_output[:, -1:, :]], dim=1)  # lstm_input.shape = [batch_size, seq_len_src, num_features]            

            all_outputs = torch.cat([all_outputs, outputs], dim=0)
        
    return all_outputs[:, :model.num_features_pred].to('cpu').detach().numpy()
