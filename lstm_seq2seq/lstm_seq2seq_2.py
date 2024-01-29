import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

# set random seed
fix_seed = 1111
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)

class SequenceDataset(Dataset):
    def __init__(self, dataframe, seq_len_src=56, seq_len_tgt=8, lag_len=1):
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

class LSTM_seq2seq(nn.Module):
    '''
    train LSTM encoder-decoder and make predictions
    モデル1：全変数をLSTMに入力し、二層の線型結合層に通して予測値を出力
    '''

    def __init__(self, num_features, hidden_size, num_layers, dropout, num_features_pred, d_linear, device):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(LSTM_seq2seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_features_pred = num_features_pred
        self.device = device

        # define LSTM layer
        self.encoder_lstm = nn.LSTM(input_size = num_features, hidden_size = hidden_size, num_layers = num_layers, batch_first=True, dropout = dropout)
        self.decoder_lstm = nn.LSTM(input_size = num_features, hidden_size = hidden_size, num_layers = num_layers, batch_first=True, dropout = dropout)
        # define fc layer for encoder
        # self.linear = nn.Linear(num_features, 256)
        # self.linear2 = nn.Linear(256, 10)
        # define fc layer for decoder
        self.linear3 = nn.Linear(hidden_size, d_linear)
        self.linear4 = nn.Linear(d_linear, num_features_pred)
        # self.linear4 = nn.Linear(1024, 2048)
        # self.linear5 = nn.Linear(2048, 512)
        # self.linear6 = nn.Linear(512, num_features_pred)

        # self.linear6 = nn.Linear(num_features, 256)
        # self.linear7 = nn.Linear(256, 10)
        # define relu
        self.relu = nn.ReLU()
        # define dropout
        self.dropout = nn.Dropout(dropout)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_criterion(self, criterion):
        self.criterion = criterion

    def encoder(self, src):
        '''
        : param src:                   input of shape (batch_size, seq_len, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''
        h0 = None
        lstm_out, (h,c) = self.decoder_lstm(src, h0)
        # output = self.linear(src)
        # output = self.relu(output)
        # # output = self.dropout(output)
        # output = self.linear2(output)
        # lstm_out, (h,c) = self.encoder_lstm(output, h0)

        return (h, c)

    def decoder(self, src, encoder_hidden_states):
        '''
        : param src:                        should be 3D (batch_size, seq_len, input_size)
        : param encoder_hidden_states:      hidden states; tuple
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence; tuple
        '''

        # output = self.linear6(src)
        # output = self.relu(output)
        # src = self.linear7(output)

        lstm_out, (h, c) = self.decoder_lstm(src, encoder_hidden_states)
        output = self.linear3(lstm_out)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear4(output)
        # output = self.relu(output)
        # output = self.dropout(output)
        # output = self.linear5(output)
        # output = self.relu(output)
        # output = self.dropout(output)
        # output = self.linear6(output)

        return output, (h, c)

    def forward(self, src, tgt):
        '''
        : param src:    input tensor with shape (batch_size, seq_len,  number features); PyTorch tensor
        : param tgt:    target tensor with shape (batch_size, seq_len, number features); PyTorch tensor
        '''

        encoder_hidden = self.encoder(src)  # only encoder_hidden is used
        decoder_output, decoder_hidden = self.decoder(tgt, encoder_hidden)  # decoder_output.shape = [batch_size, seq_len, num_features]

        return decoder_output

def train(model, data_loader):
    model.train()
    total_loss = []
    for _, (src, tgt) in enumerate(data_loader):
        src = src.to(model.device)
        tgt = tgt.to(model.device)

        # target for decoder
        input_tgt = torch.cat((src[:,-1:,:], tgt[:, :-1, :]), dim=1)

        # zero the gradient
        model.optimizer.zero_grad()

        output = model(src=src, tgt=input_tgt)

        # compute the loss
        loss = model.criterion(output[:,:,0:model.num_features_pred], tgt[:,:,0:model.num_features_pred])

        total_loss.append(loss.cpu().detach())

        # backpropagation
        loss.backward()
        model.optimizer.step()

    # change learning rate
    model.scheduler.step()

    # return loss for epoch
    return np.average(total_loss)


def eval(model, data_loader):
    model.eval()
    with torch.no_grad():
        total_loss = []
        for _, (src, tgt) in enumerate(data_loader):
            src = src.to(model.device)
            tgt = tgt.to(model.device)

            # target for decoder
            input_tgt = torch.cat((src[:,-1:,:], tgt[:, :-1, :]), dim=1)

            output = model(src=src, tgt=input_tgt)

            # compute the loss
            loss = model.criterion(output[:,:,0:model.num_features_pred], tgt[:,:,0:model.num_features_pred])

            total_loss.append(loss.cpu().detach())

        # return loss for epoch
        return np.average(total_loss)


def predict(model, data_loader):
    model.eval()
    with torch.no_grad():
        all_outputs = torch.Tensor().to(model.device)
        for src, tgt in data_loader:
            src = src.to(model.device)
            tgt = tgt.to(model.device)

            encoder_hidden = model.encoder(src)
            # input for decoder
            decoder_input = src[:, -1:, :]
            decoder_hidden = encoder_hidden

            outputs = torch.Tensor().to(model.device)
            # predict recursively
            for t in range(tgt.shape[1]):
                decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)  # decoder_output.shape = [batch_size, 1(seq_len), num_features]
                outputs = torch.cat((outputs, decoder_output[:, -1, :]), dim=1)

                # decoder input at the next period = (location output from prediction) + (acurate dummy input from test data)
                decoder_input = torch.cat((decoder_output[:, :, 0:model.num_features_pred], tgt[:, t:t+1, model.num_features_pred:]), dim=2)

            all_outputs = torch.cat([all_outputs, outputs.reshape(-1,model.num_features_pred)], dim=0)

    return all_outputs.to('cpu').detach().numpy()
