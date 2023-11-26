import numpy as np
import torch
import torch.nn as nn

class NN(nn.Module):
    '''
    train LSTM encoder-decoder and make predictions
    モデル：全変数をLSTMに入力し、二層の線型結合層に通して予測値を出力
    '''

    def __init__(self, num_features, num_features_pred, device):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        self.num_features = num_features
        self.num_features_pred = num_features_pred
        self.device = device

        super(NN, self).__init__()
        # define fc layer
        self.linear = nn.Linear(num_features, 256)
        self.linear2 = nn.Linear(256, 1024)
        self.linear3 = nn.Linear(1024, 128)
        # self.linear4 = nn.Linear(2048, 512)
        self.linear5 = nn.Linear(128, num_features_pred)
        # define relu
        self.relu = nn.ReLU()
        # define dropout
        self.dropout = nn.Dropout(0.2)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_criterion(self, criterion):
        self.criterion = criterion

    def forward(self, src):
        '''
        : param src:    input tensor with shape (batch_size, seq_len,  number features); PyTorch tensor
        '''
        output = self.linear(src)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear3(output)
        output = self.relu(output)
        output = self.dropout(output)
        # output = self.linear4(output)
        # output = self.relu(output)
        # output = self.dropout(output)
        output = self.linear5(output)

        return output

def nn_train(model, data_loader):
    total_loss = []
    for src, tgt in data_loader:
        src = src.to(model.device)
        tgt = tgt.to(model.device)

        # zero the gradient
        model.optimizer.zero_grad()

        output = model.forward(src=src)

        # compute the loss
        loss = model.criterion(output, tgt)

        total_loss.append(loss.cpu().detach())

        # backpropagation
        loss.backward()
        model.optimizer.step()

    # change learning rate
    model.scheduler.step()

    # return loss for epoch
    return np.average(total_loss)


def nn_predict(model, data_loader):
    model.eval()
    with torch.no_grad():
        all_outputs = torch.Tensor().to(model.device)
        for src, tgt in data_loader:
            src = src.to(model.device)
            tgt = tgt.to(model.device)

            output = model.forward(src=src)
            all_outputs = torch.cat([all_outputs, output], dim=0)

    return all_outputs.to('cpu').detach().numpy()
