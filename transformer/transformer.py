import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from tqdm.notebook import tqdm
from TimeFeature import time_features
from Embedding import DataEmbedding

class TransSequenceDataset(Dataset):
    def __init__(self, dataframe, seq_len_src=56, seq_len_tgt=8, lag_len=1):
        self.seq_len_src = seq_len_src
        self.lag_len = lag_len
        self.seq_len_tgt = seq_len_tgt
        self.df = torch.tensor(dataframe.values).float()
        self.len = math.floor((len(self.df) - self.seq_len_src - self.seq_len_tgt) / self.lag_len) + 1
        
        df_stamp = pd.DataFrame({"date": dataframe.index})
        self.df_mark = time_features(df_stamp, freq="3h")

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        if type(i) == slice:
            raise Exception('Exception: Expected int, inputted slice!')
            
        X = self.df[i*self.lag_len:i*self.lag_len+self.seq_len_src]
        Y = self.df[i*self.lag_len+self.seq_len_src:i*self.lag_len+self.seq_len_src+self.seq_len_tgt]
        
        X_mark = self.df_mark[i*self.lag_len:i*self.lag_len+self.seq_len_src]
        Y_mark = self.df_mark[i*self.lag_len+self.seq_len_src:i*self.lag_len+self.seq_len_src+self.seq_len_tgt]
        
        return X, Y, torch.tensor(X_mark).float(), torch.tensor(Y_mark).float()
    
    def input_size(self):
        return self.df.shape[1]

def create_mask(src, tgt, device):
    seq_len_src = src.shape[1]
    seq_len_tgt = tgt.shape[1]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt).to(device)
    mask_src = generate_square_subsequent_mask(seq_len_src).to(device)

    return mask_src, mask_tgt

def generate_square_subsequent_mask(seq_len):
    return torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

class Transformer(nn.Module):
    def __init__(self, num_features, num_features_pred, device, num_encoder_layers, num_decoder_layers,
        enc_in, dec_in, d_model, d_output,
        dim_feedforward = 512, dropout = 0.1, nhead = 8, freq='h'):

        super(Transformer, self).__init__()
        self.num_features = num_features
        self.num_features_pred = num_features_pred
        self.device = device

        # define embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, freq, dropout)

        # define encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                               )
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                                      num_layers=num_encoder_layers,
                                                      norm=encoder_norm
                                                     )

        # define decoder
        decoder_layer = TransformerDecoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                               )
        decoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layer,
                                                      num_layers=num_decoder_layers,
                                                      norm=decoder_norm)

        # define output layer
        self.linear = nn.Linear(d_model, d_output)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_criterion(self, criterion):
        self.criterion = criterion


    def forward(self, src, tgt, src_mark, tgt_mark, mask_src, mask_tgt):
        # mask_src, mask_tgtはセルフアテンションの際に未来のデータにアテンションを向けないためのマスク

        embedding_src = self.enc_embedding(src, src_mark)
        memory = self.transformer_encoder(embedding_src, mask_src)

        embedding_tgt = self.dec_embedding(tgt, tgt_mark)
        outs = self.transformer_decoder(embedding_tgt, memory, mask_tgt)

        output = self.linear(outs)
        return output

    def encode(self, src, src_mark, mask_src):
        return self.transformer_encoder(self.enc_embedding(src, src_mark), mask_src)

    def decode(self, tgt, tgt_mark, memory, mask_tgt):
        return self.transformer_decoder(self.dec_embedding(tgt, tgt_mark), memory, mask_tgt)

def trans_train(model, data_loader):
    model.train()
    total_loss = []
    for src, tgt, src_mark, tgt_mark in data_loader:

        src = src.float().to(model.device)
        tgt = tgt.float().to(model.device)
        src_mark = src_mark.float().to(model.device)
        tgt_mark = tgt_mark.float().to(model.device)

        input_tgt = torch.cat((src[:,-1:,:],tgt[:,:-1,:]), dim=1)
        input_tgt_mark = torch.cat((src_mark[:,-1:,:],tgt_mark[:,:-1,:]), dim=1)

        mask_src, mask_tgt = create_mask(src, input_tgt, model.device)

        output = model(src=src, tgt=input_tgt, src_mark = src_mark, tgt_mark = input_tgt_mark, mask_src=mask_src, mask_tgt=mask_tgt)

        model.optimizer.zero_grad()

        loss = model.criterion(output[:,:,0:model.num_features_pred], tgt[:,:,0:model.num_features_pred])

        loss.backward()
        total_loss.append(loss.cpu().detach())
        model.optimizer.step()

    # change learning rate
    model.scheduler.step()

    return np.average(total_loss)


def trans_predict(model, dataset):
    model.eval()
    with torch.no_grad():
        all_outputs = torch.zeros(1, 1, model.num_features).to(model.device)
        for t in tqdm(range(len(dataset))):  # (len(df_test)-8)/8 = 92回予測が必要
            src = dataset[t][0].float().to(model.device)
            tgt = dataset[t][1].float().to(model.device)
            src_mark = dataset[t][2].float().to(model.device)
            tgt_mark = dataset[t][3].float().to(model.device)

            # add a batch size of 1 for the encoder (= [1, :, :])
            src = src.unsqueeze(0)
            tgt = tgt.unsqueeze(0)
            src_mark = src_mark.unsqueeze(0)
            tgt_mark = tgt_mark.unsqueeze(0)

            seq_len_src = src.shape[1]
            seq_len_tgt = tgt.shape[1]

            mask_src = (torch.zeros(seq_len_src, seq_len_src)).type(torch.bool)
            mask_src = mask_src.float().to(model.device)

            memory = model.encode(src, src_mark, mask_src)
            outputs = src[:, -1:, :]

            #ループさせて逐次的に予測する
            for i in range(seq_len_tgt):

                mask_tgt = (generate_square_subsequent_mask(outputs.size(1))).to(model.device)

                output = model.decode(outputs, tgt_mark[:, i:i+1, :], memory, mask_tgt)
                output = model.linear(output)  # output.shape = [バッチサイズ1, ウィンドウサイズi(累積される), 変数]

                # convert predicted dummies to the actual ones
                output = torch.cat([output[:, -1:, :model.num_features_pred], tgt[:, i:i+1, model.num_features_pred:]], dim=2)

                # concat outputs and output above
                outputs = torch.cat([outputs, output], dim=1)

            all_outputs = torch.cat([all_outputs, outputs[:, 1:, :]], dim=1)

    return all_outputs[-1, 1:, :model.num_features_pred].to('cpu').detach().numpy()  # all_outputs[-1, 1:, :num_features_pred] → バッチサイズを消す＆all_outputsの最初の0を消す
