import math
import time
import yaml
import datetime
import numpy as np
import pandas as pd
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import plotly.graph_objects as go

 
def get_prediction(raw_df, model_type, model_num_list, result_path):
    resid_df = pd.DataFrame()
    for model_num in model_num_list:
        print('=====================================')
        # all parameters
        row = pd.read_excel(result_path, sheet_name='params')
        time_list = row.iloc[model_num-1,:]['time_list']
        features = [s.strip(" \'") for s in row.iloc[model_num-1,:]['features'][2:-2].split(',')]

        time_list = row.iloc[model_num-1,:]['time_list']
        features = [s.strip(" \'") for s in row.iloc[model_num-1,:]['features'][2:-2].split(',')]
        train_start = row.iloc[model_num-1,:]['train_start']
        train_end = row.iloc[model_num-1,:]['train_end']
        valid_start = row.iloc[model_num-1,:]['valid_start']
        valid_end = row.iloc[model_num-1,:]['valid_end']
        test_start = row.iloc[model_num-1,:]['test_start']
        test_end = row.iloc[model_num-1,:]['test_end']

        seq_len_src = int(row.iloc[model_num-1,:]['seq_len_src'])
        seq_len_tgt = int(row.iloc[model_num-1,:]['seq_len_tgt'])
        lag_len = int(row.iloc[model_num-1,:]['lag_len'])
        batch_size = int(row.iloc[model_num-1,:]['batch_size'])

        if model_type == 'lstm' or 'lstm_seq2seq':
            hidden_size  = int(row.iloc[model_num-1,:]['hidden_size'])
            num_layers = int(row.iloc[model_num-1,:]['num_layers'])
            dropout = row.iloc[model_num-1,:]['dropout']
            d_linear = int(row.iloc[model_num-1,:]['d_linear'])

        elif model_type == 'lstm_nn' or 'lstm_seq2seq_nn':
            hidden_size  = int(row.iloc[model_num-1,:]['hidden_size'])
            num_layers = int(row.iloc[model_num-1,:]['num_layers'])
            dropout = row.iloc[model_num-1,:]['dropout']

        elif model_type == 'transformer':
            nhead = int(row.iloc[model_num-1,:]['nhead'])
            d_model = int(row.iloc[model_num-1,:]['d_model'])
            dim_feedforward = int(row.iloc[model_num-1,:]['dim_feedforward'])
            num_encoder_layers = int(row.iloc[model_num-1,:]['num_encoder_layers'])
            num_decoder_layers = int(row.iloc[model_num-1,:]['num_decoder_layers'])
            dropout = row.iloc[model_num-1,:]['dropout']
            d_linear = int(row.iloc[model_num-1,:]['d_linear'])

        num_features_pred = int(row.iloc[model_num-1,:]['num_features_pred'])
        n_epochs = int(row.iloc[model_num-1,:]['n_epochs'])
        learning_rate = row.iloc[model_num-1,:]['learning_rate']
        milestones = row.iloc[model_num-1,:]['milestones']
        gamma = row.iloc[model_num-1,:]['gamma']


        # construct predict and reorganize columns
        df = raw_df[features]
        df, num_nondummy, num_dummy = my_utils.reorganize_cols(df)

        # standardize option
        standardize_data = True
        ss = my_utils.MyStandardScaler(train_start, train_end)

        # standardize data
        if standardize_data:
            ss.fit(predict.iloc[:, :num_nondummy], use_train=True)
            predict = pd.concat([ss.transform(predict), predict.iloc[:, num_nondummy:]], axis=1)  # reset dummy columns

        # split data
        df_train, df_valid, df_test = my_utils.split_data(predict, time_list, train_start, train_end, valid_start, valid_end, test_start, test_end)

        # df for prediction
        df_valid_for_pred = pd.concat([df_train.iloc[-seq_len_src:, :], df_valid])  # valid_dataの1期目から予測するために、前日もテストデータに含める。
        df_test_for_pred = pd.concat([df_valid.iloc[-seq_len_src:, :], df_test])  # test_dataの1期目から予測するために、前日もテストデータに含める。

        # Set Dataset & Dataloader
        # train
        train_dataset_for_pred = SequenceDataset(df_train, seq_len_src=seq_len_src, seq_len_tgt=seq_len_tgt, lag_len=lag_len)
        train_loader_for_pred  = DataLoader(train_dataset_for_pred, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

        # valid
        valid_dataset_for_pred = SequenceDataset(df_valid_for_pred, seq_len_src=seq_len_src, seq_len_tgt=seq_len_tgt, lag_len=lag_len)
        valid_loader_for_pred  = DataLoader(valid_dataset_for_pred, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

        # test
        test_dataset_for_pred = SequenceDataset(df_test_for_pred, seq_len_src=seq_len_src, seq_len_tgt=seq_len_tgt, lag_len=lag_len)
        test_loader_for_pred  = DataLoader(test_dataset_for_pred, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

        # # check dataset length
        # print("Length of each dataset:")
        # print("    Train:{}, Valid:{}, Test:{}".format(len(train_dataset), len(valid_dataset), len(test_dataset)))

        # # check the shape of a batch
        # X, y, X_mark, y_mark = next(iter(train_loader))
        # print("Source shape:", X.shape)  # [batch size, sequence length, number of features]
        # print("Target shape:", y.shape)
        # print("Source Time Feature shape:", X_mark.shape)  # [batch size, sequence length, number of features]
        # print("Target TimeFeature shape:", y_mark.shape)

        # =============================================================================

        # load model
        name = 'lstm_seq2seq_model_' + str(model_num).zfill(2)
        model = torch.load(root_path + model_path + name + '.pth')

        '''
        predict
        '''
        # print('-------')
        # print('predict train')
        train_pred_tr = predict(model=model, data_loader=train_loader_for_pred)

        if num_features_pred == 1:
            df_train_pred_ss = pd.DataFrame(train_pred_tr, columns=[features[0]], index=df_train.index[seq_len_src:]).add_prefix('pred_')  # for univariate prediction
        else:
            df_train_pred_ss = pd.DataFrame(train_pred_tr, columns=df.columns[:num_features_pred], index=df_train.index[seq_len_src:]).add_prefix('pred_')  # for multivariate prediction

        if standardize_data:
            df_train_pred = ss.inverse_transform(df_train_pred_ss).add_prefix('pred_')

        # name = df_train.columns[0]
        # train_ss_results.loc[name] = [mean_squared_error(df_train.iloc[seq_len_src:, :][[features[0]]], df_train_pred_ss, squared=False), mean_absolute_error(df_train.iloc[seq_len_src:, :][[features[0]]], df_train_pred_ss)]


        # # true value df
        # if num_features_pred == 1:
        #     df_train_true = df.loc[train_start:train_end].iloc[seq_len_src:, :][[features[0]]]  # for univariate prediction
        # else:
        #     df_train_true = df.loc[train_start:train_end].iloc[seq_len_src:, :num_features_pred]  # for multivariate prediction


        # results = my_utils.calc_rmse_mae(df_train_true, df_train_pred)
        # display(results)

        # train_results.loc[name] = [mean_squared_error(df_train_true, df_train_pred, squared=False), mean_absolute_error(df_train_true, df_train_pred)]


        # print('-------')
        # print('predict valid')
        valid_pred = lstm_predict(model=model, data_loader=valid_loader_for_pred)

        if num_features_pred == 1:
            df_valid_pred_ss = pd.DataFrame(valid_pred, columns=[features[0]], index=df_valid.index).add_prefix('pred_')  # for multivariate prediction
        else:
            df_valid_pred_ss = pd.DataFrame(valid_pred, columns=df.columns[:num_features_pred], index=df_valid.index).add_prefix('pred_')  # for multivariate prediction

        if standardize_data:
            df_valid_pred = ss.inverse_transform(df_valid_pred_ss).add_prefix('pred_')

        # calculate rmse and mae with standardized data
        # results_ss = my_utils.calc_rmse_mae(df_valid, df_valid_pred_ss)
        # display(results_ss)

        # valid_ss_results.loc[name] = [mean_squared_error(df_valid[[features[0]]], df_valid_pred_ss, squared=False), mean_absolute_error(df_valid[[features[0]]], df_valid_pred_ss)]

        # true value df
        # if num_features_pred == 1:
        #     df_valid_true = df.loc[valid_start:valid_end][[features[0]]]  # for univariate prediction
        # else:
        #     df_valid_true = df.loc[valid_start:valid_end].iloc[:, :num_features_pred]  # for multivariate prediction

        # results = my_utils.calc_rmse_mae(df_valid_true, df_valid_pred)
        # display(results)

        # valid_results.loc[name] = [mean_squared_error(df_valid_true, df_valid_pred, squared=False), mean_absolute_error(df_valid_true, df_valid_pred)]



        # print('-------')
        # print('predict test')
        test_pred = lstm_predict(model=model, data_loader=test_loader_for_pred)

        if num_features_pred == 1:
            df_test_pred_ss = pd.DataFrame(test_pred, columns=[features[0]], index=df_test.index).add_prefix('pred_')  # for multivariate prediction
        else:
            df_test_pred_ss = pd.DataFrame(test_pred, columns=df.columns[:num_features_pred], index=df_test.index).add_prefix('pred_')  # for multivariate prediction

        if standardize_data:
            df_test_pred = ss.inverse_transform(df_test_pred_ss).add_prefix('pred_')

        # calculate rmse and mae with standardized data
        # results_ss = my_utils.calc_rmse_mae(df_test, df_test_pred_ss)
        # display(results_ss)

        # test_ss_results.loc[name] = [mean_squared_error(df_test[features[0]], df_test_pred_ss, squared=False), mean_absolute_error(df_test[features[0]], df_test_pred_ss)]


        # true value df
        # if num_features_pred == 1:
        #     df_test_true = df.loc[test_start:test_end][[features[0]]]  # for univariate prediction
        # else:
        #     df_test_true = df.loc[test_start:test_end].iloc[:, :num_features_pred]  # for multivariate prediction

        # results = my_utils.calc_rmse_mae(df_test_true, df_test_pred)
        # display(results)

        # test_results.loc[name] = [mean_squared_error(df_test_true, df_test_pred, squared=False), mean_absolute_error(df_test_true, df_test_pred)]

        
        df_pred_ss = pd.concat([df_train_pred_ss, df_valid_pred_ss, df_test_pred_ss])
        resid_df = pd.concat([resid_df,df_pred_ss])

    return df_train_pred_ss, df_valid_pred_ss, df_test_pred_ss

if __name__ == '__main__':
    # set model_num to predict
    # model_num_list = range(183, 193)
    model_num_list = list(range(176, 183)) + list(range(193, 196))
    get_prediction(raw_df=raw_df, model_type='lstm_seq2seq', model_num_list=model_num_list, result_path=)
