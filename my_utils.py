import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# reorganzie columns
def reorganize_cols(df):
    # separate dummy and non-dummy column names
    nondummy_col_list = [df.columns[i] for i in range(df.shape[1]) if len(df.iloc[:, i].value_counts()) != 2]
    dummy_col_list = [col_name for col_name in df.columns if col_name not in nondummy_col_list]

    # reorganize df
    return pd.concat([df[nondummy_col_list], df[dummy_col_list]], axis=1), len(nondummy_col_list), len(dummy_col_list)

# split data helper function
def split_data(df, time_list, train_start, train_end, valid_start, valid_end, test_start, test_end):
    # split df into train, valid and test data
    df_train = df.loc[train_start:train_end].copy()
    df_valid = df.loc[valid_start:valid_end].copy()
    df_test = df.loc[test_start:test_end].copy()

    # choose hours to use
    if time_list != 'all':
        df_valid = df_valid.loc[df_valid.index.hour.isin(time_list)]
        df_test = df_test.loc[df_test.index.hour.isin(time_list)]

    # print the fraction of each dataset
    print("Train set fraction: {:.3f}%".format(len(df_train)/len(df)*100))
    print("Valid set fraction: {:.3f}%".format(len(df_valid)/len(df)*100))
    print("Test set fraction: {:.3f}%".format(len(df_test)/len(df)*100))

    return df_train, df_valid, df_test

# RMSE and MAE calculation helper function
def calc_rmse_mae(df_true, df_pred):
    rmse, mae = {}, {}
    for i in range(df_pred.shape[1]):
        rmse[df_true.columns[i][:-6]] = mean_squared_error(df_true.iloc[:,i], df_pred.iloc[:,i], squared=False)
        mae[df_true.columns[i][:-6]] = mean_absolute_error(df_true.iloc[:,i], df_pred.iloc[:,i])

    rmse = pd.Series(rmse)
    mae = pd.Series(mae)

    results = pd.DataFrame({'RMSE': rmse, 'MAE': mae})
    results.loc['Average'] = [np.mean(rmse), np.mean(mae)]

    return results

def plot_population(y_true, y_pred, title='Results', flag='plot', anomaly_datetime=None, anomaly_next_hours=None):
    # plot with plotly
    fig = make_subplots(rows=10,
                        cols=1,
                        horizontal_spacing=0.9,
                        subplot_titles=[name[:-6] for name in y_true.columns])

    for i in range(y_pred.shape[1]):
        fig.add_trace(go.Scatter(x=y_true.index, y=y_true.iloc[:, i],
                                 legendgroup='true', legendgrouptitle_text='True',
                                 name=y_true.columns[i],
                                 line_color='#636efa'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred.iloc[:, i],
                                 legendgroup='pred', legendgrouptitle_text='Predicted',
                                 name=y_pred.columns[i],
                                 line_color='#EF553B'), row=i+1, col=1)

        fig.update_xaxes(title='Date', showgrid=False, row=i+1, col=1)
        fig.update_yaxes(title='Population', showgrid=False, row=i+1, col=1)

    fig.update_layout(legend=dict(x=0.99,
                              y=0.99,
                              xanchor='right',
                              yanchor='top',
                              orientation='h',
                              ),
                      hovermode='x unified',
                      title=dict(text=title,font=dict(size=40))
                     )

    if flag == 'anomaly_detection':
        for t, next_t in zip(anomaly_datetime[i], anomaly_next_hours[i]):
            fig.add_vrect(
                        x0=t,
                        x1=next_t,
                        fillcolor='pink',
                        opacity=0.9,
                        line_width=0,
                        layer='below'
                        )

    fig.update_layout(
                     height=4000,
                     legend={
                        "xref": "container",
                        "yref": "container",
                    }
                    )
    fig.show()
    
    return fig

# my ss class used for standardize data
class MyStandardScaler:
    def __init__(self, train_start, train_end):
        self.ss = StandardScaler()
        self.vars = []
        self.train_start = train_start
        self.train_end = train_end

    def _adjust_df(self, df):
        # called before fitted
        if not len(self.vars):
            raise Exception('Exception: My Standard Scaler not fitted yet.')
        
        # adjust the number of cols of df
        if df.shape[1] < len(self.vars):
            dummy = pd.DataFrame(np.zeros((df.shape[0], len(self.vars) - df.shape[1])), columns=self.vars[df.shape[1]:], index=df.index)
            return pd.concat([df, dummy], axis=1)
        elif df.shape[1] > len(self.vars):
            return df.iloc[:, :len(self.vars)]
        else:
            return df

    def fit(self, df, use_train=True):
        self.vars = df.columns
        if use_train:
            self.ss.fit(df.loc[self.train_start:self.train_end])
        else:
            self.ss.fit(df)

    def transform(self, df):
        _val = self.ss.transform(self._adjust_df(df))
        return pd.DataFrame(_val, columns=self.vars, index=df.index)
    
    def inverse_transform(self, df):
        _val = self.ss.inverse_transform(self._adjust_df(df))
        return pd.DataFrame(_val, columns=self.vars, index=df.index).iloc[:, :df.shape[1]]
