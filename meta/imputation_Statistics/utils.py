# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math as mt
import random
from functools import partial
from sklearn import metrics
from sklearn.impute import KNNImputer
from statsmodels.tsa.seasonal import STL
from scipy.signal import savgol_filter
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset, is_subperiod,is_superperiod
# metodos de imputação
# def metodos(data,n):
#     _METHODS = {2: partial(pd.DataFrame.interpolate, method ='linear'),  
#                 3: partial(pd.DataFrame.interpolate, method = 'cubic'),
#                 4: partial(pd.DataFrame.interpolate, method = 'akima'),
#                 5: partial(pd.DataFrame.interpolate, method = 'polynomial', order = 5),
#                 6:partial(pd.DataFrame.interpolate, method = 'spline', order = 5),         
#                 7:partial(mediamovel, janela=3,minimo = 1),
#                 8:partial(pd.DataFrame.ffill, method = 'backfill'),
#                 9:partial(pd.DataFrame.ffill, method = 'ffill')
#                 }
#     if n == 0:
#         return data.fillna(data.mean())
#     if n == 1:
#         return data.fillna(data.median())
#     else:
#         return pd.DataFrame(_METHODS[n](data))
    
def metodos(data, n):
    _METHODS = {2: partial(pd.DataFrame.interpolate, method ='linear'),  
                3: partial(pd.DataFrame.interpolate, method = 'cubic'),
                4: partial(pd.DataFrame.interpolate, method = 'akima'),
                5: partial(pd.DataFrame.interpolate, method = 'polynomial', order = 5),
                6: partial(pd.DataFrame.interpolate, method = 'spline', order = 5),         
                7: partial(mediamovel, janela=3, minimo=1),
                8: lambda x: x.ffill(),  # usar ffill para substituição
                9: lambda x: x.bfill()}  # usar bfill para substituição
    
    if n == 0:
        return data.fillna(data.mean())
    if n == 1:
        return data.fillna(data.median())
    else:
        return pd.DataFrame(_METHODS[n](data))


def mediamovel(dado, janela , minimo = 1):
    return dado.rolling(window = janela, min_periods = minimo).mean()


def remover_sazonalidade_tendencia_stm(serie_temporal):

    serie_temporal = serie_temporal.squeeze()
    decomposicao =  STL(serie_temporal).fit()
    tendencia = decomposicao.trend.values
    sazonalidade = decomposicao.seasonal.values
    serie_sem_tendencia_sazonalidade = serie_temporal - tendencia - sazonalidade
    
    return serie_sem_tendencia_sazonalidade.to_frame(), tendencia, sazonalidade



def criar_janela(data, tamanho, padding):
    
      try:  
        if data.shape[0] == padding: return list([data])
        
        lista = [ ]
        for i in range(0,data.shape[0]-1,padding): lista.append(data.iloc[i:i+tamanho])
        return  lista
    
      except:
          
          if data.shape[0] == padding: return list([data])
          
          lista = [ ]
          for i in range(0,data.shape[0]-1,padding): lista.append(data[i:i+tamanho])
          return  lista
        

def knn (df,n_neighbors):
    imputer = KNNImputer(n_neighbors= n_neighbors, weights="uniform")
    # fit on the dataset
    imputer.fit(df)
    # transform the dataset
    Xtransf = imputer.transform(df)
    return Xtransf

def choice(a, size, replace = False):
    l = []

    while len(l) < size:
        v = random.randint(0, a - 1)
        # print(v)
        if replace:
            l.append(v)
        else:
            if v not in l:
                l.append(v)    
    
    return l


#metricas adotadas 

def Smape(A, F):
    tmp = 1* np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    if len_ == 0 and np.nansum(tmp) == 0: # Deals with a special case
        return 1
    return (1 / len(A) )* np.nansum(tmp)

def metricas(real, predicts, index=None):
    if not (index is None):

        #converter para uniformizar tratamento
        if type(real) is pd.core.frame.DataFrame:
            real = pd.Series(real.iloc[:,0])
        if type(predicts) is pd.core.frame.DataFrame:
            predicts = pd.Series(predicts.iloc[:,0])
        if type(index) is int:
            index = list([index])
    
        a = pd.Series(real[index].values)
        b = pd.Series(predicts[index].values)

        if b.isnull().sum()>0:#politica adotada para valores não imputados erro de 100%
            for i in range(len(b)):
                if mt.isnan(b[i]):
                    b[i]=b[i-1]
                if mt.isnan(b[i]):
                    b[i]=b[i+1]
                if mt.isnan(b[i]):
                    b[i]= 0  
        mae = metrics.mean_absolute_error(a, b)
        rmse = metrics.mean_squared_error(a,b)
        mape = metrics.mean_absolute_percentage_error(a,b)
        smape = Smape(a, b)
    # print('rmse-mae',rmse,mae) 
    else:
        a = real
        b = predicts
        mae = metrics.mean_absolute_error(a, b)
        smape = Smape(a, b)

    return [ round(mae,4),round(smape,8)]


def index_time_delta(td):
    
    def time_delta_(td):
        i = np.array([td.index.year[0],td.index.month[0],td.index.day[0],td.index.hour[0],td.index.minute[0],td.index.second[0],td.index.microsecond[0]])
        ii =np.array([td.index.year[1], td.index.month[1],td.index.day[1],td.index.hour[1],td.index.minute[1],td.index.second[1],td.index.microsecond[1]]) 
        conv = np.array([365.25, 30.4375, 1,0.041667,0.00069444,0.0000115741,1.1574074074074074e-11])
        return np.sum((ii-i)*conv).round(7)
    
    delta = time_delta_(td)
        
    return np.ones(td.index.shape[0])*delta 

def locf(data, axis=0):
    """ Last Observation Carried Forward

    For each set of missing indices, use the value of one row before(same
    column). In the case that the missing value is the first row, look one
    row ahead instead. If this next row is also NaN, look to the next row.
    Repeat until you find a row in this column that's not NaN. All the rows
    before will be filled with this value.

    Parameters
    ----------
    data: numpy.ndarray
        Data to impute.
    axis: boolean (optional)
        0 if time series is in row format (Ex. data[0][:] is 1st data point).
        1 if time series is in col format (Ex. data[:][0] is 1st data point).

    Returns
    -------
    numpy.ndarray
        Imputed data.

    """
    data = data.to_numpy()
    if axis == 0:
        data = np.transpose(data)
    elif axis == 1:
        pass
    else:
        raise "Error: Axis value is invalid, please use either 0 (row format) or 1 (column format)"

    nan_xy =np.argwhere(pd.isna(data))
    for x_i, y_i in nan_xy:
        # Simplest scenario, look one row back
        if x_i-1 > -1:
            data[x_i][y_i] = data[x_i-1][y_i]
        # Look n rows forward
        else:
            x_residuals = np.shape(data)[0]-x_i-1  # n datapoints left
            val_found = False
            for i in range(1, x_residuals):
                if not np.isnan(data[x_i+i][y_i]):
                    val_found = True
                    break
            if val_found:
                # pylint: disable=undefined-loop-variable
                for x_nan in range(i):
                    data[x_i+x_nan][y_i] = data[x_i+i][y_i]
            else:
                raise Exception("Error: Entire Column is NaN")
    return pd.DataFrame(data)

def get_freq(df):
   ''' N: nanossegundos U: microssegundos L: milissegundos S: segundos  T: minutos
    H: horas  D: dias W: semanas MS: meses Q: trimestres YS: anos '''
   return  pd.infer_freq(df.index)
   

def stl_decomposition(data,loess_span=13, robustness_iter=0):
    seasonal_period=7
    # Passo 1: Decomposição sazonal
    detrended = data - savgol_filter(data, window_length=seasonal_period, polyorder=1)
    
    # Passo 2: Decomposição de tendência
    for i in range(robustness_iter):
        # Estimação da tendência usando LOESS (Local Regression)
        trend = savgol_filter(detrended, window_length=loess_span, polyorder=1)
        
        # Remoção da tendência
        detrended = detrended - trend
        
    # Passo 3: Decomposição residual
    seasonal = savgol_filter(data - trend, window_length=seasonal_period, polyorder=1)
    residual = data - trend - seasonal
    
    return trend, seasonal, residual


mod = lambda x: x if x.is_integer() else np.inf

