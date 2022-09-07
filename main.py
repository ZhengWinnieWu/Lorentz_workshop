#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %      
% Main                                                                  %
%                                                                       %
% --------------------------------------------------------------------- %
%                                                                       %
% Created on Wed Sep  7 16:03:51 2022                                   %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, TensorBoard)
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import categorical_crossentropy

def HoA_database(root_data='./data/', root_results='./results/'):
    ####
    # X (Input data)
    SYY = 1980   # start year, could be changed
    EYY = 2021   # end year, could be changed

    # Predictor data preprocessing
    # can select the values and region you want by changing the parameters
    file_vars = ['ERA5_t2m', 'era5_t_850hpa', 'era5_z_200hpa', 'era5_z_500hpa', 'sst', 'era5_olr']
    header_vars = ['t2m', 't', 'z', 'z', 'sst', 'olr-mean']
    
    # select regions for the individual predictor
    lon_slices = [[-16,54],[-30,90],[-30,90],[-30,90],[-180,180],[40,180]]
    lat_slices = [[16,0],[30,-20],[-20,30],[-20,30],[40,-20],[-20,20]]
    
    for file_var, header_var, lon_slice, lat_slice in zip (file_vars, header_vars, lon_slices, lat_slices):
        if file_var=='era5_olr':
            file = xr.open_mfdataset(root_data+file_var+'_1950_2021_daily_1deg_tropics.nc',
                                     combine='by_coords',parallel=True)
            # print('olr')
            var_dim = file.sel(lon=slice(lon_slice[0],lon_slice[1]),lat=slice(lat_slice[0],lat_slice[1]))
        else:
            file = xr.open_mfdataset(root_data+file_var+'_1959-2021_1_12_daily_2.0deg.nc',
                                     combine='by_coords',parallel=True)
            # print(header_var)
            var_dim = file.sel(longitude=slice(lon_slice[0],lon_slice[1]),latitude=slice(lat_slice[0],lat_slice[1]))
        
        var_series = var_dim.sel(time=var_dim.time.dt.year.isin([np.arange(SYY,EYY+1)])).rolling(time=7, center=False).mean(skipna=True)
        var_anom_series = var_series.groupby("time.dayofyear") - var_series.groupby("time.dayofyear").mean("time",skipna=True)
        
        # save the data for future use
        np.save(root_results+'series_var_'+file_var+'_.npy',var_anom_series[header_var])
    ####
    
    ####
    # y (Target)
    file = root_data+'era5_hoa_dry_mask_2deg.nc' #0.25    
    mask=xr.open_mfdataset(root_data+file,combine='by_coords',parallel=True)
    mask_nan=mask.where(mask==1) #keep the values==1 and mask the rest
    mask_nan.tp.plot()
    mask_nan.sizes

    # Calculate the spatial mean of the tp file after applying the spatial mask
    file=xr.open_mfdataset(root_data+f'/era5_tp_1959-2021_1_12_daily_2.0deg.nc',
                              combine='by_coords',parallel=True)
    tp_dim=file.sel(longitude=slice(10,70),latitude=slice(24,-30))
    tp_series=np.multiply(mask_nan,tp_dim).mean(dim='latitude',skipna=True).mean(dim='longitude',skipna=True)
    
    # Calculate 33 percentile
    # Create daily values equal to a 31 day rolling. Select OND 2000-2020 data to decide the quantile threshold
    tp_rol=tp_series.rolling(time=31, center=True).mean().sel(time=tp_series.time.dt.year.isin([np.arange(2000,2021)]))
    tp_quantile=tp_rol.sel(time=tp_rol.time.dt.month.isin([10,11,12])).quantile(0.33)
    print('value of the 33 percentile',tp_quantile.tp.values)
    
    # Create index time series
    # Replace the values bellow the 33 percentile with 1 and the rest with zeros
    SYY = 1980   # start year, could be changed
    EYY = 2021   # end year, could be changed
    tp_rol = tp_series.rolling(time=31, center=True).mean().sel(time=tp_series.time.dt.month.isin([10,11,12]))
    tp_rol_sel = tp_rol.sel(time = slice(str(SYY),str(EYY)))
    tp_index = tp_rol_sel < tp_quantile
    tp_index = tp_index.astype(int)
    print('tp index',tp_index)
    
    # Select from the index time series the period for the target values (predictant) 
    # Oct 16 to Dec 16 for the period 1980-2020 (each day corresponds to a 31-day rolling mean)
    # The chosen time period could be changed to any time period you want
    for iyr in range(SYY,EYY+1):
        if iyr == SYY:
            tp_target = tp_index.sel(time = slice(str(iyr)+'-10-16',str(iyr)+'-12-16'))
        else:
            tp_target = xr.concat([tp_target,tp_index.sel(time = slice(str(iyr)+'-10-16',str(iyr)+'-12-16'))], dim='time')
    print('number of 0 and 1: ',np.unique(tp_target['tp'],return_counts=True))
    print(tp_target)
    # plt.plot(tp_target.tp)
    
    # Make it into a numpy array
    target = tp_target['tp'].values
    print(target.shape)
    ####
    
def get_train_test_val(data_predictor, data_target, test_frac, val_frac):
    """Splits data across periods into train, test, and validation"""
    # assign the last int(-test_frac*len(tp_predictor)) rows to test data
    test_predictor = data_predictor[int(-test_frac*len(data_target)):]
    test_target = data_target[int(-test_frac*len(data_target)):]
    
    # assign the last int(-test_frac*len(tp_predictor)) from the remaining rows to validation data
    remain_predictor = data_predictor[0:int(-test_frac*len(data_target))]
    remain_target = data_target[0:int(-test_frac*len(data_target))]
    val_predictor = remain_predictor[int(-val_frac*len(remain_predictor)):]
    val_target = remain_target[int(-val_frac*len(remain_predictor)):]
    
    # the remaining rows are assigned to train data
    train_predictor = remain_predictor[:int(-val_frac*len(remain_predictor))]
    train_target = remain_target[:int(-val_frac*len(remain_predictor))]
    return train_predictor, train_target, test_predictor, test_target, val_predictor, val_target

if __name__ == '__main__':
    HoA_database()
    
    """
    # define input and output data for LSTM
    y_all = keras.utils.to_categorical(target)
    X_all = pc_predictor
    print(X_all.shape,y_all.shape)
    
    train_X, train_y, test_X, test_y, val_X, val_y = get_train_test_val(X_all, y_all, test_frac=0.2, val_frac=0.2)
    
    # LSTM with attention layer
    ntimestep = 60    # number of time step used in the predictors
    nfeature = 30   # number of features
    input_tensor = Input(shape=(ntimestep,nfeature))
    output_tensor = ...(input_tensor)
    
    model = Model(input_tensor, output_tensor)
    plot_model(m, to_file='model.png', show_shapes=True)
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['acc'])
    """

    