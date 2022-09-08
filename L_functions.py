import numpy as np
import xarray as xr
from datetime import datetime 
from datetime import timedelta
from datetime import date
import time

import numpy as np
import xarray as xr
from datetime import datetime 
from datetime import timedelta
from datetime import date
import time

import pandas as pd

def sel_train_data_lead(nc_in_file,target_len,
                        s_target_date,e_target_date,
                        rw_1,lead_time,rw,ntimestep):
    '''
    This function inputs a 2-D file.nc, reads it as a xarray and creates
    a predictor array. 1_D:time, 2_D:features
    
    The length of the target time series must be given (target_len).
    The start date and end date that we want to predict must be given 
    (e.g., s_target_date='16-10-1980', e_target_date='16-12-2021') and
    the running window that was already applied on the predictors with center=False must be
    declared (rw_1). 
    
    The predictor is selected in a way so that the 
    needed date is predicted at a certain lead time (lead_time) and for a specific running
    window that was applied on the target with center=True (rw). If center=False, then set rw=0.
    Moreover, a selected time step for the LSTM 
    is considered (ntimestep).
    '''
    
    print('starting')

    SDD = int(s_target_date[0:2])
    SMM = int(s_target_date[3:5])
    SYY=int(s_target_date[6:10])
    print('start target date',SDD,SMM,SYY)

    EDD = int(e_target_date[0:2])
    EMM = int(e_target_date[3:5])
    EYY = int(e_target_date[6:10])
    print('end target',EDD,EMM,EYY)

    half_rw = int(rw/2)
    
    # Create correctly formated datetime
    date_target = datetime.strftime(datetime(year=SYY,month=SMM,day=SDD), "%Y.%m.%d")
    
    # Initialize shape of the final predictor array
    
    pc_predictor = [] # np.ndarray((target_len,ntimestep,int(nc_in_file[var_name].shape[1])))
    time_list = []
    it = 0
    ii = 0
    YYY = SYY
    while YYY < EYY+1:
        if YYY not in [2005,2007,2018,2004,2006]:
            date_start = datetime.strftime(datetime.strptime(date_target, "%Y.%m.%d")-    timedelta(days=half_rw+lead_time+rw_1+ntimestep-1),"%Y.%m.%d")
            date_end = datetime.strftime(datetime.strptime(date_target, "%Y.%m.%d")-    timedelta(days=half_rw+lead_time+rw_1),"%Y.%m.%d")
            #print(date_target,date_start,date_end,it)
            f = nc_in_file.sel(time = slice(date_start,date_end))
            f=f.assign_coords(time=range(ntimestep))
            time_list.append(date_target)
            pc_predictor.append(f)
        if date_target == datetime.strftime(datetime(year=YYY,month=EMM,day=EDD),"%Y.%m.%d"):
            YYY = YYY+1
            date_target = datetime.strftime(datetime(year=YYY,month=SMM,day=SDD), "%Y.%m.%d")
            it = 0
            #print(YYY)
        else:
            it = 1
        ii = ii+1
        date_target = datetime.strftime(datetime.strptime(date_target, "%Y.%m.%d")+timedelta(days=it),"%Y.%m.%d") 
    pc_predictor = xr.concat(pc_predictor,"new_time").rename({"time":"lag"}).rename({"new_time":"time"})
    pc_predictor = pc_predictor.assign_coords(time=time_list)
    pc_predictor = pc_predictor.assign_coords(time=pd.DatetimeIndex(pc_predictor.time)) #-pd.Timedelta("15 d"))
    #print('pc_predictor_shape',pc_predictor.shape)
    return pc_predictor
    