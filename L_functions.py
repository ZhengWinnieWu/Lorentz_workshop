import numpy as np
import xarray as xr
from datetime import datetime 
from datetime import timedelta
from datetime import date
import time

def sel_train_data_lead(nc_in_files,var_name,dim_to_stack,target_len,
                        s_target_date,e_target_date,
                        rw_1,lead_time,rw,ntimestep):
    '''
    This function inputs multiple nc files with the same dimension 
    (e.g., nc_in_files=path+'PC_serie*.nc') and reads them as a xarray. From these files
    a predictor array will be created. The dimension to stack the files must be given(e.g.,
    dim_to_stack='mode') and the variable name of the files (e.g., var_name='pcs').
    The length of the target must be given (target_len).
    The start date and end date that we want to predict must be given 
    (e.g., s_target_date='16-10-1980', e_target_date='16-12-2021') and
    the running window that was already applied on the predictors with center=False must be
    declared (rw_1). 
    
    The predictor is selected in a way so that the 
    needed date is predicted at a certain lead time (lead_time) and for a specific running
    window that was applied on the target (rw). Moreover, a selected time step for the LSTM 
    is considered (ntimestep).
    '''

    SDD = int(s_target_date[0:2])
    SMM = int(s_target_date[3:5])
    SYY=int(s_target_date[6:10])
    print('start target date',SDD,SMM,SYY)

    EDD = int(e_target_date[0:2])
    EMM = int(e_target_date[3:5])
    EYY = int(e_target_date[6:10])
    print('end target',EDD,EMM,EYY)

    half_rw = int(rw/2)

    # Import predictors if the predictors have been saved before
    pc_xr = xr.open_mfdataset(root_results+nc_in_files,
                              concat_dim=dim_to_stack,combine="nested")# 

    date_target = datetime.strftime(datetime(year=SYY,month=SMM,day=SDD), "%Y.%m.%d")
    pc_predictor = np.ndarray((target_len,ntimestep,int(pc_xr[dim_to_stack].shape[0])))
    it = 0
    ii = 0
    YYY = SYY
    while YYY < EYY+1:
        date_start = datetime.strftime(datetime.strptime(date_target, "%Y.%m.%d")-    timedelta(days=half_rw+lead_time+rw_1+ntimestep-1),"%Y.%m.%d")
        date_end = datetime.strftime(datetime.strptime(date_target, "%Y.%m.%d")-    timedelta(days=half_rw+lead_time+rw_1),"%Y.%m.%d")
        #print(date_target,date_start,date_end,it)
        #pc_predictor[ii,:,:] = pc_xr.sel(time = slice(date_start,date_end)).values
        pc_predictor[ii,:,:] = pc_xr.sel(time = slice(date_start,date_end))[var_name].values
        if date_target == datetime.strftime(datetime(year=YYY,month=EMM,day=EDD),"%Y.%m.%d"):
            YYY = YYY+1
            date_target = datetime.strftime(datetime(year=YYY,month=SMM,day=SDD), "%Y.%m.%d")
            it = 0
            #print(YYY)
        else:
            it = 1
        ii = ii+1
        date_target = datetime.strftime(datetime.strptime(date_target, "%Y.%m.%d")+timedelta(days=it),"%Y.%m.%d") 

    print('pc_predictor_shape',pc_predictor.shape)
    return pc_predictor
    