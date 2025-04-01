import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from dtaidistance.preprocessing import differencing
from tqdm.notebook import trange
import seaborn as sns
sns.set_theme()
import os
def differencing_custom(series, smooth=0.1):
    # differentiates the positions vectors to overrule the dtaidistance error
    return np.diff(series)

def test_car_following_by_time(data, id1, id2, tau, window, step):
    """
    DTW test if at a given window vehicle id2 is car following vehicle id1
    rool over the entire trajecory using a moving window (in the paper window size 30s stride 5s)

    inputs
    - data: trajectory dataset
    - id1 : leader id
    - id2 : follower id
    - tau : translation time
    - window : size of the moving window
    - step : size of the stride

    returns
    - time : array with all the ending points of all the windows
    - DTW : array containing for each window the Dynamic Time Warping value between the two trajectories over the given window
    """
    x1 = data[data['id'] == id1]
    x2 = data[(data['id'] == id2)]
    
    T = x2[x2['leader'] == id1]['time']
    DTW = []
    time = []
    k = min(T)
    while k < max(T):
        tmin_lead = int(k)
        tmax_lead = min(int(k + window), max(T))
        tmin_follow = k + tau
        tmax_follow = k + window + tau
        
        X1 = np.array(x1[(x1['time'] <= tmax_lead) & (x1['time'] >= tmin_lead)]['r'])
        X2 = np.array(x2[(x2['time'] <= tmax_follow) & (x2['time'] >= tmin_follow)]['r'])
        X1 = differencing_custom(X1, smooth=0.1)
        X2 = differencing_custom(X2, smooth=0.1)
    
        distance, paths = dtw.warping_paths(X1, X2)
        DTW.append(distance)
        time.append(k)
        k = k + step
    return time, DTW

def create_dtw_by_time_df(data,tau,window,step):
    """
    runs the test_car_following_by_time over the entire trajectory dataset
    
    inputs
    - data : trajectory dataset
    - tau : translation time
    - window : size of the moving window
    - step : size of the stride
    
    returns
    - df_out : dataframe with the"""
    time_list,id_list,leader_list,DTW_list = [],[],[],[]
    df_lead_follow_pairs = data[['leader', 'id']].drop_duplicates()
    df_lead_follow_pairs.dropna(inplace=True)
    df_lead_follow_pairs.reset_index(inplace=True)
    for k in range (len(df_lead_follow_pairs['id'])):
        leader = df_lead_follow_pairs['leader'][k]
        id = df_lead_follow_pairs['id'][k]
        time,DTW = test_car_following_by_time(data,leader,id,tau,window,step)
        id_list+= [id for k in range(len(time))]
        leader_list+= [leader for k in range(len(time))]
        time_list+=time
        DTW_list+=DTW
    df_out = pd.DataFrame({'id':id_list,'leader':leader_list,'time':time_list,'DTW':DTW_list})
    return df_out

