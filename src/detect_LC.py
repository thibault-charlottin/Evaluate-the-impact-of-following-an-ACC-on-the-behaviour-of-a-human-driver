import pandas as pd
import numpy as np

def detect_lane_change(df):
    '''
    detects lane changing
    
    input
    - df : trajectory dataset

    returns
    - out : trajectory dataset with the flags LC detected, overtake and vehicle in its new lane
    '''
    out = pd.DataFrame(columns=df.columns)
    for k in pd.unique(df['id']):
        data = df[df['id'] == k]
        lane_change_started = [True if data['lane_kf'].iloc[i + 1] != data['lane_kf'].iloc[i] else False for i in range(len(data) - 1)]
        lane_change_started.append(False)  # Add False for the last row
        data['Lane Change started'] = lane_change_started
        
        lane_change_occurred = [True if data['lane_kf'].iloc[i - 1] != data['lane_kf'].iloc[i] else False for i in range(1, len(data))]
        lane_change_occurred.insert(0, False)  # Insert False for the first row
        data['Lane Change occurred'] = lane_change_occurred
        
        overtake = [True if np.abs(data['lane_kf'].iloc[i - 1]) > np.abs(data['lane_kf'].iloc[i]) else False for i in range(1, len(data))]
        overtake.insert(0, False)  # Insert False for the first row
        data['Overtake'] = overtake
        
        out = pd.concat([out, data])
    return out

def assign_type(row):
    if row['ACC']=='Yes':
        return 'ACC'
    elif row['ACC leader']:
        return 'ACC leader'
    else:
        return 'other vehicle'

def assign_type_postLC(row):
    if row['ACC']=='Yes':
        return 'ACC'
    elif row['new leader ACC']:
        return 'new leader is ACC'
    else:
        return 'other vehicle'
    

