import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

def compute_DV(df):
    '''Compute speeddelta with leader.
    
    input
    - df : the trajectoryd ataset
    
    returns
    - df_out : the dataframe with the speeddelta column added'''
    df_out = pd.DataFrame(columns=df.columns)
    df_out['speeddelta'] = []

    for t in pd.unique(df['time']):
        at_t = df[df['time'] == t]
        for lane in pd.unique(at_t['lane-kf']):
            lane_at_t = at_t[at_t['lane-kf'] == lane]
            lane_at_t.sort_values(by='r', ascending=False, inplace=True)
            lane_at_t = lane_at_t.reset_index(drop = True)
            Delta_speed = []
            for k in range(len(lane_at_t['ID'])):
                if lane_at_t['leader'][k] > 0:
                    lead_df = lane_at_t[lane_at_t['ID'] == lane_at_t['leader'][k]]
                    ID_df = lane_at_t[lane_at_t['ID'] == lane_at_t['ID'][k]]
                    Delta_speed.append(list(lead_df['speed-kf'])[0] - list(ID_df['speed-kf'])[0])
                else:
                    Delta_speed.append(np.nan)
            lane_at_t['speeddelta'] = Delta_speed
            df_out = pd.concat([df_out, lane_at_t])
    return df_out

def add_stimulus_info(file_path, path_out):
    '''Clean data for a specific file.'''
    df = pd.read_csv(file_path)
    out = compute_DV(df)
    out.to_csv(file_path, index=False)


