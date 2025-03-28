import pandas as pd
import numpy as np
import os

def compute_speed_diff(df):
    """
    computes speed delta between leader and follower for LC modelling
    
    input:
    - df: trajectory dataset
    
    returns
    - df_merged: dataset containing the speeddeltas"""
    df = df[['ID','time','speed-kf','lane-kf','acceleration-kf','leader',
             'Lane Change started','Lane Change occurred','Overtake',
             'type-most-common','ACC','DHW']]
    ACC_leaders = pd.unique(df[(df['ACC']=='Yes')|(df['ACC']=='yes')]['ID'])
    df['leader_is_ACC'] = df['leader'].isin(ACC_leaders)
    df_leader = df[['ID', 'time', 'speed-kf']].rename(columns={'ID': 'leader', 'speed-kf': 'speed-leader'})
    df_merged = pd.merge(df, df_leader, how='left', on=['leader', 'time'])
    df_merged['speed-delta'] = df_merged['speed-kf'] - df_merged['speed-leader']
    return df_merged

def construct_Logit_DF(trajs,dtw_df):
    """
    constructs the dataframe that will be used for the logit model
    
    inputs:
    - trajs : trajectory dataframe with DHW, THW, speedelta
    - dtw_df : the dataframe containing the results of the DTW car-following detection algorithm
    
    returns:
    
    - dtw_df : the dataframe containing the DHW, speeddelta, lane ADAS information and all other informations for the
    lane changing decision model"""
    trajs = compute_speed_diff(trajs)
    trajs = trajs.groupby(['ID','leader']).agg(
    mean_speed=('speed-kf', 'mean'),
    mean_acceleration=('acceleration-kf', 'mean'),
    is_large_vehicle=('type-most-common', lambda x: any(x == 'large-vehicle')),
    lane_kf=('lane-kf', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
    is_ACC=('ACC', lambda x: any(x == 'yes')),
    leader_is_ACC=('leader_is_ACC', lambda x: any(x == True)),
    mean_DHW=('DHW', 'mean'),
    mean_speed_delta=('speed-delta', lambda x: (x.diff().abs().mean())),
    lane_change = ('Lane Change started', 'last'),
    overtake = ('Overtake','last')
    ).reset_index()
    dtw_df = dtw_df.merge(trajs, left_on=['id','leader'], right_on=['ID','leader'], how='left')
    return dtw_df