import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

def compute_TTC(df):
    '''Compute Time to collision (TTC) for each vehicle in the dataframe.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe containing vehicle data.

    Returns:
    --------
    DataFrame
        DataFrame with additional columns 'TTC' (time to collsion).
    '''
    df_out = pd.DataFrame(columns = df.columns)
    df_out['TTC'] = [] ; 
    for t in pd.unique(df['time']):
        at_t = df[df['time']==t]
        for lane in pd.unique(at_t['lane-kf']):
            lane_at_t = at_t[at_t['lane-kf']==lane]
            lane_at_t.sort_values(by = 'r', ascending = True, inplace = True)
            lane_at_t = lane_at_t.reset_index(drop=True)
            TTC= []
            for k in range(len(lane_at_t['ID'])):
                if lane_at_t['leader'][k]>0 :
                    lead_df = lane_at_t[lane_at_t['ID']==lane_at_t['leader'][k]]
                    lead_df.reset_index(inplace=True,drop=True)
                    ID_df = lane_at_t[lane_at_t['ID']==lane_at_t['ID'][k]]
                    ID_df.reset_index(inplace=True,drop=True)
                    if lead_df['speed-kf'][0]<ID_df['speed-kf'][0]:
                        TTC.append((ID_df['DHW'][0]-lead_df['length-smoothed'][0])/(ID_df['speed-kf'][0]-lead_df['speed-kf'][0]))
                    else: 
                        TTC.append(np.nan)
                else: 
                    TTC.append(np.nan)
            lane_at_t['TTC']=TTC
            df_out = pd.concat([df_out,lane_at_t])
    return df_out

def compute_DRAC(df):
    '''Compute Decceleration rate to avoid collision (DRAC) for each vehicle in the dataframe.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe containing vehicle data.

    Returns:
    --------
    DataFrame
        DataFrame with additional column 'DRAC' (Decceleration rate to avoid collision).
    '''
    df_out = pd.DataFrame(columns = df.columns)
    df_out['TTC'] = [] ; 
    for t in pd.unique(df['time']):
        at_t = df[df['time']==t]
        for lane in pd.unique(at_t['lane-kf']):
            lane_at_t = at_t[at_t['lane-kf']==lane]
            lane_at_t.sort_values(by = 'r', ascending = True, inplace = True)
            lane_at_t = lane_at_t.reset_index(drop=True)
            DRAC= []
            for k in range(len(lane_at_t['ID'])):
                if lane_at_t['leader'][k]>0 :
                    lead_df = lane_at_t[lane_at_t['ID']==lane_at_t['leader'][k]]
                    lead_df.reset_index(inplace=True,drop=True)
                    ID_df = lane_at_t[lane_at_t['ID']==lane_at_t['ID'][k]]
                    ID_df.reset_index(inplace=True,drop=True)
                    if lead_df['speed-kf'][0]<ID_df['speed-kf'][0]:
                        DRAC.append(((ID_df['speed-kf'][0]-lead_df['speed-kf'][0])**2)/(2*(ID_df['DHW'][0]-lead_df['length-smoothed'][0])))
                    else: 
                        DRAC.append(np.nan)
                else: 
                    DRAC.append(np.nan)
            lane_at_t['DRAC']=DRAC
            df_out = pd.concat([df_out,lane_at_t])
    return df_out


def process_file(data,file_path):
    """
    add surrogate safety indicators
    
    input:
    - data : trajecory dataframe
    - path_out : savepath of the updated trajectory dataframe with the TTC and DRAC computed
    
    returns 
    None"""
    try:
        data = compute_DRAC(data)
        data = compute_TTC(data)
        data.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

