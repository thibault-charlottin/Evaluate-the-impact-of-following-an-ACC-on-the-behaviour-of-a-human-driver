import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count

def slice_data(df_path, path_out):
    '''Slice the dataframe into separate CSV files based on the 'run_index' column.

    inputs:
     -df_path: path of the dataframe containing all the TGSIM runs for a given experiment
     path_out: path where we want to '''
    df = pd.read_csv(df_path)
    unique_runs = pd.unique(df['run_index'])
    os.makedirs(path_out, exist_ok=True)

    def process_run(run):
        out = df[df['run_index'] == run]
        out.to_csv(os.path.join(path_out, f'{run}.csv'), index=False)

    with Pool(cpu_count()) as pool:
        pool.map(process_run, unique_runs)

def detect_leader(df):
    '''Detects leaders for each vehicle in the dataframe.'''
    df_out = pd.DataFrame(columns=df.columns)
    df_out['leader'] = []
    df['r'] = np.sqrt(df['xloc_kf']**2 + df['yloc_kf']**2)
    df['theta'] = np.arctan2(df['yloc_kf'], df['xloc_kf'])
    lanes = pd.unique(df['lane_kf'])

    for t in pd.unique(df['time']):
        at_t = df[df['time'] == t]
        for l in lanes:
            lane_at_t = at_t[at_t['lane_kf'] == l]
            lane_at_t.sort_values(by='r', ascending=False, inplace=True)
            lane_at_t = lane_at_t.reset_index()
            leader = [np.nan]
            for k in range(len(lane_at_t['id']) - 1):
                leader.append(lane_at_t['id'][k])
            lane_at_t['leader'] = leader
            df_out = pd.concat([df_out, lane_at_t])
    return df_out

def compute_DHW(df):
    '''Compute Distance Headway (DHW) and Time Headway (THW) for each vehicle in the dataframe.'''
    df_out = pd.DataFrame(columns=df.columns)
    df_out['DHW'] = []
    df_out['THW'] = []

    for t in pd.unique(df['time']):
        at_t = df[df['time'] == t]
        for lane in pd.unique(at_t['lane_kf']):
            lane_at_t = at_t[at_t['lane_kf'] == lane]
            lane_at_t.sort_values(by='r', ascending=False, inplace=True)
            lane_at_t = lane_at_t.reset_index()
            DHW, THW = [], []
            for k in range(len(lane_at_t['id'])):
                if lane_at_t['leader'][k] > 0:
                    lead_df = lane_at_t[lane_at_t['id'] == lane_at_t['leader'][k]]
                    id_df = lane_at_t[lane_at_t['id'] == lane_at_t['id'][k]]
                    DHW.append(np.sqrt(
                        (list(lead_df['xloc_kf'])[0] - list(id_df['xloc_kf'])[0])**2 +
                        (list(lead_df['yloc_kf'])[0] - list(id_df['yloc_kf'])[0])**2))
                    THW.append(DHW[-1] / lane_at_t['speed_kf'][k])
                else:
                    DHW.append(np.nan)
                    THW.append(np.nan)
            lane_at_t['DHW'] = DHW
            lane_at_t['THW'] = THW
            df_out = pd.concat([df_out, lane_at_t])
    return df_out

def clean_run(run, df_path, path_out):
    '''Clean data for a specific run.'''
    df = pd.read_csv(df_path)
    out = df[df['run_index'] == run]
    out = detect_leader(out)
    out = compute_DHW(out)
    out.to_csv(os.path.join(path_out, f'{run}.csv'), index=False)

def clean_data(df_path, path_out):
    df = pd.read_csv(df_path)
    for run in df['run_index']:
        clean_run(run, df_path, path_out)