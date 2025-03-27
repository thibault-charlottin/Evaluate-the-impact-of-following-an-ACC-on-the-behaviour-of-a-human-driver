import pandas as pd
import numpy as np
import os
import biogeme.biogeme as bio
from biogeme.expressions import Beta, Variable
from biogeme.models import loglogit
import biogeme.database as bdb

def prepare_data(path):
    # Load data
    gaps_logit = pd.read_csv(path)  # Use absolute paths
    large_ids = pd.unique(gaps_logit[gaps_logit['is_large_vehicle']==True]['ID'])
    gaps_logit['leader_is_large_vehicle'] = gaps_logit['leader'].isin(large_ids)
    gaps_logit['leader_is_large_vehicle'] = gaps_logit['leader_is_large_vehicle'].astype(int)
    gaps_logit['lane_kf'] = np.abs(gaps_logit['lane_kf'])
    gaps_logit.replace([-np.inf, np.inf], np.nan, inplace=True)
    gaps_logit.dropna(inplace=True) 
    condition_L1 = (gaps_logit['leader_is_ACC']) & (gaps_logit['location'] == '294_L1_by_run/')
    condition_L2 = (gaps_logit['leader_is_ACC']) & (gaps_logit['location'] != '294_L1_by_run/')
    gaps_logit['L1'] = np.where(condition_L1, True, False) 
    gaps_logit['L2'] = np.where(condition_L2, True, False) 
    gaps_logit['L1'] = gaps_logit['L1'].astype(int)
    gaps_logit['L2'] = gaps_logit['L2'].astype(int)
    gaps_logit['is_large_vehicle'] = gaps_logit['is_large_vehicle'].astype(int)
    gaps_logit['acceleration-kf'] = np.abs(gaps_logit['mean_acceleration'])
    gaps_logit['is_ACC'] = gaps_logit['is_ACC'].astype(int)
    gaps_logit['leader_is_ACC'] = gaps_logit['leader_is_ACC'].astype(int)
    gaps_logit['lane_kf'] = np.abs(gaps_logit['lane_kf'])
    gaps_logit['lane_kf'] = [k-9 if k>9 else k for k in gaps_logit['lane_kf']]
    gaps_logit['right_lane'] = (
        ((gaps_logit['lane_kf'] == 1) & ((gaps_logit['location'] == 2) | (gaps_logit['location'] == 3))) |
        ((gaps_logit['lane_kf'] == 2) & ((gaps_logit['location'] != 0) & (gaps_logit['location'] != 1)))
    ).astype(int)
    gaps_logit['left_lane'] = (
        ((gaps_logit['lane_kf'] == 6) & ((gaps_logit['location'] == 2) | (gaps_logit['location'] == 3))) |
        ((gaps_logit['lane_kf'] == 5) & ((gaps_logit['location'] != 0) & (gaps_logit['location'] != 1)))
    ).astype(int)
    gaps_logit['middle_lane'] = (
        (gaps_logit['right_lane'] == 0) & 
        (gaps_logit['left_lane'] == 0)
    ).astype(int)

    gaps_logit = gaps_logit[gaps_logit['is_large_vehicle']==False] #dropping trucks from the analysis

    gaps_logit.reset_index(inplace = True)
    gaps_logit['LC direction'] = [0 if gaps_logit['lane_change'][k]==True and  gaps_logit['overtake'][k]==False
                                else 1 if gaps_logit['lane_change'][k]==False 
                                else 2 for k in range (len(gaps_logit['lane_change']))]

    gaps_logit['location'] = [0 if gaps_logit['location'][k] == '294_L2_by_run/' else 1 if gaps_logit['location'][k] == '294_L1_by_run/' 
                            else 2 if gaps_logit['location'][k] == '90_94_static_by_run/' 
                            else 3  for k in range (len(gaps_logit['lane_change']))]
    gaps_logit['lane_change'] = gaps_logit['lane_change'].astype(int)
    gaps_logit['overtake'] = gaps_logit['overtake'].astype(int)
    gaps_logit = gaps_logit.groupby(['id','leader']).last() #keeping the last window
    return gaps_logit


def logit_model(gaps_logit):
    database = bdb.Database('gaps_Logit', gaps_logit)


    # Variables
    DTW = Variable('DTW')
    mean_DHW = Variable('mean_DHW')
    mean_speed_delta = Variable('mean_speed_delta')
    LC_direction = Variable('LC direction')
    L1 = Variable('L1')
    L2 = Variable('L2')
    left_lane = Variable('left_lane')
    right_lane = Variable('right_lane')

    # Coefficients to estimate
    beta_DTW = Beta('beta_DTW', 0, None, None, 0)
    beta_DHW = Beta('beta_DHW', 0, None, None, 0)
    beta_mean_speed_delta = Beta('beta_mean_speed_delta', 0, None, None, 0)

    beta_leader_L1_left = Beta('beta_leader_L1_left', 0, None, None, 0)
    beta_leader_L1_right = Beta('beta_leader_L1_right', 0, None, None, 0)
    beta_leader_L2_left = Beta('beta_leader_L2_left', 0, None, None, 0)
    beta_leader_L2_right = Beta('beta_leader_L2_right', 0, None, None, 0)
    beta_constant_left = Beta('beta_constant_left', 0, None, None, 0)
    beta_constant_right = Beta('beta_constant_right', 0, None, None, 0)

    # Ajouter les lanes dans les fonctions d'utilité
    V_left = (
        beta_constant_left +
        beta_DTW * DTW +
        beta_DHW * mean_DHW +
        beta_mean_speed_delta * mean_speed_delta +
        beta_leader_L2_left * L2 +
        beta_leader_L1_left * L1 
    )

    V_right = (
        beta_constant_right +
        beta_DTW * DTW +
        beta_DHW * mean_DHW +
        beta_mean_speed_delta * mean_speed_delta +
        beta_leader_L2_right * L2 +
        beta_leader_L1_right * L1 
    )

    V_stay = 0  #reference alternative
    V = {0: V_left, 1: V_stay, 2: V_right}

    avail = {
        0: 1,  
        1: 1,              
        2: 1   
    } 
    # Multinomial logit model
    model = loglogit(V, avail, LC_direction)

    # Biogeme estimation
    biogeme = bio.BIOGEME(database, model)
    biogeme.modelName = 'gaps_Logit_test'

    results = biogeme.estimate()
    return results
