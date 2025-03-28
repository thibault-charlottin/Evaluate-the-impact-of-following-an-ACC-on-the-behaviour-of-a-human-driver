import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
sns.set_theme()

def merge_df_trajs(path_list,glob_trajs):
    """merges the trajectories datasheets into one unique dataframe
    
    inputs
    - path_list : list of the folders, one folder=one experiment type
    - glob_trajs : path of the datasheets
    
    returns 
    - merge_df : dataframe containing all trajectories and indicators with a columns describint the 
    experiment they were taken from"""
    merge_df = pd.DataFrame()
    for p in path_list:
        path_traj = glob_trajs+p
        for f in os.listdir(path_traj):
            trajs_df = pd.read_csv(path_traj+f)
            ACC_ids = pd.unique(trajs_df[(trajs_df['ACC']=='Yes')|(trajs_df['ACC']=='yes')]['ID'])
            trajs_df['location']=[p for k in range(len(trajs_df['r']))]
            trajs_df['ACC'] = trajs_df['ID'].isin(ACC_ids)
            trajs_df['leader is ACC'] = (trajs_df['leader'].isin(ACC_ids)) & (~trajs_df['ID'].isin(ACC_ids))
            merge_df = pd.concat([merge_df,trajs_df])
        print(f'all merged for {p}')
    return merge_df


def compare_groups_statistics(traj_merged):
    """
    test if the mean DHW per speed range differs accros leader type
    
    input:
    - traj_merged : the dataframe containing all the speed deltas accelerations and corresponding leader type
    
    returns :
    - summary_df : a dataframe that contains all the tests results per speed range and leader type
    """
    traj_merged = traj_merged[(traj_merged['THW'] > 0) & (traj_merged['THW'] < 10)]
    traj_merged['speed_group'] = pd.cut(x=traj_merged['speed-kf'], 
                                        bins=[k * 5 for k in range(int(max(traj_merged['speed-kf']) / 5) + 1)])


    trajs_ACC_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] == '294_L1_by_run/')]

    trajs_AV_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] != '294_L1_by_run/')]

    trajs_other = traj_merged[(traj_merged['ACC'] == False) & (traj_merged['leader is ACC'] == False)]


    trajs_ACC_leader.dropna(inplace=True)
    trajs_AV_leader.dropna(inplace=True)
    trajs_other.dropna(inplace=True)

    trajs_ACC_leader.reset_index(inplace=True, drop=True)
    trajs_AV_leader.reset_index(inplace=True, drop=True)
    trajs_other.reset_index(inplace=True, drop=True)

    results = []

    speed_groups = traj_merged['speed_group'].cat.categories
    for speed_group in speed_groups:
        # Filtrer les données par groupe de vitesse
        data_ACC_leader = trajs_ACC_leader[trajs_ACC_leader['speed_group'] == speed_group]['DHW']
        data_AV_leader = trajs_AV_leader[trajs_AV_leader['speed_group'] == speed_group]['DHW']
        data_other = trajs_other[trajs_other['speed_group'] == speed_group]['DHW']
        samples_ACC_lead = len(data_ACC_leader)
        samples_AV_lead = len(data_AV_leader)
        samples_HDV_lead = len(data_other)
        
        if len(data_ACC_leader) > 1 and len(data_AV_leader) > 1:
            stat_mw_ACC_AV, p_mw_ACC_AV = mannwhitneyu(data_ACC_leader, data_AV_leader, alternative='two-sided')
            meandiff_ACC_AV = np.abs(np.mean(data_ACC_leader) - np.mean(data_AV_leader))
        else:
            stat_mw_ACC_AV, p_mw_ACC_AV = None, None
            meandiff_ACC_AV = None

        if len(data_other) > 1 and len(data_ACC_leader) > 1:
            stat_mw_ACC_HDV, p_mw_ACC_HDV = mannwhitneyu(data_other, data_ACC_leader, alternative='two-sided')
            meandiff_ACC_HDV = np.abs(np.mean(data_ACC_leader) - np.mean(data_other))
        else:
            stat_mw_ACC_HDV, p_mw_ACC_HDV = None, None
            meandiff_ACC_HDV = None

        if len(data_other) > 1 and len(data_AV_leader) > 1:
            stat_mw_HDV_AV, p_mw_HDV_AV = mannwhitneyu(data_other, data_AV_leader, alternative='two-sided')
            meandiff_HDV_AV = np.abs(np.mean(data_other) - np.mean(data_AV_leader))
        else:
            stat_mw_HDV_AV, p_mw_HDV_AV = None, None
            meandiff_HDV_AV = None

        results.append({
            'speed_range': str(speed_group),
            'p_value_mw_ACC_AV': p_mw_ACC_AV,
            'stat_mw_ACC_AV': stat_mw_ACC_AV,
            'meandiff_ACC_AV': meandiff_ACC_AV,
            'p_value_mw_ACC_HDV': p_mw_ACC_HDV,
            'stat_mw_ACC_HDV': stat_mw_ACC_HDV,
            'meandiff_ACC_HDV': meandiff_ACC_HDV,
            'p_value_mw_HDV_AV': p_mw_HDV_AV,
            'stat_mw_HDV_AV': stat_mw_HDV_AV,
            'meandiff_HDV_AV': meandiff_HDV_AV,
            'samples ACC': samples_ACC_lead ,
            'samples AV' : samples_AV_lead,
            'samples HDV' : samples_HDV_lead
        })

    results_df = pd.DataFrame(results)

    return results_df


def plot_speed_DHW(traj_merged):
    traj_merged = traj_merged[traj_merged['DHW']< 300]
    traj_merged['speed_group'] = pd.cut(x=traj_merged['speed-kf'], 
                                        bins=[k * 5 for k in range(int(max(traj_merged['speed-kf']) / 5) + 1)])

    trajs_ACC_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] == '294_L1_by_run/')]

    trajs_AV_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] != '294_L1_by_run/')]

    trajs_other = traj_merged[(traj_merged['ACC'] == False) & (traj_merged['leader is ACC'] == False)]

    trajs_ACC_leader.dropna(inplace=True)
    trajs_AV_leader.dropna(inplace=True)
    trajs_other.dropna(inplace=True)

    trajs_ACC_leader.reset_index(inplace=True, drop=True)
    trajs_AV_leader.reset_index(inplace=True, drop=True)
    trajs_other.reset_index(inplace=True, drop=True)

    trajs_other['Group'] = 'HDV following HDV'
    trajs_ACC_leader['Group'] = 'HDV following ACC'
    trajs_AV_leader['Group'] = 'HDV following AV'

    combined_data = pd.concat([trajs_other, trajs_ACC_leader, trajs_AV_leader], ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_data, x='speed_group', y='DHW', hue='Group',palette = ['#1F77B4','#FF7F0E','#2CA02C'], dodge=True)

    plt.xlabel('Speed Group', size=16)
    plt.ylabel('DHW', size=16)
    plt.title('DHW by Speed Group and Leader Type', size=18)
    plt.legend(title='Group', loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_stat_test_results(results_df):
    # Créer les subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))

    # P-value du test statistique pour chaque intervalle de vitesse et chaque test
    results_df.plot(x='speed_range', y=['p_value_mw_ACC_AV', 'p_value_mw_ACC_HDV', 'p_value_mw_HDV_AV'], kind='scatter', marker='o', ax=axes[0])
    axes[0].set_ylabel('P-value')
    axes[0].legend(title='Test')

    # Différence de moyenne pour chaque intervalle de vitesse pour chaque test
    results_df.plot(x='speed_range', y=['meandiff_ACC_AV', 'meandiff_ACC_HDV', 'meandiff_HDV_AV'], kind='scatter', marker='o', ax=axes[1])
    axes[1].set_ylabel('DHW mean difference')
    axes[1].legend(title='Test')

    # Barplot par intervalle qui donne la taille des échantillons (samples) par speed range
    results_df.plot(x='speed_range', y=['samples ACC', 'samples AV', 'samples HDV'], kind='bar', ax=axes[2])
    axes[2].set_ylabel('Number of samples')
    axes[2].legend(title='Groupe')

    plt.tight_layout()
    plt.show()

