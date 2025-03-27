import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

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


def plot_response_vs_stimulus(traj_merged):
    """
    plot the boxplot of the accelerations per speed delta range and leader type
    
    input:
    - traj_merged : the dataframe containing all the speed deltas accelerations and corresponding leader type
    
    returns :
    None (plots the response)"""
    min_speedelta = traj_merged['speeddelta'].min()
    max_speedelta = traj_merged['speeddelta'].max()

    # Create bins of size 5 that cover all values of speeddelta
    bins = [k * 5 for k in range(int(min_speedelta / 5), int(max_speedelta / 5) + 2)]
    traj_merged = traj_merged[(traj_merged['acceleration-kf'] > -5) & (traj_merged['acceleration-kf'] < 10)]
    traj_merged['delta_speed_group'] = pd.cut(traj_merged['speeddelta'], bins=bins)

    trajs_ACC_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] == '294_L1_by_run/')]

    trajs_AV_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] != '294_L1_by_run/')]

    trajs_other = traj_merged[(traj_merged['ACC'] == False) & (traj_merged['leader is ACC'] == False)]

    trajs_ACC_leader.dropna(inplace=True, subset=['speeddelta', 'acceleration-kf'])
    trajs_AV_leader.dropna(inplace=True, subset=['speeddelta', 'acceleration-kf'])
    trajs_other.dropna(inplace=True, subset=['speeddelta', 'acceleration-kf'])

    trajs_ACC_leader.reset_index(inplace=True, drop=True)
    trajs_AV_leader.reset_index(inplace=True, drop=True)
    trajs_other.reset_index(inplace=True, drop=True)

    trajs_other['Group'] = 'HDV follower'
    trajs_ACC_leader['Group'] = 'ACC follower'
    trajs_AV_leader['Group'] = 'SAE L2 follower'

    combined_data = pd.concat([trajs_other, trajs_ACC_leader, trajs_AV_leader], ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_data, x='delta_speed_group', y='acceleration-kf', hue='Group',
                palette=['#1F77B4', '#FF7F0E', '#2CA02C'], dodge=True, whis=2.0)  # Increase whis value

    plt.xlabel('Speed difference Group [m/s]', size=16)
    plt.ylabel('Acceleration [m/s²]', size=16)
    plt.legend(title='Group', loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    

def summarize_data(traj_merged):

    min_speedelta = traj_merged['speeddelta'].min()
    max_speedelta = traj_merged['speeddelta'].max()

    # Create bins of size 5 that cover all values of speeddelta
    bins = [k * 5 for k in range(int(min_speedelta / 5), int(max_speedelta / 5) + 2)]
    traj_merged['delta_speed_group'] = pd.cut(traj_merged['speeddelta'], bins=bins)

    trajs_ACC_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] == '294_L1_by_run/')]

    trajs_AV_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] != '294_L1_by_run/')]

    trajs_other = traj_merged[(traj_merged['ACC'] == False) & (traj_merged['leader is ACC'] == False)]

    # Create a summary DataFrame
    summary_data = []

    for group, label in [(trajs_ACC_leader, 'HDV following ACC'),
                         (trajs_AV_leader, 'HDV following AV'),
                         (trajs_other, 'HDV following HDV')]:
        for speed_range, group_data in group.groupby('delta_speed_group'):
            summary_data.append({
                'Speed Range': speed_range,
                'Leader Type': label,
                'Sample Size': len(group_data)
            })

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def test_acceleration_differences(traj_merged):
    """
    test if the mmean accelerations per speed delta range differs accros leader type
    
    input:
    - traj_merged : the dataframe containing all the speed deltas accelerations and corresponding leader type
    
    returns :
    - summary_df : a dataframe that contains all the tests results per speed range and leader type"""
    min_speedelta = traj_merged['speeddelta'].min()
    max_speedelta = traj_merged['speeddelta'].max()

    # create speeddelata groups grouping per range of 5m/s
    bins = [k * 5 for k in range(int(min_speedelta / 5), int(max_speedelta / 5) + 2)]
    traj_merged['delta_speed_group'] = pd.cut(traj_merged['speeddelta'], bins=bins)

    trajs_ACC_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] == '294_L1_by_run/')]

    trajs_AV_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] != '294_L1_by_run/')]
    trajs_other = traj_merged[(traj_merged['ACC'] == False) & (traj_merged['leader is ACC'] == False)]

    trajs_ACC_leader.dropna(inplace=True, subset=['speeddelta', 'acceleration-kf'])
    trajs_AV_leader.dropna(inplace=True, subset=['speeddelta', 'acceleration-kf'])
    trajs_other.dropna(inplace=True, subset=['speeddelta', 'acceleration-kf'])

    trajs_ACC_leader.reset_index(inplace=True, drop=True)
    trajs_AV_leader.reset_index(inplace=True, drop=True)
    trajs_other.reset_index(inplace=True, drop=True)

    trajs_other['Group'] = 'HDV following HDV'
    trajs_ACC_leader['Group'] = 'HDV following ACC'
    trajs_AV_leader['Group'] = 'HDV following AV'

    combined_data = pd.concat([trajs_other, trajs_ACC_leader, trajs_AV_leader], ignore_index=True)

    # test for each speedelta group
    results = []
    for interval in combined_data['delta_speed_group'].cat.categories:
        group_HDV = combined_data[(combined_data['delta_speed_group'] == interval) &
                                  (combined_data['Group'] == 'HDV following HDV')]['acceleration-kf']
        group_ACC = combined_data[(combined_data['delta_speed_group'] == interval) &
                                  (combined_data['Group'] == 'HDV following ACC')]['acceleration-kf']
        group_AV = combined_data[(combined_data['delta_speed_group'] == interval) &
                                 (combined_data['Group'] == 'HDV following AV')]['acceleration-kf']

        # compare mean across groups
        comparisons = [
            ('HDV vs ACC', group_HDV, group_ACC),
            ('HDV vs AV', group_HDV, group_AV),
            ('ACC vs AV', group_ACC, group_AV)
        ]

        for name, group1, group2 in comparisons:
            if len(group1) > 1 and len(group2) > 1:
                t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
                results.append({
                    'speedelta_interval': interval,
                    'comparison': name,
                    't_statistic': t_stat,
                    'p_value': p_value
                })

    results_df = pd.DataFrame(results)
    return results_df




path_list = ['294_L1_by_run/','90_94_static_by_run/','90_94_moving_by_run/']
glob_trajs = 'data/by_run/'
merged = merge_df_trajs(path_list,glob_trajs)
merged.reset_index(inplace = True, drop = True)
results_df = test_acceleration_differences(merged)
results_df.to_csv('speedelta_test.csv')
plot_response_vs_stimulus(merged)
