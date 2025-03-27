import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu,levene,ks_2samp
sns.set_theme()
sns.set(font_scale=1.2)


def merge_df_trajs(path_list,glob_trajs):
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



def test_impact_follow_ACC(traj_merged,max_speed,tested_var):
    if tested_var =='TTC' :
        traj_merged = traj_merged[(traj_merged['TTC']>0)&(traj_merged['TTC']<10)]
        label = 'TTC [s]'
    else:
        traj_merged = traj_merged[(traj_merged['DRAC']>0)]
        label = 'DRAC [m/s²]'
    trajs_ACC_leader = traj_merged[(traj_merged['ACC']==False)&(traj_merged['leader is ACC']==True)&(traj_merged['location']=='294_L1_by_run/')]
    trajs_AV_leader = traj_merged[(traj_merged['ACC']==False)&(traj_merged['leader is ACC']==True)&(traj_merged['location']!='294_L1_by_run/')]
    trajs_other =  traj_merged[(traj_merged['ACC']==False)&(traj_merged['leader is ACC']==False)]
    trajs_ACC_leader_cong =trajs_ACC_leader[trajs_ACC_leader['speed-kf']<max_speed]
    trajs_AV_leader_cong =trajs_AV_leader[trajs_AV_leader['speed-kf']<max_speed]
    trajs_other_cong =trajs_other[trajs_other['speed-kf']<max_speed]
    trajs_ACC_leader_FF =trajs_ACC_leader[trajs_ACC_leader['speed-kf']>=max_speed]
    trajs_AV_leader_FF =trajs_AV_leader[trajs_AV_leader['speed-kf']>=max_speed]
    trajs_other_FF =trajs_other[trajs_other['speed-kf']>=max_speed]
    if tested_var =='TTC' :
        print(len(trajs_ACC_leader_cong[trajs_ACC_leader_cong['TTC']<1.5]['TTC']),len(trajs_ACC_leader_cong['TTC']))
        print(len(trajs_other_cong[trajs_other_cong['TTC']<1.5]['TTC']),len(trajs_other_cong['TTC']))

    trajs_ACC_leader_cong.dropna(inplace=True)
    trajs_AV_leader_cong.dropna(inplace=True)
    trajs_other_cong.dropna(inplace=True)
    trajs_ACC_leader_FF.dropna(inplace=True)
    trajs_AV_leader_FF.dropna(inplace = True)
    trajs_other_FF.dropna(inplace=True)
    statistic, p_value = ks_2samp(trajs_ACC_leader_cong[tested_var],trajs_other_cong[tested_var])
    print(f'Congested {tested_var}', statistic, p_value,len(trajs_ACC_leader_cong['ACC'])) 
    statistic, p_value = ks_2samp(trajs_ACC_leader_cong[tested_var],trajs_AV_leader_cong[tested_var])
    print(f'Congested {tested_var} ACC v AV', statistic, p_value,len(trajs_AV_leader_cong['ACC']) )
    statistic, p_value = ks_2samp(trajs_AV_leader_cong[tested_var],trajs_other_cong[tested_var])
    print(f'Congested {tested_var} AV', statistic, p_value,len(trajs_other_cong['ACC'])   )
    statistic, p_value = ks_2samp(trajs_ACC_leader_FF[tested_var],trajs_other_FF[tested_var])
    print(f'Free flow {tested_var}', statistic, p_value,len(trajs_ACC_leader_FF['ACC'])  )
    #statistic, p_value = ks_2samp(trajs_ACC_leader_FF[tested_var],trajs_AV_leader_FF[tested_var])
    #print(f'Free flow {tested_var} ACC v AV', statistic, p_value,len(trajs_AV_leader_FF['ACC'])  )
    #statistic, p_value = ks_2samp(trajs_AV_leader_FF[tested_var],trajs_other_FF[tested_var])
    #print(f'Free flow {tested_var} AV', statistic, p_value,len(trajs_other_FF['ACC'])  )
    # Création des sous-figures
    plt.figure(figsize=(12, 6))  # Taille globale de la figure

    # Sous-figure 1 pour les données congestionnées
    plt.subplot(121)  # 1 ligne, 2 colonnes, première sous-figure
    sns.ecdfplot(data=trajs_ACC_leader_cong[trajs_ACC_leader_cong[tested_var] < 10], x=tested_var,color = 'orange', label='HDV following an ACC')
    sns.ecdfplot(data=trajs_AV_leader_cong[trajs_AV_leader_cong[tested_var] < 10], x=tested_var,color = 'green', label='HDV following an L2 AV')
    sns.ecdfplot(data=trajs_other_cong[trajs_other_cong[tested_var] < 10], x=tested_var,color = 'blue', label='HDV following another vehicle')
    plt.title('Congested runs', size = 18)
    plt.xlabel(label,size= 16)
    plt.ylabel('frequency', size = 16)
    print(len(trajs_ACC_leader_cong[trajs_ACC_leader_cong[tested_var] < 1.2]['TTC'])/len(trajs_ACC_leader_cong[trajs_ACC_leader_cong[tested_var] < 10]['TTC']))
    print(len(trajs_other_cong[trajs_other_cong[tested_var] < 1.2]['TTC'])/len(trajs_other_cong[trajs_other_cong[tested_var] < 10]['TTC']))

    # Sous-figure 2 pour les données en flux libre
    plt.subplot(122)  # 1 ligne, 2 colonnes, deuxième sous-figure
    sns.ecdfplot(data=trajs_ACC_leader_FF[trajs_ACC_leader_FF[tested_var] < 10], x=tested_var,color = 'orange', label='HDV following an ACC')
    #sns.ecdfplot(data=trajs_AV_leader_FF[trajs_AV_leader_FF[tested_var] < 10], x=tested_var,color = 'green', label='HDV following an L2 AV')
    sns.ecdfplot(data=trajs_other_FF[trajs_other_FF[tested_var] < 10], x=tested_var,color = 'blue', label='HDV following another vehicle')
    plt.title('Free flow runs', size = 18)

    # Configurer une légende commune pour les deux sous-figures
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.xlabel(label,size= 16)
    plt.ylabel('frequency', size = 16)
    # Ajuster la mise en page
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust rect to make space for the legend

    # Sauvegarder la figure au format PDF
    plt.savefig(f'out/Images/{tested_var}_trajs_distribs_cong.pdf', bbox_inches='tight')
    plt.show()
    return
