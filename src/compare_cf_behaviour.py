import pandas as pd
import numpy as np
import os
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, levene
from scipy.stats import ks_2samp, kruskal
sns.set_theme()
import warnings
warnings.filterwarnings('ignore')

def define_CF(dtw_data,max_dtw):
    dtw_data['isCF'] = dtw_data['DTW'].apply(lambda x: True if x < max_dtw else False)
    return dtw_data

def merge_df_dtw(path_list, glob_dtw, glob_trajs):
    merge_df = pd.DataFrame()
    for p in path_list:
        path_dtw = os.path.join(glob_dtw, p)
        path_traj = os.path.join(glob_trajs, p)
        for f in os.listdir(path_dtw):
            df_dtw = pd.read_csv(os.path.join(path_dtw, f))
            trajs_df = pd.read_csv(os.path.join(path_traj, f))

            ACC_ids = pd.unique(trajs_df[(trajs_df['ACC'] == 'Yes') | (trajs_df['ACC'] == 'yes')]['ID'])

            df_dtw['run'] = f
            df_dtw['location'] = p
            df_dtw['ACC'] = df_dtw['id'].isin(ACC_ids)
            df_dtw['leader is ACC'] = (df_dtw['leader'].isin(ACC_ids)) & (~df_dtw['id'].isin(ACC_ids))

            # Calculer la vitesse moyenne spécifique à chaque ID sur la période time(dtw_df)[i], time(DTW_df[i]) + 30s
            mean_speeds = []
            for index, row in df_dtw.iterrows():
                id_value = row['id']
                start_time = row['time']
                end_time = start_time + 30  # Assurez-vous que 'time' est en secondes

                # Filtrer trajs_df pour l'ID et l'intervalle de temps
                filtered_trajs = trajs_df[(trajs_df['ID'] == id_value) &
                                          (trajs_df['time'] >= start_time) &
                                          (trajs_df['time'] <= end_time)]

                if not filtered_trajs.empty:
                    mean_speed = np.mean(filtered_trajs['speed-kf'])
                else:
                    mean_speed = np.nan  # Si aucune donnée n'est trouvée, utiliser NaN

                mean_speeds.append(mean_speed)

            df_dtw['mean speed'] = mean_speeds
            merge_df = pd.concat([merge_df, df_dtw])

        print(f'all merged for {f}')

    merge_df = define_CF(merge_df, 1.5)
    return merge_df




def plot_speed_DTW(traj_merged):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    print(len(traj_merged['DTW']))
    traj_merged = traj_merged[traj_merged['DTW']<=10]
    print(len(traj_merged['DTW']))
    # Filtrer les données
    traj_merged['speed_group'] = pd.cut(x=traj_merged['mean speed'], 
                                        bins=[k * 5 for k in range(int(max(traj_merged['mean speed']) / 5) + 1)])

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
    sns.boxplot(data=combined_data, x='speed_group', y='DTW', hue='Group',palette = ['blue','orange','green'], dodge=True)

    plt.xlabel('Speed Group', size=16)
    plt.ylabel('DTW', size=16)
    plt.title('DTW by Speed Group and Leader Type', size=18)
    plt.legend(title='Group', loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_CF_DTW(traj_merged):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Filtrer les données pour garder uniquement les lignes où DTW <= 10
    traj_merged = traj_merged[traj_merged['DTW'] <= 10]

    # Créer des groupes de vitesse
    traj_merged['speed_group'] = pd.cut(x=traj_merged['mean speed'],
                                        bins=[k * 5 for k in range(int(max(traj_merged['mean speed']) / 5) + 1)])

    # Filtrer les données pour les différents types de leaders
    trajs_ACC_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] == '294_L1_by_run/')]

    trajs_AV_leader = traj_merged[(traj_merged['ACC'] == False) &
                                   (traj_merged['leader is ACC'] == True) &
                                   (traj_merged['location'] != '294_L1_by_run/')]

    trajs_other = traj_merged[(traj_merged['ACC'] == False) & (traj_merged['leader is ACC'] == False)]

    # Supprimer les valeurs manquantes
    trajs_ACC_leader.dropna(inplace=True)
    trajs_AV_leader.dropna(inplace=True)
    trajs_other.dropna(inplace=True)

    # Réinitialiser les index
    trajs_ACC_leader.reset_index(inplace=True, drop=True)
    trajs_AV_leader.reset_index(inplace=True, drop=True)
    trajs_other.reset_index(inplace=True, drop=True)

    # Ajouter une colonne 'Group'
    trajs_other['Group'] = 'HDV following HDV'
    trajs_ACC_leader['Group'] = 'HDV following ACC'
    trajs_AV_leader['Group'] = 'HDV following AV'

    # Combiner les données
    combined_data = pd.concat([trajs_other, trajs_ACC_leader, trajs_AV_leader], ignore_index=True)

    # Filtrer les données pour garder uniquement les lignes où DTW < 1.5
    combined_data['DTW_less_than_1_5'] = combined_data['DTW'] < 1.5

    # Calculer le pourcentage de DTW < 1.5 par groupe de vitesse et type de leader
    percentage_data = combined_data.groupby(['speed_group', 'Group'])['DTW_less_than_1_5'].mean().reset_index()
    percentage_data['percentage'] = percentage_data['DTW_less_than_1_5'] * 100

    # Tracer le barplot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=percentage_data, x='speed_group', y='percentage', hue='Group', palette=['blue', 'orange', 'green'])

    plt.xlabel('Speed Group', size=16)
    plt.ylabel('Percentage vehicles car-following', size=16)
    plt.legend(title='Group', loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # Effectuer des tests de proportion pour chaque groupe de vitesse
    results = []
    for speed_group, group_data in combined_data.groupby('speed_group'):
        contingency_table = pd.crosstab(group_data['Group'], group_data['DTW_less_than_1_5'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Stocker les résultats
        result = {
            'speed_group': speed_group,
            'p-value': p,
            'stat': chi2,
            'significant': p < 0.05
        }

        if p < 0.05:
            # Déterminer l'ordre des proportions
            cf_proportions = percentage_data[percentage_data['speed_group'] == speed_group].set_index('Group')['percentage']
            ordered_groups = cf_proportions.sort_values(ascending=False).index.tolist()
            result['order'] = ordered_groups

        results.append(result)

    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame(results)
    
    return results_df


def compare_groups_statistics(traj_merged):

    traj_merged = traj_merged[(traj_merged['DTW'] > 0) & (traj_merged['DTW'] < 10)]
    traj_merged['speed_group'] = pd.cut(x=traj_merged['mean speed'], 
                                        bins=[k * 2.5 for k in range(int(max(traj_merged['mean speed']) / 2.5) + 1)])


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

    # Réinitialiser les index
    trajs_ACC_leader.reset_index(inplace=True, drop=True)
    trajs_AV_leader.reset_index(inplace=True, drop=True)
    trajs_other.reset_index(inplace=True, drop=True)

    # Initialiser la DataFrame pour stocker les résultats
    results = []

    # Comparer chaque plage de vitesse
    speed_groups = traj_merged['speed_group'].cat.categories
    for speed_group in speed_groups:
        # Filtrer les données par groupe de vitesse
        data_ACC_leader = trajs_ACC_leader[trajs_ACC_leader['speed_group'] == speed_group]['DTW']
        data_AV_leader = trajs_AV_leader[trajs_AV_leader['speed_group'] == speed_group]['DTW']
        data_other = trajs_other[trajs_other['speed_group'] == speed_group]['DTW']

        # Tests statistiques
        if len(data_ACC_leader) > 1 and len(data_AV_leader) > 1:
            stat_mw_ACC_AV, p_mw_ACC_AV = mannwhitneyu(data_ACC_leader, data_AV_leader, alternative='two-sided')
            stat_levene_ACC_AV, p_levene_ACC_AV = levene(data_ACC_leader, data_AV_leader)
        else:
            stat_mw_ACC_AV, p_mw_ACC_AV = None, None
            stat_levene_ACC_AV, p_levene_ACC_AV = None, None

        if len(data_other) > 1 and len(data_ACC_leader) > 1:
            stat_mw_ACC_HDV, p_mw_ACC_HDV = mannwhitneyu(data_other, data_ACC_leader, alternative='two-sided')
            stat_levene_ACC_HDV, p_levene_ACC_HDV = levene(data_other, data_ACC_leader)
        else:
            stat_mw_ACC_HDV, p_mw_ACC_HDV = None, None
            stat_levene_ACC_HDV, p_levene_ACC_HDV = None, None

        if len(data_other) > 1 and len(data_AV_leader) > 1:
            stat_mw_HDV_AV, p_mw_HDV_AV = mannwhitneyu(data_other, data_AV_leader, alternative='two-sided')
            stat_levene_HDV_AV, p_levene_HDV_AV = levene(data_other, data_AV_leader)
        else:
            stat_mw_HDV_AV, p_mw_HDV_AV = None, None
            stat_levene_HDV_AV, p_levene_HDV_AV = None, None

        results.append({
            'speed_range': str(speed_group),
            'p_value_mw_ACC_AV': p_mw_ACC_AV,
            'stat_mw_ACC_AV': stat_mw_ACC_AV,
            'p_value_levene_ACC_AV': p_levene_ACC_AV,
            'stat_levene_ACC_AV': stat_levene_ACC_AV,
            'p_value_mw_ACC_HDV': p_mw_ACC_HDV,
            'stat_mw_ACC_HDV': stat_mw_ACC_HDV,
            'p_value_levene_ACC_HDV': p_levene_ACC_HDV,
            'stat_levene_ACC_HDV': stat_levene_ACC_HDV,
            'p_value_mw_HDV_AV': p_mw_HDV_AV,
            'stat_mw_HDV_AV': stat_mw_HDV_AV,
            'p_value_levene_HDV_AV': p_levene_HDV_AV,
            'stat_levene_HDV_AV': stat_levene_HDV_AV
        })

    results_df = pd.DataFrame(results)

    return results_df

def plot_differences(df):
    plt.figure(figsize=(12, 6))
    df.dropna()
    countOK,countNO = 0,0
    for k in range(len(df['p_value_mw_ACC_HDV'])):
        if df['p_value_mw_ACC_HDV'][k]<0.05:
            plt.scatter(df['speed_range'][k],df['stat_mw_ACC_HDV'][k],color = 'blue',marker = '+', s=100, label='ACC vs HDV, p<0.05' if countOK == 0 else "")
            countOK+=1
        else : 
            plt.scatter(df['speed_range'][k],df['stat_mw_ACC_HDV'][k],color = 'blue',marker = 'o', s=100, label='ACC vs HDV, p>0.05' if countNO == 0 else "")
            countNO+=1
    countOK,countNO = 0,0
    for k in range(len(df['p_value_mw_HDV_AV'])):
        if df['p_value_mw_HDV_AV'][k]<0.05:
            plt.scatter(df['speed_range'][k],df['stat_mw_HDV_AV'][k],color = 'orange',marker = '+', s=100, label='AV vs HDV, p<0.05' if countOK == 0 else "")
            countOK+=1
        else : 
            plt.scatter(df['speed_range'][k],df['stat_mw_HDV_AV'][k],color = 'orange',marker = 'o', s=100, label='AV vs HDV, p>0.05' if countNO == 0 else "")
            countNO+=1
    countOK,countNO = 0,0
    for k in range(len(df['p_value_mw_ACC_AV'])):
        if df['p_value_mw_HDV_AV'][k]<0.05:
            plt.scatter(df['speed_range'][k],df['stat_mw_ACC_AV'][k],color = 'green',marker = '+', s=100, label='ACC vs AV, p<0.05' if countOK == 0 else "")
            countOK+=1
        else : 
            plt.scatter(df['speed_range'][k],df['stat_mw_ACC_AV'][k],color = 'green',marker = 'o', s=100, label='ACC vs AV, p>0.05' if countNO == 0 else "")
            countNO+=1
        plt.yscale('log')
    plt.ylabel('Mean test Statistic on the DHW (log scale)')
    plt.xlabel('speed range [m/s]')
    plt.xticks(rotation=45)
    plt.legend(
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.43),  # Légende encore plus basse
        ncol=3, 
    )    
    plt.savefig("out/Images/DTW statistic test results.pdf",dpi=300)
    plt.show()
