# pip install confidence_intervals

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM



df = pd.read_csv("./baseline_w_whisper_large_results.csv")

df_filtered = df[['scenario_id', 'condition', 'noise', 'n_passenger', 'pesq', 'stoi', 'composite_score_ovl', 'merged_wer']]

for column in df_filtered.columns:
    if df_filtered[column].isnull().any():
        df_filtered.loc[:, column] = df_filtered[column].fillna(0)

wer_aovrm = AnovaRM(data=df_filtered, depvar='merged_wer', subject='scenario_id', within=['condition']).fit()
print("\n \n Repeated Measures ANOVA Table (DV: WER, IV: condition) \n \n", wer_aovrm.summary())

pesq_aovrm = AnovaRM(data=df_filtered, depvar='pesq', subject='scenario_id', within=['condition']).fit()
print("\n \n Repeated Measures ANOVA Table (DV: PESQ, IV: condition) \n \n", pesq_aovrm.summary())

stoi_aovrm = AnovaRM(data=df_filtered, depvar='stoi', subject='scenario_id', within=['condition']).fit()
print("\n \n Repeated Measures ANOVA Table (DV: STOI, IV: condition) \n \n", stoi_aovrm.summary())

composite_aovrm = AnovaRM(data=df_filtered, depvar='composite_score_ovl', subject='scenario_id', within=['condition']).fit()
print("\n \n Repeated Measures ANOVA Table (DV: Composite, IV: condition) \n \n", composite_aovrm.summary())

system_names_mapping = {1: 'baseline', 2: 'SE only', 3: 'SS only', 4: 'SE + SS', 5: 'SS + SE'}
df_filtered.loc[:, 'condition'] = df_filtered['condition'].map(system_names_mapping)

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='condition', y='merged_wer', data=df_filtered, color='#99c2a2')
ax = sns.swarmplot(x='condition', y='merged_wer', data=df_filtered, color='#7d0013')
plt.title('Box Plot of WER per Condition')
plt.xlabel('Condition')
plt.ylabel('Merged WER')
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='condition', y='pesq', data=df_filtered, color='#99c2a2')
ax = sns.swarmplot(x='condition', y='pesq', data=df_filtered, color='#7d0013')
plt.title('Box Plot of PESQ scores per Condition')
plt.xlabel('Condition')
plt.ylabel('PESQ')
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='condition', y='stoi', data=df_filtered, color='#99c2a2')
ax = sns.swarmplot(x='condition', y='stoi', data=df_filtered, color='#7d0013')
plt.title('Box Plot of STOI scores per Condition')
plt.xlabel('Condition')
plt.ylabel('STOI')
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='condition', y='composite_score_ovl', data=df_filtered, color='#99c2a2')
ax = sns.swarmplot(x='condition', y='composite_score_ovl', data=df_filtered, color='#7d0013')
plt.title('Box Plot of Composite Scores per Condition')
plt.xlabel('Condition')
plt.ylabel('Composite Score')
plt.show()