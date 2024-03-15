# pip install confidence_intervals

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("./baseline_w_whisper_large_results.csv")

df_filtered = df[['scenario_id', 'condition', 'noise', 'n_passenger', 'pesq', 'stoi', 'composite_score_ovl', 'merged_wer']]

for column in df_filtered.columns:
    if df_filtered[column].isnull().any():
        df_filtered.loc[:, column] = df_filtered[column].fillna(0)

wer_lm = ols('merged_wer ~ C(condition)', data=df_filtered).fit()
pesq_lm = ols('pesq ~ C(condition)', data=df_filtered).fit()
stoi_lm = ols('stoi ~ C(condition)', data=df_filtered).fit()
composite_lm = ols('composite_score_ovl ~ C(condition)', data=df_filtered).fit()

wer_table = sm.stats.anova_lm(wer_lm)
pesq_table = sm.stats.anova_lm(pesq_lm)
stoi_table = sm.stats.anova_lm(stoi_lm)
composite_table = sm.stats.anova_lm(composite_lm)

print("\n \n ANOVA Table (DV: WER, IV: condition) \n \n", wer_table)
print("\n \n ANOVA Table (DV: PESQ, IV: condition) \n \n", pesq_table)
print("\n \n ANOVA Table (DV: STOI, IV: condition) \n \n", stoi_table)
print("\n \n ANOVA Table (DV: Composite measure, IV: condition) \n \n", composite_table)

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