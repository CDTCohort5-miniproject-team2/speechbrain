# Load required modules/packages
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, normaltest, bartlett, levene

def boxplots(depvar, indvar):
    """
    Module responsible for creating boxplots
    """
    
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=indvar, y=depvar, data=df_filtered, color='#99c2a2')
    ax = sns.swarmplot(x=indvar, y=depvar, data=df_filtered, color='#7d0013')
    if depvar == 'merged_wer':
        plt.title('Box Plot of WER per Condition')
        plt.xlabel('Condition')
        plt.ylabel('Merged WER')
    if depvar == 'pesq':
        plt.title('Box Plot of PESQ per Condition')
        plt.xlabel('Condition')
        plt.ylabel('PESQ')
    if depvar == 'stoi':
        plt.title('Box Plot of STOI per Condition')
        plt.xlabel('Condition')
        plt.ylabel('STOI')
    if depvar == 'composite_score_ovl':
        plt.title('Box Plot of Composite Score per Condition')
        plt.xlabel('Condition')
        plt.ylabel('Composite Speech Intelligibilty Score')
    plt.show()


def run_stats_tests(depvar, indvar, conditions):
    """
    Module responsible for performing a one way repeated measures ANOVA, post hoc tests and tests of ANOVA assumptions
    """
    
    # One way Repeated Measures ANOVA

    depvar_aovrm = AnovaRM(data=df_filtered, depvar= depvar, subject='scenario_id', within=['condition']).fit()
    print("\n \n Repeated Measures ANOVA Table (DV: WER, IV: condition) \n \n", depvar_aovrm.summary())

    # Testing normality using histograms and qqplot
    plt.figure(figsize=(10,6))
    plt.hist(df_filtered[depvar])
    plt.show()

    qqplot(df_filtered[depvar], line='s')
    plt.show()

    # Shapiro-Wilk test of normality
    alpha = 0.05
    stat, p = shapiro(df_filtered[depvar])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    # D’Agostino’s K^2 test of normality
        
    stat, p = normaltest(df_filtered[depvar])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    ## Testing for Homogeneity of Variance - whether the variances are approximately the same across the conditions
    
    conditions_data = {}
    for cond in conditions: 
        conditions_data[cond] = df_filtered[df_filtered['condition'] == cond][depvar].values

    # Bartlett test - if data is normally distributed
    test_statistic, p_value = bartlett(*conditions_data.values()) 
    print(test_statistic, p_value) 

    # Levene's test - if data not normally distributed
    stat, p = levene(*conditions_data.values())
    print(test_statistic, p_value)

    ## Testing for Sphericity
    sphericity_test = pg.sphericity(data = df_filtered, dv = depvar, subject = 'scenario_id', within = [indvar])
    print(sphericity_test)

    # Post Hoc Pairwise T-test

    p_value = depvar_aovrm.anova_table['Pr > F'][indvar]
    if p_value < 0.05:
        post_hocs = pg.pairwise_tests(data = df_filtered, dv=depvar, within = [indvar], subject = 'scenario_id', padjust = 'holm')
        print(post_hocs)


if __name__ == "__main__":
    df = pd.read_csv("./baseline_w_whisper_large_results.csv")

    relevant_col = ['scenario_id', 'condition', 'noise', 'n_passenger', 'pesq', 'stoi', 'composite_score_ovl', 'merged_wer']

    df_filtered = df[relevant_col].fillna(0)

    indvar = relevant_col[1]
    depvar = relevant_col[4:]
    conditions = df_filtered[indvar].unique().tolist()
    

    for variable in depvar: 
        boxplots(variable, indvar)

    for variable in depvar:
        run_stats_tests(variable, indvar, conditions)

