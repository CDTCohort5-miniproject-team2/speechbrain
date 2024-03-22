# Load required modules/packages
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, normaltest, kruskal
import scikit_posthocs as sp



def boxplots(df, depvar, indvar):
    """
    Module responsible for creating boxplots
    """
    
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=indvar, y=depvar, data=df, color='#99c2a2')
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
    plt.show()


def run_anova(df, depvar, indvar):
    """
    Module responsible for performing a one way repeated measures ANOVA
    """
    
    # One way Repeated Measures ANOVA

    depvar_aovrm = AnovaRM(data=df, depvar= depvar, subject='scenario_id', within=[indvar]).fit()
    print(f"\n \n Repeated Measures ANOVA Table (DV: {depvar}, IV: condition) \n \n", depvar_aovrm.summary())

    return depvar_aovrm

def test_anova_assumptions(df, depvar, indvar):
    """
    Module responsible for testing assumptions of one-way RM Anova (normality, sphrericity)
    """

    # Testing normality using histograms and qqplot
    plt.figure(figsize=(10,6))
    plt.hist(df[depvar])
    if depvar == 'wer':
        plt.title('Histogram of WER')
        plt.xlabel('Merged WER')
    if depvar == 'Difference_PESQ_score':
        plt.title('Histogram of PESQ scores')
        plt.xlabel('Difference in PESQ score between unchanged audio and enhanced audio')
    if depvar == 'Difference_STOI_score':
        plt.title('Histogram of STOI scores')
        plt.xlabel('Difference in STOI between unchanged audio and enhanced audio')
    plt.ylabel('frequency')
    plt.show()

    qqplot(df[depvar], line='s')
    plt.show()

    # Shapiro-Wilk test of normality
    alpha = 0.05
    shapiro_wilk_stat, shapiro_wilk_pval = shapiro(df[depvar])
    print("\nResult of Shapiro-Wilk test of normality: \nStatistics=%.3f, p=%.3f" % (shapiro_wilk_stat, shapiro_wilk_pval))
    if shapiro_wilk_pval > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    # D’Agostino’s K^2 test of normality
        
    k2_stat, k2_pval = normaltest(df[depvar])
    print("\nResult of D'Agostino's K^2 test of normality: \nStatistics=%.3f, p=%.3f" % (k2_stat, k2_pval))
    if k2_pval > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    ## Testing for Sphericity
    sphericity_test = pg.sphericity(data = df, dv = depvar, subject = 'scenario_id', within = [indvar])
    print(f"\nResult of Sphericity test: \n {sphericity_test}")

    return shapiro_wilk_pval, k2_pval, sphericity_test

def run_post_hoc_anova(df, depvar, indvar, depvar_aovrm):
    
    # Post Hoc Pairwise T-test
    p_value = depvar_aovrm.anova_table['Pr > F'][indvar]
    print(p_value)
    if p_value < 0.05:
        post_hocs = pg.pairwise_tests(data = df, dv=depvar, within = [indvar], subject = 'scenario_id', padjust = 'holm')
        print(post_hocs)

def kruskall_Wallis_test(df, dep_var, indep_var, subject):

    depv_data = df.pivot_table(index=subject, columns=indep_var, values=dep_var)

    depvar_array = depv_data.values
    stats, p_value = kruskal(*depvar_array.T)
    print(f"\nKruskal-Wallis Test for {dep_var}:\n Test Statistic: {stats} \n P-value: {p_value}")

    if p_value < 0.05:
        p_values= sp.posthoc_dunn(depvar_array, p_adjust = 'holm')
        print(f'\nDunn Test P-value: {p_values}')
 

if __name__ == "__main__":
    
    # Loading datasets into seperate dataframes
    baseline_df = pd.read_csv("results_spreadsheets/baseline_w_medium_en_results.csv")
    SE_only_df = pd.read_csv("results_spreadsheets/aec_enhancer_asr_w_medium_en_results.csv")
    SS_only_df = pd.read_csv("results_spreadsheets/aec_separator_asr_w_medium_en_results.csv")
    SE_SS_df = pd.read_csv("results_spreadsheets/baseline_w_medium_en_results.csv")
    SS_SE_df = pd.read_csv("results_spreadsheets/baseline_w_medium_en_results.csv")
    constant_df = pd.read_csv("results_spreadsheets/CONSTANT_PESQ_STOI_STATS_ONLY.csv")

    # Adding a column for condition to each
    baseline_df['condition'] = 'baseline'
    SE_only_df['condition'] = 'SE only'
    SS_only_df['condition'] = 'SS only'
    SE_SS_df['condition'] = 'SE + SS'
    SS_SE_df['condition'] = 'SS + SE'

    # Filtering out the unnessary datasets
    relevant_col = ['scenario_id', 'condition', 'has_noise', 'num_passengers', 'pesq', 'stoi', 'wer']
    
    baseline_df = baseline_df[relevant_col]
    SE_only_df = SE_only_df[relevant_col]
    SS_only_df = SS_only_df[relevant_col]
    SE_SS_df = SE_SS_df[relevant_col]
    SS_SE_df = SS_SE_df[relevant_col]
    constant_df = constant_df[["scenario_id", 'CONSTANT_closetalk_v_noisy_wall_mic_pesq', 
                                       'CONSTANT_closetalk_v_noisy_wall_mic_stoi']]
    
    
    df_list = [baseline_df, SE_only_df, SS_only_df, SE_SS_df, SS_SE_df]
    merged_df = pd.concat(df_list, ignore_index= True)
    merged_df = pd.merge(merged_df, constant_df, on = "scenario_id")

    merged_df = merged_df[merged_df['has_noise'] != 'Not used']
    merged_df = merged_df.dropna(subset=['wer'])
    merged_df['has_noise'] = merged_df['has_noise'].fillna('no noise')
    
    
    merged_df['Difference_PESQ_score'] = merged_df['pesq'] - merged_df['CONSTANT_closetalk_v_noisy_wall_mic_pesq']
    merged_df['Difference_STOI_score'] = merged_df['stoi'] - merged_df['CONSTANT_closetalk_v_noisy_wall_mic_stoi']


    indvar = 'condition'
    depvar = ['Difference_STOI_score', 'Difference_PESQ_score', 'wer']
    conditions = merged_df[indvar].unique().tolist()

    for variable in depvar:
        boxplots(merged_df, variable, indvar)
        depvar_aovrm = run_anova(merged_df, variable, indvar)
        assumption_p_values = []
        assumption_p_values = test_anova_assumptions(merged_df, variable, indvar)
        
        if assumption_p_values[0] < 0.05 or assumption_p_values[1] < 0.05 or assumption_p_values[2] < 0.05:
            kruskall_Wallis_test(merged_df, variable, indvar, 'scenario_id')

            






