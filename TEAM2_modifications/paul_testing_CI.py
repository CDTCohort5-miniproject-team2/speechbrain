# pip install confidence_intervals

from confidence_intervals import evaluate_with_conf_int
import sklearn.metrics
import pandas as pd
import numpy as np

wer_df = pd.read_csv("./baseline_w_whisper_large_results.csv")

conditions = np.array(wer_df['condition'])
wer_array = np.array(wer_df['merged_wer'])

#condition_wer_df = wer_df[['condition', 'merged_wer']]


#condition_1_wer = wer_df[wer_df['condition'] == 1]['merged_wer'].values
#condition_2_wer = wer_df[wer_df['condition'] == 2]['merged_wer'].values

#WER_per_condition_array = np.column_stack((condition_1_wer, condition_2_wer))

alpha = 5
num_bootstraps = int(50/alpha*100)

def metric(wer):
    return np.average(wer)

# Run the function. In this case, the samples are represented by the categorical decisions
# made by the system which, along with the labels, is all that is needed to compute the metric.

output = evaluate_with_conf_int(samples=wer_array, metric = metric, labels = None, conditions = conditions, num_bootstraps=num_bootstraps, alpha=alpha, samples2 = None)
print(output)

# First number if the metric value on the full dataset. The list indicates the lower and upper bound of the confidence interval

# Arguments for evaluate_with_conf_int: 
# Samples = array with a value needed to compute the metric for each sample e.g., the systems output
# Labels = array with the label needed to comptue the metric 
# Metric = used to assess performance - i.e. 
# Number of bootstrap sets default to 1000
# alpha = p.value - we will want to set this to 0.05 (convention) - current default

# For our task, will need to be:
# evaluate_with_conf_int(samples, Metric = Weighted Av. WER, labels = num words, condition = speaker, num_bootstraps = 1000, alpha=5)
