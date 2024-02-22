# pip install confidence_intervals

from confidence_intervals import evaluate_with_conf_int

# Define the metric of interest (could be a custom method)
from sklearn.metrics import accuracy_score

# Create a toy dataset for this example (to be replaced with your actual data)
from confidence_intervals.utils import create_data
decisions, labels, conditions = create_data(200, 200, 20)

# Run the function. In this case, the samples are represented by the categorical decisions
# made by the system which, along with the labels, is all that is needed to compute the metric.
samples = decisions
output = evaluate_with_conf_int(samples, accuracy_score, labels, conditions, num_bootstraps=1000, alpha=5)
print(output)
