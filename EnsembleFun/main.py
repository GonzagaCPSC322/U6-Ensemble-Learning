import numpy as np

# pasted from DecisionTreeFun
header = ["att0", "att1", "att2", "att3"]
attribute_domains = {"att0": ["Junior", "Mid", "Senior"], 
        "att1": ["Java", "Python", "R"],
        "att2": ["no", "yes"], 
        "att3": ["no", "yes"]}
X = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]

y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
# stitch X and y together to make one table
table = [X[i] + [y[i]] for i in range(len(X))]

# TODO: begin notes on ensemble learning here

# TODO: begin notes on bagging here

# adapted from Estimating Classifier Accuracy Lab:
# 4. bootstrap method
# like random subsampling but with replacement
# create a training set by sampling N instances
# with replacement 
# N is the number instances in the dataset
# (note this N is different from ensemble learning's use of N)
# the instances not sampled form your test set
# ~63.2% of instances will be sampled into training set
# ~36.8% of instances will not (form test set)
# see github for math intuition
# repeat the bootstrap sampling k times
# accuracy is the weighted average accuracy
# over the k runs
# (weighted because test set size varies over k runs)

# (done on PA5) Ensemble Lab Task 1: 
# Write a bootstrap function to return a random sample of rows with replacement
# (test your function with the interview dataset)
def compute_bootstrapped_sample(table):
    n = len(table)
    # np.random.randint(low, high) returns random integers from low (inclusive) to high (exclusive)
    sampled_indexes = [np.random.randint(0, n) for _ in range(n)]
    sample = [table[index] for index in sampled_indexes]
    out_of_bag_indexes = [index for index in list(range(n)) if index not in sampled_indexes]
    out_of_bag_sample = [table[index] for index in out_of_bag_indexes]
    return sample, out_of_bag_sample

training_sample, validation_sample = compute_bootstrapped_sample(table)
print("training instances:")
for row in training_sample:
    print(row)
print("validation instances:")
for row in validation_sample:
    print(row)

# TODO: begin notes on random attribute subsets here

# TODO: Ensemble Lab Task 2:
# Define a python function that selects F random attributes from an attribute list
# (test your function with att_indexes (or header))

# project notes
# implement a random forest (all learners are trees)
# with bagging and with random attribute subsets
# will need to modify tree generation: for each node 
# in our tree, we use random attribute subsets.
# call compute_random_subset() right before a call to
# select_attribute() in tdidt pass the return subset
# (size F) into select_attribute()

# TODO: Ensemble Lab Task 3 (For Extra Practice):
# see https://github.com/GonzagaCPSC322/U6-Ensemble-Learning/blob/master/A%20Ensemble%20Learning.ipynb