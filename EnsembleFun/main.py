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
# ensemble learning: a collection of "weak" learners that
# work together to make predictions
# the ensemble is "stronger", but this is not guaranteed

# ensemble classification: a collection of "weak" classifiers
# that work together to make classifications using some
# voting policy (simple majority voting, weighted majority voting,
# track record voting, etc.)

# let N be the number of classifiers in our initial ensemble
# example: N = 100, 100 classifiers in the ensemble
# homegeneous ensemble: all the classifiers are of the same type
# example: N = 100 decision trees
# this is called a random forest (what we will implement
# for project)
# heterogeneous ensemble: a mix of classifier types

# let M be the number of "better" classifiers from our intial
# ensemble of N classifiers that we retain to form our ensemble
# M < N

# how might we generate the initial N classifiers/
# goal is to have diversity amongst them
# 1. generate a classifier (tree) using different training sets 
# 2. generate a classifier (tree) using different attribute subsets
# 3. generate a classifier (tree) using different attribute
# selection techniques
# 4. others?? the only limit is your creativity

# let's focus on #1 first

# TODO: begin notes on bagging here
# recall: we went over 4 ways to divide a dataset into
# training and testing
# 1. holdout method (train_test_split) 
# 2. random subsampling
# 3. cross validation
# 4. **bootstrap method**

# pasted from ClassificationFun (note this N is different from ensemble learning's use of N):
# 4. bootstrap method
# like random subsampling but with replacement
# create a training set by sampling N instances
# with replacement 
# N is the number instances in the dataset
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

# bagging: bootstrap aggregating
# an ensemble technique for genreating N diverse trees
# using N bootstrapped samples and retaining the "best"
# M trees for the ensemble
# basic approach
# 1. divide your dataset into a test set (passed into predict())
# and a "remainder set" (passed into fit())
# 2. generate N bootstrapped samples (N as in number of initial
# trees in our ensemble) to create N decision trees
# for each decision tree:
#    build the tree using the ~63% of instances in this tree's
#      bootstrapped sample
#    reserve the ~37% out of bag instances for this tree's 
#      VALIDATION SET (each tree has a different validation set)
# 3. measure the performance of each tree on its validation set
# using some performance measure (e.g. accuracy, F1, precision, recall, etc.)
# retain the "best" M trees using these validation set performance results
# 4. for each instance in the test set, get a prediction from the
# ensemble using simple majority voting over the M trees

# advantages of bagging
# 1. simple idea, simple implement
# 2. reduces overfitting
# 3. generally improves accuracy
# (by reducing the classification variance over the classifiers)

# TODO: begin notes on random attribute subsets here
# let F be the number of random attributes to select from
# the available attributes right before we call select_attribute()
# in tdidt()
# F >= 2

# TODO: Ensemble Lab Task 2:
# Define a python function that selects F random attributes from an attribute list
# (test your function with att_indexes (or header))
def compute_random_subset(values, num_values):
    # could use np.random.choice()
    # I'll use np.random.shuffle() and slicing
    values_copy = values.copy() # shallow copy
    np.random.shuffle(values_copy) # inplace shuffle
    return values_copy[:num_values]

F = 2
subset = compute_random_subset(header, F)
print(subset)

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