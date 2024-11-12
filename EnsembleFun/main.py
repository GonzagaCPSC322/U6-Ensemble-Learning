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
# ensemble learning: a collection of "weak" learners
# that work together to make predictions
# together the weak learners are "stronger"
# though there is no guarantee that the ensemble
# is better than a single learner

# ensemble classification: a collection of "weak" classifiers
# that work together to make classifications
# using some voting policy (e.g., simple majority voting, weighted
# majority voting, track record voting, etc.)

# we are going to cover 3 "hyperparameters": N, M, F
# let N be the number of initial "weak" classifiers
# in our ensemble
# example: N = 100 classifiers
# homogeneous ensemble: when all N classifiers are of the
# same type
# example: N = 100 decision trees (random forest; what we will
# implement for our project)

# let M be the number of "good" classifiers that we retain
# to form our final ensemble
# M < N

# so what are some ways to generate N decision trees?
# goal: to have some diversity amongst the trees
# 1. generate a classifier (tree) using "different" attribute
# subsets
# 2. generate a classifier (tree) using "different" attribute
# selection techniques
# 3. generate a classifier (tree) using "different" training sets
# 4. others?? the only limit is your creativity

# let's zoom on #3
# recall: we talked about 4 different ways to divide
# a dataset into training and test sets
# 1. holdout method (train_test_split())
# 2. random sampling
# 3. cross validation
# 4. **bootstrap method**
# we will using #4. bootstrap method for a technique
# called bagging that implements #3 above

# TODO: begin notes on bagging here
# bagging: bootstrap aggregating
# a technique for generating N decision trees using 
# bootstrapped samples, and selecting the best M
# trees to retain for the final ensemble

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

# basic approach to bagging
# 1. divide your dataset into a test set (passed to predict())
# and a "remainder" set (passed to fit())
# 2. generate N bootstrapped samples to create N trees
# for each tree's sample:
#    ~63% of the remainder set will be sampled to form this tree's
#       training set
#    #~37% of the remainder set not sampled (out of bag sample)
#       will form this tree's validation set
# 3. evaluate each tree using its validation set and a performance
# metric (e.g, accuracy, precision, ....) and retain
# the best M trees based on their validation set performance
# 4. for each test instance, make a prediction by simple majority
# voting amonst the M trees

# why using bagging? advantages:
# 1. simple idea, simple to implement
# 2. reduces overfitting of a single decision tree
# 3. generally improves accuracy (not guaranteed!)


# TODO: begin notes on random attribute subsets here
# (expansion of #1 above)
# let F be the size of the available_attributes that we pass
# to select_attribute() in our tdidt algorithm
# F >= 2
# example: F = 2 for our interview dataset, then on an initial
# split we are only pass in a random 2 of the 4 attributes

# TODO: Ensemble Lab Task 2:
# Define a python function that selects F random attributes from an attribute list
# could just call np.random.choice()
def compute_random_subset(values, num_values):
    # let's use np.random.shuffle()
    values_copy = values.copy()
    np.random.shuffle(values_copy) # inplace
    return values_copy[:num_values]

# (test your function with att_indexes (or header))
F = 2
print(compute_random_subset(header, F))

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