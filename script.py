import numpy as np
from data import Dataset

trainset = Dataset(file_dir="../data/trainset/", load_type="age")
testset = Dataset(file_dir="../data/testset/", load_type="age")

max_age = None
min_age = None
for batch in range(len(trainset)):
    x, ages = trainset[batch]
    at = np.max(ages)
    it = np.min(ages)
    if max_age is None or max_age < at:
        max_age = at
    if min_age is None or min_age > it:
        min_age = it

print(max_age, min_age)

max_age = None
min_age = None
for batch in range(len(trainset)):
    x, ages = trainset[batch]
    at = np.max(ages)
    it = np.min(ages)
    if max_age is None or max_age < at:
        max_age = at
    if min_age is None or min_age > it:
        min_age = it

print(max_age, min_age)
