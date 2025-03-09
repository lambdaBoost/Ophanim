import pickle
import os
import shutil

THRESHOLD = 0.29
DEST = "inference_data/candidates/NC6"
PREDICTIONS = 'NCset6-predictions.pickle'
FILENAMES = 'NCset6-filenames.pickle'

with open(PREDICTIONS, 'rb') as f:
    predictions = pickle.load(f)

with open(FILENAMES, 'rb') as g:
    filenames = pickle.load(g)

predictions = predictions.flatten().tolist()

d = dict(zip(filenames,predictions))

positives = dict((k, v) for k, v in d.items() if v >= THRESHOLD)

positive_files = list(positives.keys())

for fl in positive_files:
    shutil.copy(fl, DEST)