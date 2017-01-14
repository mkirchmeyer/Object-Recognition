import numpy as np
import os

path='data/cat_stat/'
f='catstat_train.npy'

dst=f.split('.')[0] + ".csv"

array = np.load(os.path.join(path,f)).item()
with open(dst,"w") as target:
	for key in array.keys():
		target.write("%s,%.2f\n" % (key,100*array[key]))