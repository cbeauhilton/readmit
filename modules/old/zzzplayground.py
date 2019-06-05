try:
    import cPickle as pickle
except BaseException:
    import pickle
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
import os
import time
import config
import traceback
from pathlib import Path

pkl_model = Path(r"C:\Users\hiltonc\Desktop\readmit\models\2019-04-09\readmitted30d\readmitted30d_285_features_MODEL_285_2019-04-09-1039_.pickle")
gbm_model = pickle.load(pkl_model)

with open(pkl_model, "wb") as fout:
    pickled = pickle.dump(self.gbm_model, fout)
print(pkl_model)
frozen = jsonpickle.encode(pickled)
print("Saving GBM model to .h5 file...")
file_title = f"{self.target}_{n_features}_everything_"
timestr = time.strftime("_%Y-%m-%d")
ext = ".h5"
title = file_title + timestr + ext
h5_file = self.modelfolder / title
with h5py.File(h5_file, 'a') as f:
    try:
        f.create_dataset('gbm_model', data=frozen)
    except Exception as exc:
        print(traceback.format_exc())
        print(exc)
        try:
            del f["gbm_model"]
            f.create_dataset('gbm_model', data=frozen)
            print("Successfully deleted old gbm model and saved new one!")
        except:
            print("Old gbm model persists...")
print(h5_file)