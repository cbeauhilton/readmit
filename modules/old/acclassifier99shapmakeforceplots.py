#%%
import glob
import os
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import shap
shap.initjs()
# import matplotlib as mpl
# import importlib
# print(mpl.get_configdir())
# print(mpl.rcParams)
# print(mpl.matplotlib_fname())
# mpl.rcdefaults()
# print(mpl.rcParams)
bknds = ['agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import sys
sys.path.append("C:\\Users\\hiltonc\\Desktop\\readmit\\readmit\\modules")
from cbh import config

print("Loading path to h5...")
model_dir = config.MODELS_DIR
shap_csv_dir = config.SHAP_CSV_DIR
prettified_csv_dir = shap_csv_dir / "prettified"
tmp_csv_dir = prettified_csv_dir / "tmp"
datestr = "2019-05-23"
dirpath = model_dir  # / datestr

#%%
def make_force_plots():
    for bkn in bknds:
        try:
            
            # mpl.use(f"{bkn}")
            # mpl.use('agg')
            # # mpl.use('Cairo') # nope, worse
            # mpl.use('pdf')
            # mpl.use('pgf')
            # mpl.use('ps')
            # mpl.use('svg')
            # mpl.use('template')

            import matplotlib.pyplot as plt
            # import matplotlib.rcsetup as rcsetup

            # from matplotlib.backends.backend_pgf import PdfPages
            for filename in Path(dirpath).glob("**/**/*.h5"):

                try:
                    justname = os.path.split(filename)[1]
                    savefile = Path(prettified_csv_dir / justname)
                    f = h5py.File(Path(filename), "r")
                    # keylist = list(f.keys())
                    # print("This h5 file contains", keylist)
                    shap_expected_value = pd.read_hdf(
                        Path(f"{filename}"), key="shap_expected_value"
                    )
                    target = shap_expected_value.iloc[0]["target"]
                    print(filename)
                    print(target)
                    # print(rcsetup.interactive_bk)
                    # print(rcsetup.non_interactive_bk)
                    # print(rcsetup.all_backends)
                    n_plots = 1
                    # name_for_figs = shap_expected_value.iloc[0]["name_for_figs"]
                    class_thresh = shap_expected_value.iloc[0]["class_thresh"]
                    expected_value = shap_expected_value.iloc[0]["shap_exp_val"]
                    shap_vals = pd.read_hdf(filename, key="shap_values")
                    shap_values = shap_vals.to_numpy(copy=True)
                    features_shap = pd.read_hdf(filename, key="features_shap")

                    shap_indices = np.random.choice(shap_values.shape[0], n_plots)
                    for pt_num in shap_indices:
                        print("Pt_num:", pt_num)
                        
                        bknd = plt.get_backend()
                        figure_title = f"{target}_SHAP_forceplot_Pt_{pt_num}"
                        # ext = ".pdf"
                        # ext = ".pdf"
                        ext = ".png"
                        title = figure_title + ext
                        shap.force_plot(
                            expected_value,  # this version uses the standard base value
                            shap_values[pt_num, :],
                            features_shap.iloc[
                                pt_num, :
                            ],  # grabs the identities and values of components
                            text_rotation=15,  # easier to read
                            matplotlib=True,  # instead of Javascript
                            show=False,  # allows saving, etc.
                            link="logit",
                            feature_names=features_shap.columns,
                        )
                        plt.savefig(
                            (title),
                            dpi=1200,
                            transparent=True,
                            bbox_inches="tight",
                            format='png',
                        )
                        # pp = PdfPages('multipage.pdf')
                        # pp.savefig()
                        # pp.close()
                        # pdf=PdfPages('figure2.pdf')
                        # pdf.savefig(plt.gcf())
                except Exception as exc:
                    print(traceback.format_exc())
        except Exception as exc:
            print(traceback.format_exc())

make_force_plots()


#%%
