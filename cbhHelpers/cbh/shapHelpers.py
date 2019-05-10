import os
import sys
import time
import traceback

from subprocess import call
import fileinput

import h5py
# import texfig first to configure Matplotlib's backend
import texfig
import matplotlib as mpl
# then, import PyPlot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap



try:
    import cPickle as pickle
except BaseException:
    import pickle

# mpl.use("pgf")
# pgf_with_custom_preamble = {"pgf.rcfonts": False}
# # don't setup fonts from rc parameters
# mpl.rcParams.update(pgf_with_custom_preamble)


class shapHelpers:
    def __init__(
        self, target, features_shap, shap_values, shap_int_vals, model, figfolder, datafolder, modelfolder
    ):
        self.target = target
        self.features_shap = features_shap
        self.shap_values = shap_values
        self.shap_int_vals = shap_int_vals
        self.model = model
        self.figfolder = figfolder
        self.datafolder = datafolder
        self.modelfolder = modelfolder
        self.n_features = str(len(self.features_shap.columns))
        self.timestr = time.strftime("_%Y-%m-%d-%H%M")
        """ 
        target = prediction target, for getting file names right (string, e.g. "readmitted30d")
        features_shap = selection of features from original dataset (X)
        shap_values  = output of SHAP explainer, e.g. 
        -----explainer = shap.TreeExplainer(gbm_model)
        -----shap_values = explainer.shap_values(features_shap)
        figfolder, datafolder, modelfolder = desired output folders for figures, data, models
        """

    def shap_save_to_disk(self):
        n_features = self.n_features
        print(f"Saving {self.target} SHAP values to pickle...")
        file_title = f"{self.target}_SHAP_values_"
        # timestr = time.strftime("_%Y-%m-%d-%H%M")
        timestr = self.timestr
        ext = ".pickle"
        title = file_title + n_features + timestr + ext
        full_path = self.modelfolder / title
        # Pickled numpy array
        with open(full_path, "wb") as f:
            pickle.dump(self.shap_values, f)
        print("File available at", full_path)

        file_title = f"{self.target}_SHAP_features_"
        title = file_title + n_features + timestr + ext
        full_path = self.modelfolder / title
        with open(full_path, "wb") as f:
            pickle.dump(self.features_shap, f)
        print("File available at", full_path)
        print("Saving SHAP to .h5 file...")
        file_title = f"{self.target}_{n_features}_everything_"
        timestr = time.strftime("_%Y-%m-%d")
        ext = ".h5"
        title = file_title + timestr + ext
        h5_file = self.modelfolder / title
        shap_val_df = pd.DataFrame(self.shap_values)
        shap_feat_df = pd.DataFrame(self.features_shap)
        shap_int_vals_df = pd.DataFrame(self.shap_int_vals)
        shap_val_df.to_hdf(h5_file, key='shap_values', format="table")
        shap_feat_df.to_hdf(h5_file, key='features_shap', format="table")
        shap_int_vals_df.to_hdf(h5_file, key = 'shap_int_vals', format='table')

    def save_requirements(self):
        
        if os.name == 'nt':
            call("bash", shell=True)
        call("conda list > requirements.txt", shell=True)

        for line in fileinput.input("requirements.txt", inplace=True):
        # inside this loop the STDOUT will be redirected to the file
            print(line.replace("# Name", "Name"))

        call("cat requirements.txt | tr -s '[:blank:]' ',' > ofile.csv", shell=True)

        reqs = pd.read_csv("ofile.csv")
        reqs = reqs[1:] # select rows with meaningful data
        cols = [4,5] # define empty columns
        reqs.drop(reqs.columns[cols],axis=1,inplace=True)
        reqs = reqs.rename(columns=reqs.iloc[0])
        reqs = reqs[["Name", "Version", "Build", "Channel"]]
        print(reqs.head)

        os.remove("requirements.txt")
        os.remove("ofile.csv")
        # If on Unix, could do this instead:
        # call("rm requirements.txt", shell=True)
        # call("rm ofile.csv", shell=True)

        file_title = f"{self.target}_{self.n_features}_everything_"
        timestr = time.strftime("_%Y-%m-%d")
        ext = ".h5"
        title = file_title + timestr + ext
        h5_file = self.modelfolder / title
        reqs.to_hdf(h5_file, key='requirements', format="table")

    def shap_save_ordered_values(self):
        n_features = self.n_features
        print("Saving SHAP values to disk in order of importance...")
        df_shap_train = pd.DataFrame(
            self.shap_values, columns=self.features_shap.columns.values
        )
        imp_cols = (
            df_shap_train.abs().mean().sort_values(ascending=False).index.tolist()
        )
        pickle_title = f"{self.target}_shap_df.pickle"
        df_shap_train.to_pickle(self.datafolder / pickle_title)
        print("Pickle available at", self.datafolder / pickle_title)
        imp_cols = pd.DataFrame(imp_cols)
        print(imp_cols)
        timestr = time.strftime("%Y-%m-%d-%H%M")
        csv_title = f"{timestr}_{self.target}_{n_features}_shap.csv"
        imp_cols.to_csv(self.datafolder / csv_title)
        print("CSV available at", self.datafolder / csv_title)
        print("Saving SHAP df and important columns to .h5 file...")
        file_title = f"{self.target}_{n_features}_everything_"
        timestr = time.strftime("_%Y-%m-%d")
        ext = ".h5"
        title = file_title + timestr + ext
        h5_file = self.modelfolder / title
        df_shap_train.to_hdf(h5_file, key='df_shap_train', format="table")
        imp_cols.to_hdf(h5_file, key='ordered_shap_cols', format="table")

    def shap_top_dependence_plots(self, n_plots):
        n_features = self.n_features
        # Dependence plots don't like the "category" dtype, will only work with
        # category codes (ints).
        # Make a copy of the df that uses cat codes
        # Then use the original df as the "display"
        df_with_codes = self.features_shap.copy()
        df_with_codes.columns = (
            df_with_codes.columns.str.strip()
            .str.replace("_", " ")
            .str.replace("__", " ")
            .str.capitalize()
        )
        # This requires that you already set dtype to "category" as appropriate
        # in "features_shap" or earlier
        for col in list(df_with_codes.select_dtypes(include="category")):
            df_with_codes[col] = df_with_codes[col].cat.codes

        for i in range(n_plots):
            try:
                shap.dependence_plot(
                    "rank(%d)" % i,
                    self.shap_values,
                    df_with_codes,
                    display_features=self.features_shap,
                    show=False,
                    feature_names=self.features_shap.columns,
                )

                print(f"Making dependence plot for {self.target} feature ranked {i}...")
                figure_title = f"{self.target}_SHAP_dependence_{i}_"
                timestr = time.strftime("_%Y-%m-%d-%H%M_")
                ext = ".png"
                title = figure_title + n_features + timestr + ext
                plt.savefig(
                    (self.figfolder / title),
                    dpi=1200,
                    transparent=True,
                    bbox_inches="tight",
                )
                try:
                    title1 = figure_title + n_features + timestr
                    title1 = str(self.figfolder) + "/" + title1
                    texfig.savefig(
                        title1, dpi=1200, transparent=True, bbox_inches="tight"
                    )
                except Exception as exc:
                    print(traceback.format_exc())
                    print(exc)
                    print("Aww, LaTeX!..")
                plt.close()
            except:
                print(f"Plot for feature {i} failed, moving on...")

    def shap_prettify_column_names(self, prettycols_file):
        # This part is just to make the names match the ugly half of the prettifying file
        self.features_shap.columns = (
            self.features_shap.columns.str.strip()
            .str.replace("\t", "")
            .str.replace("_", " ")
            .str.replace("__", " ")
            .str.replace(", ", " ")
            .str.replace(",", " ")
            .str.replace("'", "")
            .str.capitalize()
        )
        # Define and load the prettifying file
        prettycols_file = prettycols_file
        prettycols = pd.read_csv(prettycols_file)
        # Now the magic happens
        # make a dict out of the old and new columns
        # and map the new onto the old
        di = prettycols.set_index("feature_ugly").to_dict()
        self.features_shap.columns = self.features_shap.columns.to_series().map(
            di["feature_pretty"]
        )
        return self.features_shap

    def shap_plot_summaries(self, title_in_figure):
        n_features = self.n_features
        print(f"{self.target} Making SHAP summary bar plot PNG...")
        mpl.rcParams.update(mpl.rcParamsDefault)
        shap.summary_plot(
            self.shap_values,
            self.features_shap,
            title=title_in_figure,
            plot_type="bar",
            show=False,
            feature_names=self.features_shap.columns,
        )
        figure_title = f"{self.target} SHAP_summary_bar_"
        timestr = time.strftime("_%Y-%m-%d-%H%M")
        ext = ".png"
        title = figure_title + n_features + timestr + ext
        plt.savefig(
            (self.figfolder / title), dpi=1200, transparent=True, bbox_inches="tight"
        )
        plt.close()

        print(f"{self.target} Making SHAP summary bar plot PDF...")
        import texfig # set rcParams to texfig version...
        shap.summary_plot(
            self.shap_values,
            self.features_shap,
            title=title_in_figure,
            plot_type="bar",
            show=False,
            feature_names=self.features_shap.columns,
        )
        figure_title = f"{self.target} SHAP_summary_bar_"
        try:
            title1 = figure_title + n_features + timestr
            title1 = str(self.figfolder) + "/" + title1
            texfig.savefig(title1, dpi=1200, transparent=True, bbox_inches="tight")
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
            print("Aww, LaTeX!..")
        plt.close()

        print(f"{self.target} Making SHAP summary plot PNG...")
        mpl.rcdefaults()
        mpl.use("Qt5Agg")
        print("Current mpl backend:", mpl.get_backend())
        shap.summary_plot(
            self.shap_values,
            self.features_shap,
            title=title_in_figure,
            feature_names=self.features_shap.columns,
            show=False,
        )
        figure_title = f"{self.target}_SHAP_summary_"
        timestr = time.strftime("%Y-%m-%d-%H%M")
        ext = ".png"
        title = figure_title + n_features + timestr + ext
        plt.savefig(
            (self.figfolder / title), dpi=1200, transparent=True, bbox_inches="tight"
        )
        plt.close()

        print(f"{self.target} Making SHAP summary plot PDF...")
        import texfig # set rcParams to texfig version...
        shap.summary_plot(
            self.shap_values,
            self.features_shap,
            title=title_in_figure,
            feature_names=self.features_shap.columns,
            show=False,
        )
        try:
            title1 = figure_title + n_features + timestr
            title1 = str(self.figfolder) + "/" + title1
            # texfig.savefig(title1, dpi=1200, transparent=True, bbox_inches="tight")
            texfig.savefig(title1, bbox_inches="tight")
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
            print("Aww, LaTeX!..")
        plt.close()

    def shap_int_vals_heatmap(self):
        tmp = np.abs(self.shap_int_vals).sum(0)
        for i in range(tmp.shape[0]):
            tmp[i,i] = 0
        inds = np.argsort(-tmp.sum(0))[:50]
        tmp2 = tmp[inds,:][:,inds]
        pl.figure(figsize=(12,12))
        pl.imshow(tmp2)
        pl.yticks(range(tmp2.shape[0]), self.features_shap.columns[inds], rotation=50.4, horizontalalignment="right")
        pl.xticks(range(tmp2.shape[0]), self.features_shap.columns[inds], rotation=50.4, horizontalalignment="left")
        pl.gca().xaxis.tick_top()
        figure_title = f"{self.target} SHAP_int_vals_heatmap"
        try:
            title1 = figure_title + self.n_features + self.timestr
            title1 = str(self.figfolder) + "/" + title1
            texfig.savefig(title1, dpi=1200, transparent=True, bbox_inches="tight")
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
            print("Aww, LaTeX!..")
        plt.close()

    def shap_random_force_plots(self, n_plots, expected_value):
        n_features = self.n_features
        n_plots = n_plots
        # Needs explainer.expected_value
        expected_value = expected_value
        shap_indices = np.random.choice(
            self.shap_values.shape[0], n_plots
        )  # select 5 random patients
        for pt_num in shap_indices:
            # set params to default for png...
            mpl.rcdefaults()
            mpl.use("Qt5Agg")
            print("Current mpl backend:", mpl.get_backend())
            print(f"{self.target} Making force plot for patient", pt_num, "...")
            shap.force_plot(
                expected_value,  # this version uses the standard base value
                self.shap_values[pt_num, :],
                self.features_shap.iloc[
                    pt_num, :
                ],  # grabs the identities and values of components
                text_rotation=15,  # easier to read
                matplotlib=True,  # instead of Javascript
                show=False,  # allows saving, etc.
                link="logit",
                feature_names=self.features_shap.columns,
            )
            figure_title = f"{self.target}_{n_features}_SHAP_"
            patient_number = f"_Pt_{pt_num}"
            timestr = time.strftime("%Y-%m-%d-%H%M")
            ext = ".png"
            title = figure_title + timestr + patient_number + ext
            forcefolder = self.figfolder / "force_plots"
            if not os.path.exists(forcefolder):
                print("Making folder called", forcefolder)
                os.makedirs(forcefolder)
            plt.savefig(
                (forcefolder / title), dpi=1200, transparent=True, bbox_inches="tight"
            )
            try:
                print(f"{self.target} Making SVG force plot for patient", pt_num, "...")
                # update params for svg...
                mpl.use("svg")
                print("Current mpl backend:", mpl.get_backend())
                new_rc_params = {'text.usetex': False,
                    "svg.fonttype": 'none'
                    }
                mpl.rcParams.update(new_rc_params)
                shap.force_plot(
                    expected_value,  # this version uses the standard base value
                    self.shap_values[pt_num, :],
                    self.features_shap.iloc[
                        pt_num, :
                    ],  # grabs the identities and values of components
                    text_rotation=15,  # easier to read
                    matplotlib=True,  # instead of Javascript
                    show=False,  # allows saving, etc.
                    link="logit",
                    feature_names=self.features_shap.columns, 
                )
            
                ext0 = ".svg"
                title0 = figure_title + timestr + patient_number + ext0
                if not os.path.exists(forcefolder):
                    print("Making folder called", forcefolder)
                    os.makedirs(forcefolder)
                plt.savefig(
                    (forcefolder / title0), dpi=1200, transparent=True, bbox_inches="tight"
                )
                title1 = figure_title + timestr + patient_number

                # reset mpl params for LaTeX PDF via texfig.py
                import texfig 
                shap.force_plot(
                    expected_value,  # this version uses the standard base value
                    self.shap_values[pt_num, :],
                    self.features_shap.iloc[
                        pt_num, :
                    ],  # grabs the identities and values of components
                    text_rotation=15,  # easier to read
                    matplotlib=True,  # instead of Javascript
                    show=False,  # allows saving, etc.
                    link="logit",
                    feature_names=self.features_shap.columns, 
                )
                title1 = figure_title + n_features + timestr
                title1 = str(forcefolder) + "/" + title1
                texfig.savefig(title1, dpi=1200, transparent=True, bbox_inches="tight")
            except Exception as exc:
                print(traceback.format_exc())
                print(exc)
                print("Aww, LaTeX!..")
            plt.close()
