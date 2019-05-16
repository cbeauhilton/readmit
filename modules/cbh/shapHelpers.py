import fileinput
import json
import os
import shutil
import sys
import time
import traceback
from subprocess import call
from tqdm import tqdm

tqdm.pandas()
import cbh.config as config
# import cbh.texfig first to configure Matplotlib's backend
import cbh.texfig as texfig
import conda.cli
import h5py
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
        self, target, features_shap, shap_values, shap_expected, model, figfolder, datafolder, modelfolder
    ):
        self.target = target
        self.features_shap = features_shap
        self.shap_values = shap_values
        self.shap_expected = shap_expected
        self.model = model
        self.figfolder = figfolder
        self.datafolder = datafolder
        self.modelfolder = modelfolder
        self.n_features = str(len(self.features_shap.columns))
        self.timestr = time.strftime("_%Y-%m-%d-%H%M_")
        self.timestr_d = time.strftime("_%Y-%m-%d_")
        file_title = f"{self.target}_{self.n_features}_everything_"
        ext = ".h5"
        title = file_title + self.timestr_d + ext
        self.h5_file = self.modelfolder / title

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
        # file_title = f"{self.target}_{n_features}_everything_"
        # timestr = time.strftime("_%Y-%m-%d")
        # ext = ".h5"
        # title = file_title + timestr + ext
        # h5_file = self.modelfolder / title
        h5_file = self.h5_file
        shap_val_df = pd.DataFrame(self.shap_values)
        shap_feat_df = pd.DataFrame(self.features_shap)
        d = [["Expected Value:",self.shap_expected]]
        exp_df = pd.DataFrame(d, columns=("A","B"))
        # print(exp_df)
        shap_val_df.to_hdf(h5_file, key='shap_values', format="table")
        shap_feat_df.to_hdf(h5_file, key='features_shap', format="table")
        exp_df.to_hdf(h5_file, key = 'shap_expected_value', format='table')

    def save_requirements(self):
        try:
            cwd = os.getcwd() # record current workikng directory
            tmp_dir = config.PROJECT_DIR/"tmp/"
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)
            os.chdir(tmp_dir)
            jsonf = tmp_dir / "requirements.json"

            sys.stdout = open(jsonf, 'w') 
            # open the json file to write to
            reqs = conda.cli.main('conda', 'list', '--json', '-e') 
            # calls conda list, output as json, which is recorded by stdout
            sys.stdout.close() # close the redirected stdout
            sys.stdout = sys.__stdout__ 
            # restore the previous stdout.
            reqs = pd.read_json(jsonf)
            os.chdir(cwd) # back to the original directory
            shutil.rmtree(tmp_dir) # remove temp directory and files

            # file_title = f"{self.target}_{self.n_features}_everything_"
            # timestr = time.strftime("_%Y-%m-%d")
            # ext = ".h5"
            # title = file_title + timestr + ext
            # h5_file = self.modelfolder / title
            h5_file = self.h5_file
            reqs.to_hdf(h5_file, key='requirements', format="table")
            print("Requirements saved!")
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)

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
        df_shap_train.to_pickle(self.modelfolder / pickle_title)
        print("Pickle available at", self.modelfolder / pickle_title)
        imp_cols = pd.DataFrame(imp_cols)
        # print(imp_cols)
        # timestr = time.strftime("_%Y-%m-%d-%H%M_")
        timestr = self.timestr
        csv_title = f"{self.target}_{n_features}{timestr}shap.csv"
        imp_cols.to_csv(self.modelfolder / csv_title)
        print("CSV available at", self.modelfolder / csv_title)
        print("Saving SHAP df and important columns to .h5 file...")
        # file_title = f"{self.target}_{n_features}_everything_"
        # timestr = time.strftime("_%Y-%m-%d")
        # ext = ".h5"
        # title = file_title + timestr + ext
        # h5_file = self.modelfolder / title
        h5_file = self.h5_file
        df_shap_train.to_hdf(h5_file, key='df_shap_train', format="table")



        imp_cols.to_hdf(h5_file, key='ordered_shap_cols', format="table")
                
        # This part is just to make the names match the ugly half of the prettifying file        
        imp_cols = (imp_cols.progress_apply(
            lambda x: x.str.strip()
            .str.replace("\t", "")
            .str.replace("_", " ")
            .str.replace("__", " ")
            .str.replace(", ", " ")
            .str.replace(",", " ")
            .str.replace("'", "")
            .str.capitalize()
            ))
        # print(imp_cols)
        # Define and load the prettifying file
        prettycols_file = config.PRETTIFYING_COLUMNS_CSV
        prettycols = pd.read_csv(prettycols_file)

        # Now the magic happens
        # make a dict out of the old and new columns
        # and map the new onto the old
        di = dict(zip(prettycols.feature_ugly, prettycols.feature_pretty))
        # di = prettycols.set_index("feature_ugly").to_dict()
        # di = pd.Series(di)
        # print(di)
        # pretty_imp_cols = imp_cols[0].update(pd.Series(di))
        pretty_imp_cols = imp_cols[0].map(di).fillna(imp_cols[0])
        # self.features_shap.columns = self.features_shap.columns.to_series().map(
        #     di["feature_pretty"]
        # )
        # print(pretty_imp_cols)
        # select first column
        pretty_imp_cols.to_hdf(self.h5_file, key='pretty_imp_cols', format="table")
        
        print("Columns prettified and saved to h5!")

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
                timestr = self.timestr
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
            except Exception as exc:
                print(traceback.format_exc())
                print(exc)
                print(f"Plot for feature {i} failed, moving on...")

    def shap_top_dependence_plots_self(self, n_plots):
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

        # get list of most important variables, in order
        df_shap_train = pd.DataFrame(
            self.shap_values, columns=self.features_shap.columns.values
        )
        imp_cols = (
            df_shap_train.abs().mean().sort_values(ascending=False).index.tolist()
        )
        # imp_cols[i]
        for i in range(n_plots):
            try:
                shap.dependence_plot(
                    imp_cols[i],
                    self.shap_values,
                    df_with_codes,
                    # "rank(%d)" % i,
                    # here's the difference: set interaction index to the feature itself
                    interaction_index=imp_cols[i],
                    # interaction_index="rank(%d)" % i, #
                    display_features=self.features_shap,
                    show=False,
                    feature_names=self.features_shap.columns,
                )

                print(f"Making self dependence plot for {self.target} feature ranked {i}...")
                figure_title = f"{self.target}_SHAP_dependence_self_{i}_"
                # timestr = time.strftime("_%Y-%m-%d-%H%M_")
                timestr = self.timestr
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
            except Exception as exc:
                print(traceback.format_exc())
                print(exc)
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

        # Prepare to save to h5
        pretty_shap_col_series = self.features_shap.columns.to_series()
        pretty_shap_cols = pretty_shap_col_series.to_frame().reset_index()
        # select first column
        pretty_shap_cols = pretty_shap_cols[[0]]
        # print(pretty_shap_cols.head())
        pretty_shap_cols.to_hdf(self.h5_file, key='pretty_shap_cols', format="table")
        
        print("Columns prettified and saved to h5!")
        return self.features_shap

    def shap_plot_summaries(self, title_in_figure):
        n_features = self.n_features
        timestr = self.timestr
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
        figure_title = f"{self.target}_SHAP_summary_bar_"
        ext = ".png"
        title = figure_title + n_features + timestr + ext
        plt.savefig(
            (self.figfolder / title), dpi=1200, transparent=True, bbox_inches="tight"
        )
        plt.close()

        print(f"{self.target} Making SHAP summary bar plot PDF...")
        import cbh.texfig as texfig # set rcParams to texfig version...
        shap.summary_plot(
            self.shap_values,
            self.features_shap,
            title=title_in_figure,
            plot_type="bar",
            show=False,
            feature_names=self.features_shap.columns,
        )
        figure_title = f"{self.target}_SHAP_summary_bar_"
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
        # reset mpl backend and params...
        mpl.rcParams.update(mpl.rcParamsDefault)
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
        # timestr = time.strftime("_%Y-%m-%d-%H%M_")
        timestr = self.timestr
        ext = ".png"
        title = figure_title + n_features + timestr + ext
        plt.savefig(
            (self.figfolder / title), dpi=1200, transparent=True, bbox_inches="tight"
        )
        plt.close()

        print(f"{self.target} Making SHAP summary plot PDF...")
        import cbh.texfig as texfig # set rcParams to texfig version...
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
        # takes a couple minutes since SHAP interaction values take a factor of 2 * # features 
        # # more time than SHAP values to compute, since this is just an example we only explain
        # the first 2,000 people in order to run quicker
        print("Getting interaction values...")
        try:
            shap_int_vals = shap.TreeExplainer(self.model).shap_interaction_values(df_with_codes.iloc[:2000,:]) 
            # file_title = f"{self.target}_{self.n_features}_everything_"
            # timestr = time.strftime("_%Y-%m-%d")
            # ext = ".h5"
            # title = file_title + timestr + ext
            # h5_file = self.modelfolder / title
            h5_file = self.h5_file
            shap_int_vals_df = pd.DataFrame(shap_int_vals)
            shap_int_vals_df.to_hdf(h5_file, key = 'shap_int_vals', format='table')
            tmp = np.abs(shap_int_vals).sum(0)
            for i in range(tmp.shape[0]):
                tmp[i,i] = 0
            inds = np.argsort(-tmp.sum(0))[:50]
            tmp2 = tmp[inds,:][:,inds]
            pl.figure(figsize=(12,12))
            pl.imshow(tmp2)
            pl.yticks(range(tmp2.shape[0]), df_with_codes.columns[inds], rotation=50.4, horizontalalignment="right")
            pl.xticks(range(tmp2.shape[0]), df_with_codes.columns[inds], rotation=50.4, horizontalalignment="left")
            pl.gca().xaxis.tick_top()
            figure_title = f"{self.target}_SHAP_int_vals_heatmap_"
            try:
                title1 = figure_title + self.n_features + self.timestr
                title1 = str(self.figfolder) + "/" + title1
                texfig.savefig(title1, dpi=1200, transparent=True, bbox_inches="tight")
            except Exception as exc:
                print(traceback.format_exc())
                print(exc)
                print("Aww, LaTeX!..")
            plt.close()
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)

    def shap_random_force_plots(self, n_plots, expected_value):
        n_features = self.n_features
        timestr = self.timestr
        n_plots = n_plots
        # Needs explainer.expected_value
        expected_value = expected_value
        shap_indices = np.random.choice(
            self.shap_values.shape[0], n_plots
        )  # select 5 random patients
        for pt_num in shap_indices:
            # set params to default for png...
            mpl.rcParams.update(mpl.rcParamsDefault)
            mpl.rcdefaults()
            mpl.use("Qt5Agg")
            print("mpl backend for png:", mpl.get_backend())

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
            
            figure_title = f"{self.target}_SHAP_forceplot_Pt_{pt_num}_{n_features}{self.timestr}"
            # timestr = time.strftime("%Y-%m-%d-%H%M")
            ext = ".png"
            title = figure_title + ext
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
                print("mpl backend for svg:", mpl.get_backend())
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
                title0 = figure_title + ext0
                if not os.path.exists(forcefolder):
                    print("Making folder called", forcefolder)
                    os.makedirs(forcefolder)
                plt.savefig(
                    (forcefolder / title0), dpi=1200, transparent=True, bbox_inches="tight"
                )
                plt.close()

                # reset mpl params for LaTeX PDF via texfig.py
                import cbh.texfig as texfig 
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
                title1 = figure_title
                title1 = str(forcefolder) + "/" + title1
                print("mpl backend for PDF:", mpl.get_backend())
                print("The forceplots need the svg backend, even for PDF...")
                texfig.savefig(title1, dpi=1200, transparent=True, bbox_inches="tight")
            except Exception as exc:
                print(traceback.format_exc())
                print(exc)
                print("Aww, LaTeX!..")
            plt.close()
