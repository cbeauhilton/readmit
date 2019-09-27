import schedule
import time


def runall():
    import sys
    import os

    sys.path.append("modules")
    from datetime import datetime

    startTime = datetime.now()
    from cbh import config

    print("About to run", os.path.basename(__file__))

    """ 

    TODO AND NOTES 

        TABLEONE 
        - [DONE] for paper: full + LoS binary 7 + readmitted30d (so 2+5 cols), make CSVs and stitch, then load df and --> LaTeX 

        REFACTOR 
        - 
        
        FEATURE ENGINEERING/CLEANING
        - !!! drop labs not available within 24ish hours for LoS problems (bool column, then mask?) (would probs be small %)
        - Time until death:
        - - select dispo of "expired"
        - - discharge minus admission
        - - get rid of negative LoS
        
        MODELS
        - CV for all models reported on
        - [DONE]: Tune important classifications (esp calibration). NOTE: tried both main calibration algos, was not useful.
        - Tune important regressions. 
        - This is fantastic: https://blog.dominodatalab.com/shap-lime-python-libraries-part-2-using-shap-lime/
        - ^ modification of dependency plots to show where an individual prediction lies within the distribution 

        FIGURES
        - Prediction % histograms
        - Confusion matrices
        - Readmission timing histogram (see methods section)
        - MAYBE: A bunch more force diagrams?
        - MAYBE: The combo force diagrams as HTML?
        - MAYBE: Exploratory data analysis diagrams? (Also maybe as HTML?)
        - MAYBE: PR curves

        TODO: (just so the search will work)

        - Discussion: can increase AUC if add in transfer patients
        - 


    """

    """ Preprocessing """
    # import aageodata00unzip
    # import aageodata01download
    # import aageodata02get_noaa
    # import aaprelim06dxcodes
    # import aaziptopickle
    # import abclean00
    # import abclean01
    # import abclean02
    # import abclean03
    # import abclean04geo

    # # # import abclean05xx # placeholder
    # import aaprelim06dxcodes

    # # # import abclean07xx # placeholder
    # # # import abclean08xx # placeholder
    # import abclean09categories
    # import abclean10split
    # import abclean11tableone
    # import zzdeidentify
    # # # import abclean12targetedpickling

    # # # """ Pictures and tables """
    # # # import ca00tableoneexplore
    # # # import ca01tableonepaper
    # # # import ca02plotly
    # import ca03numberspaper

    # """ Classifiers """

    import acclassifier07shapit

    # from acclassifier01 import classifiermany

    # test_targets = config.CLASSIFIER_TEST_TARGETS
    # debugfraction = 0.015  # default is 0.005
    # class_thresh = 0.26
    # debug=False
    # # default is 0.15. ###UPDATE: added automatic thresh calculation in loop. May need adjusting.

    # classifiermany(
    #     test_targets,
    #     debugfraction,
    #     class_thresh,
    #     debug=debug,
    #     trainmodels=True,
    #     generatemetrics=True,
    #     generateshap=True,
    # )

    # import acclassifier06readmit

    #     """ Regressors """
    # from baregressor00 import regressormany

    # test_targets = config.REGRESSOR_TEST_TARGETS
    # debugfraction = 0.1  # default is 0.005

    # regressormany(
    #     test_targets,
    #     debugfraction,
    #     debug=False,
    #     trainmodels=True,
    #     generateshap=True,
    #     generatemetrics=True,
    # )

    # import acclassifier02cvtargeted

    # How long did this take?
    print("This program,", os.path.basename(__file__), "took")
    print(datetime.now() - startTime)
    print("to run.")


runall()

# """

# Uncomment out the rest of the lines to enable the program to run at midnight +1min

# """

#     return schedule.CancelJob


# schedule.every().day.at("00:01").do(runall)

# while True:
#     schedule.run_pending()
#     time.sleep(1)

