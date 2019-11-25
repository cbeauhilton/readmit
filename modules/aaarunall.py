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


    # """ Classifiers """

    import acclassifier07shapit


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

