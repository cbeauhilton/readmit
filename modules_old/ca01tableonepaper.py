import os
import sys

# sys.path.append("modules")
from datetime import datetime

import pandas as pd
from plotly.offline import iplot
from tableone import TableOne
import time
from cbh import config
import io

print("About to run", os.path.basename(__file__))
startTime = datetime.now()


# display all columns when viewing dataframes: make the number
# anything bigger than your number of columns
pd.options.display.max_columns = 2000

print("Loading data files...")
figures_path = config.FIGURES_DIR
tables_path = config.TABLES_ONE_DIR
# data = pd.read_pickle(tables_path, "finaltable.pickle")

# if you need to generate the files, then import ca00tableoneexplore.py
# import ca00tableoneexplore

first_file = os.path.join(tables_path, "_tableone_.csv")
second_file = os.path.join(tables_path, "readmitted30d_tableone_.csv")
third_file = os.path.join(tables_path, "length_of_stay_over_7_days_tableone_.csv")
first = pd.read_csv(first_file)
second = pd.read_csv(second_file)
third = pd.read_csv(third_file)

merge1 = first.merge(
    second,
    left_on=["variable", "level"],
    right_on=["Unnamed: 0", "Unnamed: 1"],
    how="outer",
)
merge2 = merge1.merge(
    third,
    left_on=["variable", "level"],
    right_on=["Unnamed: 0", "Unnamed: 1"],
    how="outer",
)

data = merge2

data = data.drop(
    ["Unnamed: 0_y", "Unnamed: 1_y", "Unnamed: 0_x", "Unnamed: 1_x"], axis=1
)
data = data.drop(
    [
        3,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
        42,
        44,
        46,
        48,
        50,
        52,
        54,
        56,
        58,
        60,
        62,
        64,
        66,
        68,
        71,
        73,
        97,
        100,
        101,
        102,
    ],
    axis=0,
)

data = data.reset_index(drop=True)

data = data.rename(
    index=str,
    columns={
        "Grouped by 30-day readmissions": "Not readmitted within 30 days",
        "Grouped by 30-day readmissions.1": "Readmitted within 30 days",
        "Grouped by length_of_stay_over_7_days": "Hospital stay less than 7 days",
        "Grouped by length_of_stay_over_7_days.1": "Hospital stay over 7 days",
        "overall": "Overall",
        "variable": "Characteristic",
    },
)
data = data.fillna(value="")
# is_duplicate = data.Characteristic.apply(pd.Series.duplicated(keep="first"), axis=1)
# data = data.where(~is_duplicate, "")
is_duplicate = pd.Series(data.Characteristic).duplicated(keep="first")
# print(is_duplicate)
data.Characteristic = data.where(~is_duplicate, "")
data["level"] = data["level"].apply(lambda x: "\hspace{3mm}" + f" {x}")

data = data.rename(index=str, columns={"level": ""})
# data.rename(columns=lambda x: f"\textbf{{{x}}} ", inplace=True)

data.replace(
    {
        r"\hspace{3mm} 1.0": "",
        r"\hspace{3mm} 1": "",
        r"\hspace{3mm} Y": "",
        r"\hspace{3mm}": "",
        r"\hspace{3mm} ": "",
        # r"\hspace{3mm} Y": "",
        # r"\hspace{3mm} Y": "",
    },
    inplace=True,
)
# print(data.index)
x = [0,1,2,3,6,11,15,36,38,42,50,62,62]
rows = []
cur = {}
# print(list(data))
for i in data.index.astype(int):
    if i in x:
        cur['index'] = i
        cur['Characteristic'] = data.iloc[i]['Characteristic']*0
        cur['Overall'] = data.iloc[i]['Overall']*0
        cur['Not readmitted within 30 days'] = data.iloc[i]['Not readmitted within 30 days']*0
        cur['Readmitted within 30 days'] = data.iloc[i]['Readmitted within 30 days']*0
        cur['Hospital stay less than 7 days'] = data.iloc[i]['Hospital stay less than 7 days']*0
        cur['Hospital stay over 7 days'] = data.iloc[i]['Hospital stay over 7 days']*0
        rows.append(cur)
        cur = {}

print(rows)
offset = 0; #tracks the number of rows already inserted to ensure rows are inserted in the correct position

for d in rows:
    data = pd.concat([data.head(d['index'] + offset), pd.DataFrame([d]), data.tail(len(data) - (d['index']+offset))],sort=False)
    offset+=1


data.reset_index(inplace=True)
data.drop('index', axis=1, inplace=True)
data.drop('level_0', axis=1, inplace=True)

data

data.Characteristic = data.Characteristic.str.replace("%", r"\%")
data = data.fillna(value="")
print(list(data.columns.values))
cols = list(data.columns.values)
data.set_index("Characteristic", inplace=True)
# data = data.replace('%', '\%')
print(data.head())
finaltable = data

final_file = os.path.join(tables_path, "finaltable")
final_file_pickle = str(final_file) + ".pickle"
final_file_csv = str(final_file) + ".csv"
final_file_html = str(final_file) + ".html"
finaltable.to_pickle(final_file_pickle)
finaltable.to_csv(final_file_csv)
finaltable.to_html(final_file_html)

tex_tables_path = config.TEX_TABLE_DIR
final_file_tex_fol = os.path.join(tex_tables_path, "finaltable")
final_file_tex = str(final_file_tex_fol) + ".tex"

# finaltable.to_latex(final_file_tex,index_names=False,index=False,escape=False,column_format="L{20em}C{10em}C{10em}C{10em}C{10em}C{10em}")

# with open(final_file_tex, 'r') as file :
#   filedata = file.read()
# # Replace the % with escaped \%
# filedata = filedata.replace('%', '\%')
# # Write the file
# with open(final_file_tex, 'w') as file:
#   file.write(filedata)

# # Add more header stuff
# with open(final_file_tex, 'r') as file :
#   filedata = file.read()
tex_line_one = r"\rowcolors{3}{}{NEJMAlternatingRows}%" + "\n"
tex_line_two = (
    r"\begin{adjustbox}{width={\textwidth},totalheight={\textheight},keepaspectratio,frame,padding=0ex 0ex -910ex 0ex}%"
    + "\n"
)
tex_line_thr = r"\sffamily %" + "\n"
tex_line_for = r"{%" + "\n"
tex_line_fve = r""
tex_lines = tex_line_one + tex_line_two + tex_line_thr + tex_line_for + tex_line_fve
# with open(final_file_tex, 'w') as modified:
#     modified.write(tex_lines + filedata)

# print(final_file_tex)


def convertToLaTeX(data):
    """
    Convert a pandas dataframe to a LaTeX tabular.
    Prints labels in bold, does not use math mode
    """
    numColumns = data.shape[1]
    numRows = data.shape[0]
    output = io.StringIO()
    colFormat = "L{20em}C{10em}C{10em}C{10em}C{10em}C{10em}"
    # Write header
    output.write(tex_lines)
    output.write("\\begin{tabular}{%s" % colFormat)
    output.write("}%\n")
    output.write(r"\rowcolor{NEJMTopRow} \multicolumn{6}{l}%" + "\n")
    output.write(r"{{\textbf{\color{NEJMRed} Table 1.}}%" + "\n")
    output.write(
        r"\textbf{Characteristics  of Hospital Encounters in the Study Sample, Overall and According to Readmission and Extended Length of Stay.}}\\[10pt]%"
    )
    output.write("\n" + r"\hline%" + "\n")
    output.write(r" & & & & & \\%" + "\n")
    columnLabels = ["\\textbf{%s}" % label for label in data.columns]
    output.write("& %s\\\\\\hline\n" % " & ".join(columnLabels))
    # Write data lines
    for i in range(numRows):
        output.write(
            "%s & %s\\\\\n" % (data.index[i], " & ".join([str(val) for val in data.ix[i]]))
        )
    # Write footer
    output.write("\\end{tabular}" + "\n")
    output.write(r"\label{table:table1}" + "\n")
    output.write(r"}" + "\n")
    output.write(r"\end{adjustbox}" + "\n")
    output.write(r"")
    return output.getvalue()


filedata = convertToLaTeX(finaltable)
with open(final_file_tex, "w") as file:
    file.write(filedata)

print(final_file_tex)
# The to_latex command is not quite plug-and-play.

# Using Tex Live via Atom/VSCode (whatever) Latex package, make sure you have the following to get a usable PDF:

# \documentclass{article}
# \usepackage{booktabs, adjustbox}
# \usepackage[T1]{fontenc}
# \usepackage{lmodern}
# \begin{document}
# \begin{adjustbox}{width={\textwidth},totalheight={\textheight},keepaspectratio}%
# \begin{tabular}{llllll}
# the number of "l"s is autogenerated. It's the number of left-justified columns.

# and at the bottom:

# \bottomrule
# \end{tabular}
# \end{adjustbox}
# \end{document}


print("Done.")

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
