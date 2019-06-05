from subprocess import call
import fileinput
import os
import pandas as pd

if os.name == 'nt':
    call("bash", shell=True)
call("conda list > requirements.txt", shell=True)

for line in fileinput.input("requirements.txt", inplace=True):
    # inside this loop the STDOUT will be redirected to the file
    # the comma after each print statement is needed to avoid double line breaks
    print(line.replace("# Name", "Name"))

call("cat requirements.txt | tr -s '[:blank:]' ',' > ofile.csv", shell=True)

reqs = pd.read_csv("ofile.csv", ) #skiprows=2
reqs = reqs[1:]
cols = [4,5] # empty columns
reqs.drop(reqs.columns[cols],axis=1,inplace=True)
reqs = reqs.rename(columns=reqs.iloc[0])
reqs = reqs[["Name", "Version", "Build", "Channel"]]
print(reqs.head)

os.remove("requirements.txt")
os.remove("ofile.csv")

# If on Unix, could do this instead:
# call("rm requirements.txt", shell=True)
# call("rm ofile.csv", shell=True)
