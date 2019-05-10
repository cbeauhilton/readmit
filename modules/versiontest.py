from subprocess import call
import fileinput
import os

if os.name == 'nt':
    call("bash", shell=True)
call("conda list > requirements.txt", shell=True)

for line in fileinput.input("requirements.txt", inplace=True):
    # inside this loop the STDOUT will be redirected to the file
    # the comma after each print statement is needed to avoid double line breaks
    print(line.replace("# Name", "Name"))

call("cat requirements.txt | tr -s '[:blank:]' ',' > ofile.csv", shell=True)
call("rm requirements.txt", shell=True)
call("rm ofile.csv", shell=True)
