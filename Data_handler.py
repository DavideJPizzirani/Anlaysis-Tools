

import os
import pandas as pd
import shutil
import os
import fnmatch
import re
import glob
import json
import natsort
from natsort import natsorted, ns
import sys

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def read_the_file(folder, pattern):
   for path, subdir, files in os.walk(folder):
    for file in files:
        if file.endswith(".dat") or file.endswith(pattern):
            filepath = os.path.join(path, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                 text = f.read()
                 #print('%s read' % filepath)
                 f.close()
                 print(file)


def read_copy_files(rootdir, destinationdir, pattern1, pattern2):
    if not os.path.exists(rootdir):
        print(bcolors.FAIL + "Root directory does not exist" + bcolors.ENDC)
    else:
        print(bcolors.OKGREEN + "Root directory does exist" + bcolors.ENDC)
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if not file.endswith('.csv') and not file.endswith(pattern1) and not file.endswith(pattern2):
                    pass
                else:
                    print(os.path.join(file))
    if not os.path.isdir(destinationdir):
        os.makedirs(destinationdir)
        print(bcolors.OKCYAN + "Destination directory does not exist, one has been created!" + bcolors.ENDC)
        for path, subdir, files in os.walk(rootdir):
            for name in files:
                if name.endswith(pattern1) or name.endswith(pattern2):
                    try:
                        file = os.path.join(path, name)
                        shutil.copy2(file, destinationdir)
                        print(bcolors.OKGREEN + "Files have been duplicated and moved to the directory destination."+ bcolors.ENDC) 
                        for name in file:
                            res = re.findall(".r(\d+).dat", name)
                            res.sort(key=lambda f: int(filter(str.isdigit, f)))
                        if not res: continue
                    except shutil.SameFileError:
                        pass
                        
def read_copy_sort_files(rootdir, destinationdir, pattern):
    if not os.path.exists(rootdir):
        print(bcolors.FAIL + "Root directory does not exist" + bcolors.ENDC)
    else:
        print(bcolors.OKGREEN + "Root directory does exist" + bcolors.ENDC)
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                if not file.endswith('.csv') and not file.endswith(pattern):
                    pass
                else:
                    print(os.path.join(file))
    if not os.path.isdir(destinationdir):
        os.makedirs(destinationdir)
        print(bcolors.OKCYAN + "Destination directory does not exist, one has been created!" + bcolors.ENDC)
        for path, subdir, files in os.walk(rootdir):
            for name in files:
                if name.endswith(pattern):
                    try:
                        file = os.path.join(path, name)
                        shutil.copy2(file, destinationdir)
                        print(bcolors.OKGREEN + "Files have been duplicated and moved to the directory destination."+ bcolors.ENDC) 
                        for name in file:
                            res = re.findall(".r(\d+).dat", name)
                            res.sort(key=lambda f: int(filter(str.isdigit, f)))
                        if not res: continue
                    except shutil.SameFileError:
                        pass


def change_comment_from_Claudius_tool(folder):
    os.chdir(folder)
    for files in os.listdir(folder):
        if not files.endswith(".dat"):
            print(bcolors.WARNING + "No .dat files here" + bcolors.ENDC)
        else: 
            #print(files)
            file = pd.read_csv(files, sep="\t")
            a = file["#freq"][0]
            b = file["ampl"][0]
            file.iloc[0][0] = b
            file.iloc[0][1] = a
            file.to_csv(files, sep="\t")
    print("Fixed the position of the field range comment!")
