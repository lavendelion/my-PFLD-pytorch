import torch
import numpy as np
import random
import os


def transform_oriData_to_selfData():
    listpath = r"I:\Dataset\WFLW\WFLW_for_PFLD2\testlist.txt"
    savepath = r"I:\Dataset\WFLW\WFLW_for_PFLD2"
    dir = "train" if "train" in listpath else "test"
    f_txt = open(os.path.join(savepath, dir+".txt"), "w")
    f_csv = open(os.path.join(savepath, dir+".csv"), "w")
    with open(listpath, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            imgname = line[0].split('\\')[-1]
            imgpath = os.path.join(os.path.join(savepath,dir),imgname)
            line = line[1:]
            line.extend(["0","0","0","0"])
            line[-3:] = line[-7:-4]
            line[-7:-3] = ["-1","-1","-1","-1"]
            f_txt.write(imgpath+'\n')
            f_csv.write(','.join(line)+'\n')
    f_csv.close()
    f_txt.close()

if __name__ == '__main__':
    transform_oriData_to_selfData()