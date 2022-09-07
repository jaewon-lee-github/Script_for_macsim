#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import datetime
import numpy as np

from joblib import Parallel, delayed

macsim_path = Path('/fast_data/jaewon/GPU_SCA/macsim')
result_path = macsim_path / 'bin' / 'dvfs_test'

def run_program(frq):
    new_dir = result_path / (str(frq)+"GH")    
    os.chdir(str(new_dir))
    print("CUR_DIR =%s"%(new_dir))
    os.system("./macsim --clock_gpu=%d"%(frq))
    return 1

def main():
    frq_list = [1, 1.5, 2]
    results = Parallel(n_jobs=-1, verbose=10)(delayed(run_program)(frq)
                                             for frq in frq_list)

if __name__ == '__main__':
    main()
