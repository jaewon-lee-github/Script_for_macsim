#!/usr/bin/env python3

import itertools
import re
import os
import glob
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from pprint import pprint

"""
CAT 1 = ALU Operation
CAT 2 = L1_HIT_GPU
CAT 3 = TOTAL_DRAM_MERGE
CAT 4 = Total Instruction
total cycles
"""
re_INST = re.compile(r'PARENT_UOP_CORE_[0-9]+[0-9]*')
re_SEND = re.compile(r'OP_CAT_GED_SEND[CS]*[CS]*_CORE_[0-9]+[0-9]*')
re_L1_MISS = re.compile(r'L1_MISS_GPU')
re_DRAM = re.compile(r'TOTAL_DRAM_MERGE')
re_CYCLE = re.compile(r'CYC_COUNT_TOT')
# re_INST = re.compile(r'INST_COUNT_TOT')

# {"file to search" : [[counter nubmer, Regular expression],...],...}
pattern_dict = {
    "inst.stat.out": [[0, re_INST], [1, re_SEND], ],
    "memory.stat.out": [[2, re_L1_MISS], [3, re_DRAM]],
    "general.stat.out": [[4, re_CYCLE]]  # 5, Frequency
}

now = datetime.datetime.now()
date = '%s%d' % (now.strftime("%b").lower(), now.day)
np.set_printoptions(suppress=True)

macsim_path = Path('/fast_data/jaewon/GPU_SCA/macsim')
result_path = macsim_path / 'bin' / 'dvfs_test'

inst_start = 3000
inst_period = 1000
num_sample = 2
# frq_list = ['1', '1.5', '2']
frq_list = [1, 2]
stat_list = ["INST", "SEND", "L1_MISS", "DRAM", "CYCLE"]
type_list = ["ALU", "L1_MISS", "DRAM", "INST"]
insts_list = [i for i in range(
    inst_start, inst_start+inst_period*(num_sample-1)+1, inst_period)]

# 5 counters for each files
out = []
counters = np.zeros([len(frq_list), len(insts_list), len(stat_list)])
counters_fixed = np.zeros([len(frq_list), len(insts_list), len(type_list)])


for frq, i in enumerate(frq_list):
    new_dir = str(result_path)+"/"+str(i)+"GH"
    os.chdir(new_dir)
    print("DIR: %s" % (new_dir))
    # counters[frq, :, -1] = i
    for inst, j in enumerate(insts_list):
        for k in pattern_dict.keys():
            for f in glob.glob('%s.inst_%d' % (k, j)):
                with open(f) as tfile:
                    for line in tfile:
                        for d in pattern_dict[k]:
                            if (d[1].search(line)):
                                counters[frq][inst][d[0]] = counters[frq][inst][d[0]] + \
                                    int(line.split()[1])

# stat_list = ["INST", "SEND", "L1_MISS", "DRAM", "CYCLE" ]

print(stat_list)
print(counters)

# counters_fixed [freq,inst,type]
counters_fixed[:, :, 0] = counters[:, :, 0] - counters[:, :, 1]
counters_fixed[:, :, 1:3] = counters[:, :, 2:4]
counters_fixed[:, :, 3] = counters[:, :, 0]
print(type_list)
print(counters_fixed)

"""
Power_cur = (Summation (Const_n*F_cur*(Event_n_cur)))/(cycle/F_cur)) + c_offset_list*F_cur
power_next = (Summation (Const_n*F_next*(Event_n_next)))/(cycle/F_next)) + c_offset_list*F_next
"""
# [From Hz, to Hz], scenario length should be same with num_sample
scenario_list = [[1, 2], [2, 1]]
power_dif_list = []
c_max = 10
c_start = 1
c_interval = 1
c_array = np.arange(c_start, c_max+c_interval, c_interval).reshape((1, -1))


def x(start, max):
    return np.arange(start, max+1, start)


# type_list = ["ALU", "L1_MISS", "DRAM", "INST"]
# c_matrix = np.array(np.meshgrid(x(c_interval, c_max), x(
#     c_interval, c_max*2), x(c_interval, c_max*3), x(c_interval, c_max*4), x(c_interval, c_max*5)),).T
c_matrix = np.array(np.meshgrid(
    c_array, c_array, c_array, c_array, c_array)).T.reshape(-1, 5)
# c_matrix[:, [0, 1]] = c_matrix[:, [1, 0]]


"""
First objective: when we change frequency from 1GHz to 2GHz, or 2GHz -> 1GHz
what constant will remain the power same.
1. The counter can determine consumed energy.
    - If we divide the energy by time,which is cycle over Frequency, Then it is definitely power
2. Get Freq from Scenario_x[0]
3. From 1000 to n*1000, get the power number from the equation above
4. power equation
    Power_cur = (Summation(Const_n*F_cur*(Event_n_cur)))/(cycle/F_cur)) + c_offset_list*F_cur
    power_next= (Summation(Const_n*F_next*(Event_n_next)))/(cycle/F_next)) + c_offset_list*F_next
5. search the index fulfill the c
"""

def est_power(frq_list, stat_list, insts_list, counters, counters_fixed, inst, c_matrix, cur_freq):
    result_cur = c_matrix[:, 0:-1] * (cur_freq ** 2)
    result_cur = result_cur * \
        counters_fixed[frq_list.index(cur_freq), insts_list.index(inst), :]
    result_cur = result_cur / \
        counters[frq_list.index(cur_freq), insts_list.index(
            inst), stat_list.index("CYCLE")]
    result_cur = np.add.reduce(result_cur, -1, keepdims=True)
    result_cur = result_cur + c_matrix[:, np.newaxis, -1]*(cur_freq)
    return result_cur


os.chdir(str(result_path))
for scn_idx, scenario in enumerate(scenario_list):
    for inst_idx, inst in enumerate(insts_list):
        cur_freq = scenario[inst_idx]
        result_cur = est_power(frq_list, stat_list, insts_list, counters,
                               counters_fixed, inst, c_matrix, cur_freq)
        if inst_idx == 0:
            prev_freq = cur_freq
            result_prev = result_cur
            continue
        power_diff = (result_cur-result_prev).reshape(-1)
        # find the criteria
        criteria = np.min(np.absolute(power_diff)) + 1
        idx_power_diff = np.array(np.nonzero(
            np.absolute(power_diff) <= criteria)).reshape(-1)

        # union the power diff and const matrix
        res_matrix = np.concatenate(
            (c_matrix, power_diff.reshape((-1, 1))), axis=1)

        with open("test_%d_%d_%d.out" % (prev_freq, cur_freq, inst), 'w', encoding="utf-8") as f:
            np.savetxt(f, res_matrix,
            # np.savetxt(f, res_matrix[idx_power_diff, :],
                       delimiter=',', fmt='%10.5f',
                       header="%10s,%10s,%10s,%10s,%10s,%10s\n" %
                       ("c_ALU", "c_L1_MISS", "c_DRAM", "c_INST", "c_OFFSET", "DIFF"))
        prev_freq = cur_freq
        result_prev = result_cur
