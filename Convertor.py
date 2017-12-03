#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os.path import exists, dirname

data_file = sys.argv[1]
path = dirname(data_file)

if not exists(data_file):
    raise Exception("ERROR: file {} does not exist!!".format(data_file))

with open(data_file, "r", encoding="utf-8") as fin, open(path + ".txt", "w", encoding="utf-8") as fout,\
        open(path + "_tagged.txt", "w", encoding="utf-8") as ftagged:
    lines = fin.readlines()
    index = 0
    length = len(lines)
    while index < length:
        if not lines[index].startswith("#"):
            if len(lines[index]) > 1:
                tmp = lines[index].split()
                fout.write(tmp[1] + " ")
                ftagged.write(tmp[1] + "/" + tmp[3] + " ")
            else:
                fout.write("\n")
                ftagged.write("\n")
        index += 1

