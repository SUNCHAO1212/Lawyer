# -*- coding:UTF-8 -*-
# !/usr/bin/env python3

import json
import pandas as pd

column = ["text", "label"]
dataset = []
with open("rt-polaritydata/rt-polarity.pos") as f:
    for i, line in enumerate(f):
        dataset.append([line.strip(), 1])
        print(i)
with open("rt-polaritydata/rt-polarity.neg") as f:
    for i, line in enumerate(f):
        dataset.append([line.strip(), 0])
        print(i)
data_frame = pd.DataFrame(data=dataset, columns=column)
data_frame.to_csv(path_or_buf="training_data.tsv", index=False, sep='\t')
