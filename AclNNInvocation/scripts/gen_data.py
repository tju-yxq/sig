#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gen_golden_data_simple():
    input_x = np.random.uniform(-3, 3, [8, 2048]).astype(np.float16)
    golden = sigmoid(input_x)

    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
