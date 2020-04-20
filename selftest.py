import redis
import base64
import pprint
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

a = [[[[1, 2], 
     [5, 2]],
     [[0, 6], 
     [8, 9]]]]
a = torch.Tensor(a).cuda()
print(a)
b = torch.Tensor([0, 0, 0, 0]).long().cuda()
c = a[0, b, 0, 0]
a[0, b, 1, 1] = 1
print("a[0]: ", a)
# print("a[0][b]: ", a[0][b])
# print(c)
print(b)