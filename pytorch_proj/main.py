# import torch
#
# form = torch.arange(20)
# form = form.view(4,5)
# print(form)
# a = torch.arange(20,40)
# print(a)
# a = a.view_as(form)
# print(a)
# import os.path
#
# path1 = os.path.dirname(__file__)
# path1 = os.path.join(path1,"YOLO_V3_detector","cfg", "yolo_v3.cfg")
# print(path1)
# with open(path1, "r") as f:
#     for line in f:
#         print(line)
# import numpy as np
# import torch
# a = np.array([[1,1],[1,2], [1,3]])
# a = torch.from_numpy(a)
# # x = (a[:,1]>2).float().unsqueeze(1)
# x = (a[:,1]>2).unsqueeze(1)
# print(x)
# import numpy
# import numpy as np
# import torch

# # a = torch.from_numpy(np.arange(10).reshape((5,2)))
# # a = a.view((2,5))
# # print(a)
# # a = a.squeeze(1)
# # print(a)
#
# # a = torch.tensor([.1, .2, .3])
# # print(a)
# # b = a.new(a.shape)
#
# # a = torch.arange(9)
# # a = a.view(3,3)
# # b = a[:,0:1]
# # print(b)
# # c = a[:, 0]
# # print(c)
# # a = torch.tensor([])
# # c = torch.arange(27)
# # c = c.view(3,3,3)
# # print(a.shape)
# # d = c[a, :]
# # print(d)
# # print(d.shape)
# a = torch.tensor([.5])
# b = torch.arange(10)/10
# c = torch.max(a, b)
# print(c)
import time
import pickle

import torch
import torch.nn as nn
import torchvision


a = torch.Tensor([2])
a.requires_grad=True
b = torch.Tensor([3])
b.requires_grad=True
z = a*b
z.backward()
print(a.grad)