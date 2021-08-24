#!/usr/bin/env python
# coding: utf-8
import torch

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x ** 3

dydx = torch.autograd.grad(y, x, grad_outputs=torch.ones(x.shape), create_graph=True,
                           retain_graph=True)

print(dydx)

dydx2 = torch.autograd.grad(dydx, x, grad_outputs=torch.ones(x.shape),create_graph=True)

print(dydx2)

dydx2 = torch.autograd.grad(dydx2, x,create_graph=True)

print(dydx2)



