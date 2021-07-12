import torch
import torch.nn as nn
from kmeans_pytorch import kmeans, kmeans_predict
import operator
import argparse
from utils import getData, getModel, setup
from plot_weights import plot_all_weights