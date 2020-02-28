# -*- coding: utf-8 -*-
from os.path import exists, join, dirname, basename
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import operator
import random
import string
import pickle
import glob
import time
import os
import re
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing import Process, Pool
import sklearn
import inspect
from methods import *
import functools
from numpy.random import uniform as uniform
import nltk
import datetime
import numpy as np
import pandas as pd

# =============================================================================
# CONSTANTES
# =============================================================================
from consts import *


#import methods
#ram = {k:{} for k in dir(methods)}


#def yolo(a,v):
#    print(list(vars().values()))
#    print(inspect.stack()[0][3])
#    
#    
#yolo(5, True)