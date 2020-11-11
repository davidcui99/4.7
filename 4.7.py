import pandas as pd
import numpy as np
from scipy.stats import stats
import matplotlib.pyplot as plt


dataf = pd.read_csv('m_ret_10stocks.txt', sep="\t", header=None, index_col=0)
data = pd.read_csv('m_sp500ret_3mtcm.txt', sep="\t", header=None)
data=data.dropna(axis='columns')

