import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from gan import Generator, Discriminator

df = pd.read_csv('anomaly_score')
