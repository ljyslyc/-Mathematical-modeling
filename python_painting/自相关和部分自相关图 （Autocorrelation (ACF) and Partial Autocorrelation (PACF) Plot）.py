
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Import Data

df = pd.read_csv('https://github.com/selva86/datasets/raw/master/AirPassengers.csv')

# Draw Plot

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6), dpi= 80)

plot_acf(df.traffic.tolist(), ax=ax1, lags=50)

plot_pacf(df.traffic.tolist(), ax=ax2, lags=20)

# Decorate

# lighten the borders

ax1.spines["top"].set_alpha(.3); ax2.spines["top"].set_alpha(.3)

ax1.spines["bottom"].set_alpha(.3); ax2.spines["bottom"].set_alpha(.3)

ax1.spines["right"].set_alpha(.3); ax2.spines["right"].set_alpha(.3)

ax1.spines["left"].set_alpha(.3); ax2.spines["left"].set_alpha(.3)

# font size of tick labels

ax1.tick_params(axis='both', labelsize=12)

ax2.tick_params(axis='both', labelsize=12)

plt.show()