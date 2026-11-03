from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

ni_loss  = [0.624, 3.977, 1.207, 2.636, 5.943]
clm_loss = [3.364, 5.120, 3.701, 4.89, 6.413]

df = pd.DataFrame()
df['NI Loss'] = ni_loss
df['CLM Loss'] = clm_loss
sns.regplot(df, x='CLM Loss', y='NI Loss', ci=None)
plt.ylabel('Invertibility Loss', size='large')
plt.xlabel('Causal Loss', size='large')
plt.show()
plt.close()

model = LinearRegression()
slope, intercept, r_value, p_value, std_err = stats.linregress(clm_loss, ni_loss)
print (slope, intercept, r_value**2, p_value, std_err)
