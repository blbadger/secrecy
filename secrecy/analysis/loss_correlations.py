from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

ni_loss  = [0.624, 3.977, 1.207, 2.636, 5.943, 0.028, 7.324, 7.06]
clm_loss = [3.364, 5.120, 3.701, 4.89, 6.413, 2.94, 7.324, 6.91]
# model = ['mixer', 'mixer', 'mixer', 'mixer', 'transformer', 'transformer', 'transformer']


df = pd.DataFrame()
df['NI Loss'] = ni_loss
df['CLM Loss'] = clm_loss
sns.regplot(df, x='CLM Loss', y='NI Loss', ci=None, truncate=False, label='Constant Compute')

plt.ylabel('Invertibility Loss', size='large')
plt.xlabel('Causal Loss', size='large')
plt.legend()
plt.show()
plt.close()

model = LinearRegression()
slope, intercept, r_value, p_value, std_err = stats.linregress(clm_loss, ni_loss)
print (slope, intercept, r_value**2, p_value, std_err)

# -1 starting attn residual
ni_loss = [0.3431576, 0.3431576, 0.2873463, 0.25231738, 0.20882913, 0.08983468, 0.09646272, 0.08005682, 0.08005682, 0.03755606, 0.12716336, 0.10418744, 0.03733898, 0.03375178, 0.07548156, 0.10171961, 0.10171961, 0.05287834, 0.04054369]
clm_loss = [3.69493973, 3.69493973, 3.41125239, 3.30953566, 3.25862859, 3.24962161, 3.2101595, 3.1210004, 3.1210004, 3.064386, 3.04281296, 3.02583457, 3.01107687, 2.99907152, 2.98633281, 2.97633772, 2.97633772, 2.95745397, 2.94992058]

# -5 mid residuals on mixer layers
mixer_ni_5_loss = [4.35704442, 3.79608451, 3.71395866, 3.51673269, 3.27492295, 3.15070564, 3.01807671, 2.93672512, 2.86960893, 2.84140787]
mixer_clm_5_loss = [5.69162354, 5.02482429, 4.87277869, 4.71879777, 4.56487054, 4.47185171, 4.37776766, 4.31578747, 4.26553009, 4.23872489]

# -4 mid residuals on mixer layers
mixer_ni_loss = [3.5970203, 3.03196619, 2.80099323, 2.73049685, 2.50274096, 2.47404929, 2.42798148, 2.32931308, 2.20422629, 2.15162335, 2.13235653, 2.05064451, 2.03772691, 2.05203302, 2.06818532, 1.9945084, 1.97235154, 1.94725876, 1.94244463]
mixer_clm_loss = [5.29924647, 4.76861494, 4.55057208, 4.44850006, 4.31072795, 4.25121727, 4.20152747, 4.13909335, 4.06328175, 4.01141623, 3.98802993, 3.94258293, 3.92873747, 3.9143644, 3.90246925, 3.86988116, 3.85652805, 3.8406536, 3.8311293]

# -1 middle mixer resid, freeze
mixer_freeze_ni_loss = [3.88620986, 3.33546483, 3.22782473, 3.20427421, 3.22402431, 3.20081534, 3.20081534, 3.14708474, 3.16827752, 3.18280326]
mixer_freeze_clm_loss = [5.51618821, 4.91352177, 4.779423, 4.70101765, 4.64750714, 4.60507725, 4.60507725, 4.57443105, 4.51951658, 4.49760043]

plt.xlim(2.8, 6)

df = pd.DataFrame()
df['NI Loss'] = mixer_ni_5_loss
df['CLM Loss'] = mixer_clm_5_loss
sns.regplot(df, x='CLM Loss', y='NI Loss', ci=None, truncate=False, label='Mixer, -5 residuals')

df = pd.DataFrame()
df['NI Loss'] = mixer_ni_loss
df['CLM Loss'] = mixer_clm_loss
sns.regplot(df, x='CLM Loss', y='NI Loss', ci=None, truncate=False, label='Mixer, -4 residuals')

df = pd.DataFrame()
df['NI Loss'] = mixer_freeze_ni_loss
df['CLM Loss'] = mixer_freeze_clm_loss
sns.regplot(df, x='CLM Loss', y='NI Loss', ci=None, truncate=False, label='Mixer, -1 residual (frozen)')

df = pd.DataFrame()
df['NI Loss'] = ni_loss
df['CLM Loss'] = clm_loss
sns.regplot(df, x='CLM Loss', y='NI Loss', ci=None, truncate=False, label='Transformer, -1 residual')

plt.ylabel('Invertibility Loss', size='large')
plt.xlabel('Causal Loss', size='large')
plt.legend()
plt.show()
plt.close()

model = LinearRegression()
slope, intercept, r_value, p_value, std_err = stats.linregress(clm_loss, ni_loss)
print (slope, intercept, r_value**2, p_value, std_err)
