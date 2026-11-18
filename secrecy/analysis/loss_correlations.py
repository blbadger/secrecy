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
sns.regplot(df, x='CLM Loss', y='NI Loss', ci=None, truncate=False)


plt.xticks(size='large')
plt.yticks(size='large')
plt.ylabel('Invertibility Loss', size='large')
plt.xlabel('Causal Loss', size='large')
# plt.legend()
plt.show()
plt.close()

model = LinearRegression()
slope, intercept, r_value, p_value, std_err = stats.linregress(clm_loss, ni_loss)
print (slope, intercept, r_value**2, p_value, std_err)

# -1 starting attn residual
ni_loss = [0.3431576, 0.3431576, 0.2873463, 0.25231738, 0.20882913, 0.08983468, 0.09646272, 0.08005682, 0.08005682, 0.03755606, 0.12716336, 0.10418744, 0.03733898, 0.03375178, 0.07548156, 0.10171961, 0.10171961, 0.05287834, 0.04054369]
clm_loss = [3.69493973, 3.69493973, 3.41125239, 3.30953566, 3.25862859, 3.24962161, 3.2101595, 3.1210004, 3.1210004, 3.064386, 3.04281296, 3.02583457, 3.01107687, 2.99907152, 2.98633281, 2.97633772, 2.97633772, 2.95745397, 2.94992058]

# - 6 mid residuals on mixer layers
mixer_ni_6_loss = [5.83585758, 5.47310582, 5.44318829, 5.42889326, 5.36313015, 5.33550104, 5.33220358, 5.31828008, 5.29482946, 5.29447356]
mixer_clm_6_loss =[6.5450299, 5.99901157, 5.86456377, 5.79597212, 5.72897831, 5.6839584, 5.64814216, 5.61381636, 5.59011412, 5.5644269]

mixer_ni_6_loss_2 = [5.29099252, 5.28661547, 5.28711, 5.27975144, 5.27102544, 5.2709374, 5.25826831, 5.22875727, 5.22616288, 5.21023661, 5.20579806, 5.19310712, 5.19761872, 5.18586844, 5.18798859]
mixer_clm_6_loss_2 = [5.55730499, 5.61659619, 5.5941415, 5.57703854, 5.55594504, 5.54446606, 5.53252003, 5.5102077, 5.49361238, 5.48295574, 5.46877997, 5.45861161, 5.44959714, 5.43909409, 5.43074586]

# -5 mid residuals on mixer layers
mixer_ni_5_loss = [4.35704442, 3.79608451, 3.71395866, 3.51673269, 3.27492295, 3.15070564, 3.01807671, 2.93672512, 2.86960893, 2.84140787]
mixer_clm_5_loss = [5.69162354, 5.02482429, 4.87277869, 4.71879777, 4.56487054, 4.47185171, 4.37776766, 4.31578747, 4.26553009, 4.23872489]

# -3 mid residuas
mixer_ni_3_loss = [1.08053128, 0.31325414, 0.23987513, 0.2323378, 0.23440786, 0.23591155, 0.23522577, 0.23229984, 0.20338247, 0.21183855]
mixer_clm_3_loss = [4.12452639, 3.43946468, 3.29833266, 3.23800995, 3.1978023, 3.16576025, 3.13904308, 3.11653625, 3.09204258, 3.07465208]

# -4 mid residuals on mixer layers
mixer_ni_loss = [3.5970203, 3.03196619, 2.80099323, 2.73049685, 2.50274096, 2.47404929, 2.42798148, 2.32931308, 2.20422629, 2.15162335, 2.13235653, 2.05064451, 2.03772691, 2.05203302, 2.06818532, 1.9945084, 1.97235154, 1.94725876, 1.94244463]
mixer_clm_loss = [5.29924647, 4.76861494, 4.55057208, 4.44850006, 4.31072795, 4.25121727, 4.20152747, 4.13909335, 4.06328175, 4.01141623, 3.98802993, 3.94258293, 3.92873747, 3.9143644, 3.90246925, 3.86988116, 3.85652805, 3.8406536, 3.8311293]

mixer_noni_ni_loss = [0.00209649, 0.00241484, 0.00298, 0.00314176, 0.00336851, 0.00347814, 0.00347814, 0.00349792, 0.00373779, 0.003714]
mixer_noni_clm_loss = [3.54868795, 3.28132524, 3.18692243, 3.13168077, 3.09349556, 3.06358521, 3.06358521, 3.01590036, 2.99720456, 2.98108472]


# -1 middle mixer resid, freeze
mixer_freeze_ni_loss = [3.88620986, 3.33546483, 3.22782473, 3.20427421, 3.22402431, 3.20081534, 3.20081534, 3.14708474, 3.16827752, 3.18280326]
mixer_freeze_clm_loss = [5.51618821, 4.91352177, 4.779423, 4.70101765, 4.64750714, 4.60507725, 4.60507725, 4.57443105, 4.51951658, 4.49760043]

mixer_freeze_pre_ni_loss = [6.51312615, 4.88263767, 3.78363802, 3.04002845, 2.49370214, 2.12120037, 1.82501595, 1.62369063, 1.46974561, 1.3944852]
mixer_freeze_pre_clm_loss = [6.7426814, 5.77077693, 5.11410827, 4.67280754, 4.36276425, 4.14131309, 3.98072257, 3.86422847, 3.77850732, 3.71751525]

transformer_ni_4_loss = [1.68140663, 0.14027626, 0.21827042, 0.23797719, 0.39922073, 0.49603439, 0.51218736, 0.54728207, 0.53386425, 0.56575594] + [0.55166919, 0.49977533, 0.56048022, 0.5496327, 0.56586806, 0.52113467, 0.6181051, 0.6558642, 0.64556241, 0.65247538, 0.66117949, 0.63915243, 0.66178726, 0.6427665, 0.65263125, 0.6425085, 0.6357055, 0.60242175, 0.62878804, 0.61471441]
transformer_clm_4_loss = [5.33631651, 3.7729077, 3.51361662, 3.42266861, 3.29930985, 3.25099654, 3.18802668, 3.15786251, 3.13230236, 3.11305803] + [3.10698136, 3.16131108, 3.1290124, 3.10497797, 3.07824738, 3.06055811, 3.04391503, 3.0300371, 3.01841076, 3.00763829, 2.99722797, 2.98784823, 2.98010191, 2.97212208, 2.96485187, 2.95782486, 2.9519939, 2.94633558, 2.94084117, 2.93614599]



transformer_ni_5_loss = [0.30150969, 0.10281681, 0.10325098, 0.18710661, 0.19567754, 0.20648424, 0.3994749, 0.40984296, 0.3933475, 0.17643956]
transformer_clm_5_loss = [3.69486949, 3.33648791, 3.20563447, 3.13573251, 3.08873856, 3.05707521, 3.0291158, 3.00744161, 2.98918663, 2.97316763]

transformer_ni_6_loss = [0.75125442, 2.37095482, 1.02416266, 0.88408253, 0.73489294, 1.10367071, 1.10367071, 0.80007685, 0.7348376, 0.7348376]
transformer_clm_6_loss =[3.84067077, 5.42513125, 4.31245161, 3.81797662, 3.84067046, 4.24333171, 4.24333171, 3.88550398, 3.78519528, 3.78519528]

transformer_ni_2all_freeze_loss = [2.37095482, 1.02416266, 0.85736053, 0.88408253, 0.73489294, 1.10367071, 1.11214094, 0.80007685, 0.7348376, 0.77026679]
transformer_clm_2all_freeze_loss = [5.42513125, 4.31245161, 3.9838249, 3.81797662, 3.84067046, 4.24333171, 4.22753157, 3.88550398, 3.78519528, 3.70488737]

transformer_ni_2nomid_freeze_loss = [0.02251758, 0.01966827, 0.02088618, 0.02185705, 0.02184726, 0.02313531, 0.02257063, 0.02264056, 0.02262765, 0.0068888]
transformer_clm_2nomid_freeze_loss = [3.58815631, 3.20182687, 3.09678521, 3.04322881, 3.00320692, 2.97464372, 2.95223086, 2.93202963, 2.91561882, 2.90102127]

plt.xlim(2.8, 7)

slopes, train_rates, intercepts = [], [], []
df = pd.DataFrame()
df['NI Loss'] = transformer_ni_4_loss
df['CLM Loss'] = transformer_clm_4_loss
sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Transformer, -4 residuals')
slope, intercept, r_value, p_value, std_err = stats.linregress(transformer_ni_4_loss, transformer_ni_4_loss)
slopes.append(slope)
intercepts.append(intercept)
train_rates.append(transformer_ni_4_loss[0] - transformer_ni_4_loss[-1])

slopes, train_rates, intercepts = [], [], []
df = pd.DataFrame()
df['NI Loss'] = transformer_ni_5_loss
df['CLM Loss'] = transformer_clm_5_loss
sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Transformer, -5 residuals')
slope, intercept, r_value, p_value, std_err = stats.linregress(transformer_ni_5_loss, transformer_ni_5_loss)
slopes.append(slope)
intercepts.append(intercept)
train_rates.append(transformer_ni_5_loss[0] - transformer_ni_5_loss[-1])

slopes, train_rates, intercepts = [], [], []
df = pd.DataFrame()
df['NI Loss'] = transformer_ni_6_loss
df['CLM Loss'] = transformer_clm_6_loss
sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Transformer, -6 residuals')
slope, intercept, r_value, p_value, std_err = stats.linregress(transformer_ni_6_loss, transformer_ni_6_loss)
slopes.append(slope)
intercepts.append(intercept)
train_rates.append(transformer_ni_6_loss[0] - transformer_ni_6_loss[-1])

df = pd.DataFrame()
df['NI Loss'] = ni_loss
df['CLM Loss'] = clm_loss
sns.regplot(df, x='CLM Loss', y='NI Loss', ci=None, truncate=False, label='Transformer, -1 residual')

# df = pd.DataFrame()
# df['NI Loss'] = mixer_ni_6_loss + mixer_ni_6_loss_2
# df['CLM Loss'] = mixer_clm_6_loss + mixer_clm_6_loss_2
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Mixer, -6 residuals')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_ni_6_loss, mixer_ni_6_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_clm_6_loss[0] - mixer_clm_6_loss[-1])


# df = pd.DataFrame()
# df['NI Loss'] = mixer_freeze_ni_loss
# df['CLM Loss'] = mixer_freeze_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Mixer, -1 residual (frozen)')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_freeze_ni_loss, mixer_freeze_ni_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_freeze_clm_loss[0] - mixer_freeze_clm_loss[-1])


# df = pd.DataFrame()
# df['NI Loss'] = mixer_ni_5_loss
# df['CLM Loss'] = mixer_clm_5_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Mixer, -5 residuals')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_ni_5_loss, mixer_ni_5_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_clm_5_loss[0] - mixer_clm_5_loss[-1])

# df = pd.DataFrame()
# df['NI Loss'] = mixer_ni_6_loss_2
# df['CLM Loss'] = mixer_clm_6_loss_2
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Mixer, -6 residuals')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_ni_6_loss_2, mixer_ni_6_loss_2)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_clm_6_loss_2[0] - mixer_clm_6_loss_2[-1])

# df = pd.DataFrame()
# df['NI Loss'] = mixer_ni_loss
# df['CLM Loss'] = mixer_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Mixer, -4 residuals')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_ni_loss, mixer_ni_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_clm_loss[0] - mixer_clm_loss[-1])

# slopes, train_rates, intercepts = [], [], []
# df = pd.DataFrame()
# df['NI Loss'] = mixer_freeze_pre_ni_loss
# df['CLM Loss'] = mixer_freeze_pre_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Mixer, -1 resid (frozen) prefixed')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_freeze_pre_ni_loss, mixer_freeze_pre_clm_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_freeze_pre_clm_loss[0] - mixer_freeze_pre_clm_loss[-1])


# df = pd.DataFrame()
# df['NI Loss'] = mixer_ni_3_loss
# df['CLM Loss'] = mixer_clm_3_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Mixer, -3 residuals')

# df = pd.DataFrame()
# df['NI Loss'] = mixer_noni_ni_loss
# df['CLM Loss'] = mixer_noni_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Mixer')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_freeze_ni_loss, mixer_freeze_ni_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_freeze_clm_loss[0] - mixer_freeze_clm_loss[-1])


plt.xticks(size='large')
plt.yticks(size='large')
plt.ylabel('Invertibility Loss', size='large')
plt.xlabel('Causal Loss', size='large')
plt.legend()
plt.show()
plt.close()

model = LinearRegression()
slope, intercept, r_value, p_value, std_err = stats.linregress(clm_loss, ni_loss)
print (slope, intercept, r_value**2, p_value, std_err)


# plt.scatter(train_rates, intercepts)
# plt.show()
# plt.close()
