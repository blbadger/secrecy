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
mixer_ni_6_loss = [5.83585758, 5.47310582, 5.44318829, 5.42889326, 5.36313015, 5.33550104, 5.33220358, 5.31828008, 5.29482946, 5.29447356] #+ [5.29099252, 5.28661547, 5.28711, 5.27975144, 5.27102544, 5.2709374, 5.25826831, 5.22875727, 5.22616288, 5.21023661, 5.20579806, 5.19310712, 5.19761872, 5.18586844, 5.18798859]
mixer_clm_6_loss =[6.5450299, 5.99901157, 5.86456377, 5.79597212, 5.72897831, 5.6839584, 5.64814216, 5.61381636, 5.59011412, 5.5644269] #+ [5.55730499, 5.61659619, 5.5941415, 5.57703854, 5.55594504, 5.54446606, 5.53252003, 5.5102077, 5.49361238, 5.48295574, 5.46877997, 5.45861161, 5.44959714, 5.43909409, 5.43074586]

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

mixer_freeze_all_ni_loss = [4.2056073, 3.47238713, 3.29646414, 3.3200181, 3.38948328, 3.37627003, 3.32204053, 3.28678999, 3.4609375, 3.31549338]
mixer_freeze_all_clm_loss = [5.60989013, 4.8957429, 4.73662534, 4.65520788, 4.60283097, 4.55687072, 4.52499632, 4.4956804, 4.47113459, 4.44771145]

# no mlp, no mixer resids, freeze
mixer_freeze_2_ni_loss = [7.30691536, 7.31565322, 7.31597541, 7.31602661, 7.31608843, 7.31612215, 7.31614463, 7.31615087, 7.31609592, 7.31619395] # + [7.31621955, 7.31610716, 7.31601288, 7.31606658, 7.31615337, 7.31613901, 7.31612465, 7.31616648, 7.31597854, 7.31617148, 7.3162027, 7.3160697, 7.31615587, 7.31615212, 7.3162208, 7.31618147, 7.31622143, 7.31622268, 7.31624953, 7.31623766]
mixer_freeze_2_clm_loss = [6.6865865, 6.2578687, 6.09313122, 6.000128, 5.93998636, 5.89802914, 5.85945055, 5.83082865, 5.80559226, 5.7827036] # + [5.77301041, 5.86738981, 5.8396046, 5.82182105, 5.80401689, 5.78813089, 5.77308034, 5.76043498, 5.74871311, 5.73509244, 5.72379516, 5.70694196, 5.69508147, 5.68461652, 5.67538413, 5.66635093, 5.65903545, 5.65073092, 5.64288158, 5.63514901]

mixer_freeze_2_ni_loss_2 = [7.31621955, 7.31610716, 7.31601288, 7.31606658, 7.31615337, 7.31613901, 7.31612465, 7.31616648, 7.31597854, 7.31617148, 7.3162027, 7.3160697, 7.31615587, 7.31615212, 7.3162208, 7.31618147, 7.31622143, 7.31622268, 7.31624953, 7.31623766]
mixer_freeze_2_clm_loss_2 = [5.77301041, 5.86738981, 5.8396046, 5.82182105, 5.80401689, 5.78813089, 5.77308034, 5.76043498, 5.74871311, 5.73509244, 5.72379516, 5.70694196, 5.69508147, 5.68461652, 5.67538413, 5.66635093, 5.65903545, 5.65073092, 5.64288158, 5.63514901]

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

transformer_ni_loss = [0.04872876, 0.06363548, 0.068986, 0.07335183, 0.07534346, 0.07316657, 0.07962276, 0.07911607, 0.07966097, 0.07909201]
transformer_clm_loss = [3.03242075, 2.88793383, 2.81575157, 2.76875356, 2.73392106, 2.70551052, 2.68256973, 2.66329979, 2.64691396, 2.63601724]

transformer_c512_ni_loss = [2.36902745, 2.4256865, 2.48489559, 2.46276528, 2.31645089, 2.36790634, 2.34633814, 2.32907565, 2.23902491, 2.20509998]
transformer_c512_clm_loss = [5.24894164, 5.12491602, 4.78631826, 4.68173553, 4.62704117, 4.56279347, 4.49779462, 4.42498202, 4.40218827, 4.36954992]

transformer_c512_x2_ni_loss = [2.8586866, 3.36473298, 3.43293844, 4.24521459, 4.13980423, 3.39062875, 3.26257605, 2.94263882, 3.71512973, 3.3651045]
transformer_c512_x2_clm_loss = [5.70107634, 5.03530704, 4.84813954, 5.06654424, 5.43420004, 4.74324898, 4.74047383, 4.41747141, 4.29132127, 4.18050265]

transformer_permuted_ni_loss = [2.26086207, 2.81807715, 3.26099257, 2.62224577, 2.06849846, 2.05819694, 2.06311285, 1.8218913, 1.86598022, 1.81950999]
transformer_permuted_clm_loss = [6.76922529, 6.72343987, 6.71546066, 6.72478421, 6.7291113, 6.73173066, 6.73786915, 6.73743394, 6.74164864, 6.74234422]

transformer_c512_ni_ni_loss = [3.7399253, 2.24709014, 2.1867593, 2.01304529, 2.1672958, 2.14696298, 1.69152442, 1.53287844, 1.3717514, 1.32580972]
transformer_c512_ni_clm_loss = [6.07621558, 5.09096268, 5.0446693, 4.91072258, 4.99767629, 4.95780488, 4.61406518, 4.58004784, 4.40357663, 4.35565201]


plt.xlim(2.8, 7)

slopes, train_rates, intercepts = [], [], []



# df = pd.DataFrame()
# df['NI Loss'] = transformer_ni_4_loss
# df['CLM Loss'] = transformer_clm_4_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-4 residuals')
# slope, intercept, r_value, p_value, std_err = stats.linregress(transformer_ni_4_loss, transformer_ni_4_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(transformer_ni_4_loss[0] - transformer_ni_4_loss[-1])

# df = pd.DataFrame()
# df['NI Loss'] = transformer_ni_5_loss
# df['CLM Loss'] = transformer_clm_5_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-5 residuals')
# slope, intercept, r_value, p_value, std_err = stats.linregress(transformer_ni_5_loss, transformer_ni_5_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(transformer_ni_5_loss[0] - transformer_ni_5_loss[-1])

# df = pd.DataFrame()
# df['NI Loss'] = transformer_ni_6_loss
# df['CLM Loss'] = transformer_clm_6_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-6 residuals')
# slope, intercept, r_value, p_value, std_err = stats.linregress(transformer_ni_6_loss, transformer_ni_6_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(transformer_ni_6_loss[0] - transformer_ni_6_loss[-1])

# df = pd.DataFrame()
# df['NI Loss'] = transformer_permuted_ni_loss
# df['CLM Loss'] = transformer_permuted_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, fit_reg=False, label='Permuted input')
# slope, intercept, r_value, p_value, std_err = stats.linregress(transformer_ni_4_loss, transformer_ni_4_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(transformer_permuted_ni_loss[0] - transformer_permuted_ni_loss[-1])

# df = pd.DataFrame()
# df['NI Loss'] = transformer_c512_ni_loss
# df['CLM Loss'] = transformer_c512_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Embedding c=512')
# slope, intercept, r_value, p_value, std_err = stats.linregress(transformer_c512_ni_loss, transformer_c512_clm_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(transformer_c512_ni_loss[0] - transformer_c512_ni_loss[-1])


# df = pd.DataFrame()
# df['NI Loss'] = transformer_c512_ni_ni_loss
# df['CLM Loss'] = transformer_c512_ni_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Embedding c=512, +Ni')
# slope, intercept, r_value, p_value, std_err = stats.linregress(transformer_c512_ni_ni_loss, transformer_c512_ni_clm_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(transformer_c512_ni_ni_loss[0] - transformer_c512_ni_ni_loss[-1])

# df = pd.DataFrame()
# df['NI Loss'] = transformer_c512_x2_ni_loss
# df['CLM Loss'] = transformer_c512_x2_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Embedding c=512 x2')
# slope, intercept, r_value, p_value, std_err = stats.linregress(transformer_c512_x2_ni_loss, transformer_c512_x2_clm_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(transformer_c512_x2_ni_loss[0] - transformer_c512_x2_ni_loss[-1])

# df = pd.DataFrame()
# df['NI Loss'] = ni_loss
# df['CLM Loss'] = clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', ci=None, truncate=False, label='Transformer, -1 residual')

##################################################################################################################

df = pd.DataFrame()
df['NI Loss'] = mixer_freeze_2_ni_loss
df['CLM Loss'] =mixer_freeze_2_clm_loss
sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-2 all residuals (frozen)')
slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_freeze_2_ni_loss, mixer_freeze_2_ni_loss)
slopes.append(slope)
intercepts.append(intercept)
train_rates.append(mixer_freeze_2_clm_loss[0] - mixer_freeze_2_clm_loss[-1])

df = pd.DataFrame()
df['NI Loss'] = mixer_freeze_2_ni_loss_2
df['CLM Loss'] =mixer_freeze_2_clm_loss_2
sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-2 all residuals (frozen)')
slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_freeze_2_ni_loss_2, mixer_freeze_2_ni_loss_2)
slopes.append(slope)
intercepts.append(intercept)
train_rates.append(mixer_freeze_2_clm_loss[0] - mixer_freeze_2_clm_loss[-1])


df = pd.DataFrame()
df['NI Loss'] = mixer_ni_6_loss 
df['CLM Loss'] = mixer_clm_6_loss 
sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-6 residuals')
slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_ni_6_loss, mixer_ni_6_loss)
slopes.append(slope)
intercepts.append(intercept)
train_rates.append(mixer_clm_6_loss[0] - mixer_clm_6_loss[-1])

df = pd.DataFrame()
df['NI Loss'] = mixer_ni_6_loss_2
df['CLM Loss'] = mixer_clm_6_loss_2 
sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-6 residuals')
slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_ni_6_loss, mixer_ni_6_loss)
slopes.append(slope)
intercepts.append(intercept)
train_rates.append(mixer_clm_6_loss[0] - mixer_clm_6_loss[-1])



# df = pd.DataFrame()
# df['NI Loss'] = mixer_freeze_all_ni_loss
# df['CLM Loss'] =mixer_freeze_all_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-1 all residuals (frozen)')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_freeze_all_ni_loss, mixer_freeze_all_ni_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_freeze_clm_loss[0] - mixer_freeze_clm_loss[-1])

# df = pd.DataFrame()
# df['NI Loss'] = mixer_freeze_ni_loss
# df['CLM Loss'] = mixer_freeze_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-1 residual (frozen)')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_freeze_ni_loss, mixer_freeze_ni_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_freeze_clm_loss[0] - mixer_freeze_clm_loss[-1])


# df = pd.DataFrame()
# df['NI Loss'] = mixer_ni_5_loss
# df['CLM Loss'] = mixer_clm_5_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-5 residuals')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_ni_5_loss, mixer_ni_5_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_clm_5_loss[0] - mixer_clm_5_loss[-1])



# df = pd.DataFrame()
# df['NI Loss'] = mixer_ni_loss
# df['CLM Loss'] = mixer_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-4 residuals')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_ni_loss, mixer_ni_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_clm_loss[0] - mixer_clm_loss[-1])

# df = pd.DataFrame()
# df['NI Loss'] = mixer_freeze_pre_ni_loss
# df['CLM Loss'] = mixer_freeze_pre_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-1 prefix resid (frozen)')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_freeze_pre_ni_loss, mixer_freeze_pre_clm_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_freeze_pre_clm_loss[0] - mixer_freeze_pre_clm_loss[-1])


# df = pd.DataFrame()
# df['NI Loss'] = mixer_ni_3_loss
# df['CLM Loss'] = mixer_clm_3_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='-3 residuals')

# df = pd.DataFrame()
# df['NI Loss'] = mixer_noni_ni_loss
# df['CLM Loss'] = mixer_noni_clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', truncate=False, label='Mixer')
# slope, intercept, r_value, p_value, std_err = stats.linregress(mixer_noni_ni_loss, mixer_noni_clm_loss)
# slopes.append(slope)
# intercepts.append(intercept)
# train_rates.append(mixer_freeze_clm_loss[0] - mixer_freeze_clm_loss[-1])


plt.xticks(size='large')
plt.yticks(size='large')
plt.ylabel('Invertibility Loss', size='large')
plt.xlabel('Causal Loss', size='large')
plt.legend(fontsize='large')
plt.show()
plt.close()

model = LinearRegression()
slope, intercept, r_value, p_value, std_err = stats.linregress(clm_loss, ni_loss)
print (slope, intercept, r_value**2, p_value, std_err)


# plt.scatter(train_rates, intercepts)
# plt.show()
# plt.close()
