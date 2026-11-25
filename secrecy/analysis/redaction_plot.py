from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

redacted = [0, 5, 10, 20, 30, 40, 50, 90, 100]  # percent redacted

no_mask   = [2.573, 2.596, 2.632, 2.678, 2.712, 2.729, 2.734, 2.742, 2.746]
attn_mask = [None, 2.541, 2.596, 2.661, 2.705, 2.737, 2.740, 2.746, None]
routed    = [2.519, 2.554, 2.514, 2.552, 2.605, 2.647, 2.675, 2.713, None]

plt.plot(redacted, no_mask)
plt.scatter(redacted, no_mask)
plt.plot(redacted, attn_mask)
plt.scatter(redacted, attn_mask)
plt.plot(redacted, routed)
plt.scatter(redacted, routed)
plt.xticks(size='large')
plt.yticks(size='large')
plt.ylabel('Causal Loss', size='large')
plt.xlabel('Percent Redacted', size='large')
plt.show()
plt.close()

