import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
import os
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

#path = "{data_root}/fineweb-edu-encodings-emb_clmoverfit/secret_{i}"
path = "{data_root}/fineweb-edu-encodings-untrained/secret_{i}"
num_samples = 100
all_data = []
for i in tqdm(range(num_samples)):
	dataset = load_from_disk(path.format(data_root=data_root, i=0))
	#data = [i for i in dataset['encodings'][0] if len(i) > 0] # first sample is always the same for non-randomized eval
	data = dataset['encodings'][i][-3:]
	data = np.array(data).flatten()
	all_data.append(data)

#path = "{data_root}/fineweb-edu-encodings-s0-overfit-tagged-c16/secret_{i}" # NI training only
path = "{data_root}/fineweb-edu-encodings-s0-clmoverfit-78ths-tagged-c16/secret_{i}" # NI -> NI + CLM path
all_data2 = []
for i in tqdm(range(num_samples)):
	dataset = load_from_disk(path.format(data_root=data_root, i=4))
	data = dataset['encodings'][i][-3:]
	data = np.array(data).flatten()
	all_data2.append(data)

all_data = np.array(all_data)
all_data2 = np.array(all_data2)

plt.imshow(all_data2, cmap='magma', interpolation='nearest')
#plt.colorbar(label="Activation Value")
plt.savefig(f"{data_root}/embedding_figure", dpi=350, bbox_inches='tight')
plt.close()
