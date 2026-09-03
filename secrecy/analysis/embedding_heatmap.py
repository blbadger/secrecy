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
path = "{data_root}/fineweb-edu-encodings-s0-overfit-tagged-c16/secret_{i}"
num_samples = 100
all_data = []
for i in tqdm(range(num_samples)):
	dataset = load_from_disk(path.format(data_root=data_root, i=i))
	#data = [i for i in dataset['encodings'][0] if len(i) > 0] # first sample is always the same for non-randomized eval
	data = dataset['encodings'][0][100:103]
	data = np.array(data).flatten()
	all_data.append(data)

all_data = np.array(all_data)
plt.imshow(all_data, cmap='magma', interpolation='nearest')
#plt.colorbar(label="Activation Value")
plt.savefig(f"{data_root}/embedding_figure", dpi=350, bbox_inches='tight')
plt.close()
