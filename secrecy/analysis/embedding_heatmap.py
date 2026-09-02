import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk

dataset = load_from_disk()
path = "{data_root}/fineweb-edu-encodings-emb_clmoverfit/secret_{}"
num_samples = 100
all_data = []
for i in range(num_samples):
	dataset = load_from_disk(path.format(data_root=data_root, i=i))
	data = np.array(dataset['encodings'][0]) # first sample is always the same for non-randomized eval
	data = data.flatten()
	all_data.append(data)

all_data = np.array(all_data)
plt.imshow(all_data, cmap='magma', interpolation='nearest')
plt.figsize(15, 15)
plt.colorbar(label="Activation Value")
plt.savefig(f"{data_root}/embedding_figure", dpi=350, bbox_inches='tight')
plt.close()