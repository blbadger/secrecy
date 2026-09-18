import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
import os
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from safetensors.torch import load_file
import torch

load_dotenv()
checkpoint_root = os.getenv('CHECKPOINT_ROOT')
data_root = os.getenv('DATA_ROOT')

def single_attn_map():
	plt.xlabel("")
	plt.ylabel("")
	# plt.xticks([])
	# plt.yticks([])
	data = load_file(f'/home/bbadger/Desktop/untrained_attn_matrix_15.safetensors')
	data = torch.sqrt(torch.tensor(data['matrix']))
	plt.imshow(data, cmap='magma', interpolation='nearest', vmin=0, vmax=0.1)
	#plt.colorbar(label="Activation Value")
	plt.xlabel('token position', fontsize='large')
	plt.ylabel('token position', fontsize='large')
	plt.tick_params(labelsize=12)
	plt.show()
	plt.close()
	return

all_data = []
for i in range(16):
	try:
		data = load_file(f'/home/badger/attn_matrix_{i}.safetensors')
		data = (torch.tensor(data['matrix']))
		all_data.append(data)
	except:
		all_data.append(np.zeros((512, 512)))

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(6, 6))

# Flatten the axes array to easily iterate over it in 1D
axs = axs.flatten()
for i, ax in enumerate(axs):
    if i < 16:
        ax.imshow(all_data[i]**0.5, cmap='magma', vmin=0, vmax=0.1)
        ax.axis('off')  # Hide ticks and borders
        plt.tight_layout()
        plt.axis('off')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('/home/badger/attn_map.png')
