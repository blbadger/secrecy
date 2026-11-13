import os
import torch
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn

device = 'cuda'
def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		nn.GELU(),
		nn.Linear(inner_dim, dim)
	)

def ConvForward(dim, expansion_factor=1):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Conv1d(dim, inner_dim, 1),
		nn.GELU(),
		nn.Conv1d(inner_dim, dim, 1)
		)

class DoubleMixerBlock(nn.Module):

	def __init__(self, dim, length, clm_mask=False, expand_conv=False):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernormf = nn.LayerNorm(dim)
		self.seq_layernormr = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			self.convf = nn.Conv1d(length, length, 1)
			self.convr = nn.Conv1d(length, length, 1)
		self.clm_mask = clm_mask
		self.expand_conv = expand_conv
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x: torch.tensor, y: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')
			y = rearrange(y, 'b p t f -> (b p) t f')

		# for CLM training, apply lower triangular mask to convolution weights
		if self.clm_mask:
			if self.expand_conv:
				rearranged_shape = rearrange(self.conv[0].weight, 'f d p -> f (d p)').shape
				mask = torch.tril(torch.ones(rearranged_shape)).to(device)
				applied_mask = rearrange(self.conv[0].weight, 'f d p -> f (d p)') * mask
				self.conv[0].weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

				rearranged_shape = rearrange(self.conv[2].weight, 'f d p -> f (d p)').shape
				mask = torch.tril(torch.ones(rearranged_shape)).to(device)
				applied_mask = rearrange(self.conv[2].weight, 'f d p -> f (d p)') * mask
				self.conv[2].weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

			else:
				rearranged_shape = rearrange(self.conv.weight, 'f d p -> f (d p)').shape
				mask = torch.tril(torch.ones(rearranged_shape)).to(device)
				applied_mask = rearrange(self.conv.weight, 'f d p -> f (d p)') * mask
				self.conv.weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

		else:

			masked_convf = torch.tril(rearrange(self.convf.weight, 'f d p -> p f d'))
			self.convf.weight.data = rearrange(masked_convf, 'p f d -> f d p').contiguous()

			masked_convr = torch.triu(rearrange(self.convr.weight, 'f d p -> p f d'), diagonal=2)
			self.convr.weight.data = rearrange(masked_convr, 'p f d -> f d p').contiguous()

		residualf, residualr = x, y
		x, y = self.seq_layernormf(x), self.seq_layernormr(y)
		x, y = self.convf(x) + residualf, self.convr(y) + residualr
		residualf, residualr = x, y
		x, y = self.patch_layernorm(x), self.patch_layernorm(y)
		x, y = self.patch_ff(x) + residualf, self.patch_ff(y) + residualr
		return x, y

class HeadedRepeatCausalLinear(nn.Module):
	"""
	Mixed-headed repeat module for ParallelRepeatHeads
	"""

	def __init__(self, dim: int, heads: int, decay=False, decay_constant=1):

		super().__init__()

		# Standard weight + bias
		self.weight = nn.Parameter(torch.randn(heads, dim))
		self.bias = nn.Parameter(torch.zeros(heads, dim))
		self.heads = heads
		if decay:
			self.decay_value = nn.Parameter(torch.ones(2, 1))
			self.decay_constant = decay_constant
		else:
			self.decay_value = None

	def vector_to_matrix(self, v: torch.Tensor) -> torch.Tensor:
		"""
		Given a matrix v of shape (k, m) and head number h >= 0, 
		returns an (k x m x m) matrix M where M[i, j] = v[i] if 
		j >= i, and 0 otherwise.

		For example, if v = [[a, b, c, d], [e, f, g, h]], k=2 then M will be:

		[[
			[ a  a  a  a ]
			[ 0  b  b  b ]
			[ 0  0  c  c ]
			[ 0  0  0  d ]
		],
		[
			[ a  b  c  d ]
			[ 0  b  c  d ]
			[ 0  0  c  d ]
			[ 0  0  0  d ]
		]]

		"""
		# Expects v is a preformed tensor with shape [k, D]
		m = v.shape[-1] # vector shape

		# Create index grids for rows and columns
		i, j = torch.meshgrid(
			torch.arange(m, device=v.device),
			torch.arange(m, device=v.device),
			indexing="ij",
		)
		if self.decay_value is not None:
			M = torch.where(
				j >= i, v[:self.heads//2, j]*(torch.clip(self.decay_value[1], min=0.9, max=1)**((j-i)/self.decay_constant)), torch.zeros(m, m, device=v.device, dtype=v.dtype)
			)
			M = torch.cat((M, torch.where(
				j >= i, v[self.heads//2:, i]*(torch.clip(self.decay_value[1], min=0.9, max=1)**((j-i)/self.decay_constant)), torch.zeros(m, m, device=v.device, dtype=v.dtype)
			)), dim=0)
		else:
			M = torch.where(
				j >= i, v[:self.heads//2, j], torch.zeros(m, m, device=v.device, dtype=v.dtype)
			)
			M = torch.cat((M, torch.where(
				j >= i, v[self.heads//2:, i], torch.zeros(m, m, device=v.device, dtype=v.dtype)
			)), dim=0)

		return M

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x.to(device) # x has shape [b * h, e, t]
		# W has shape [h, t, t] and is repeated to make [b * h, t, t]
		W = self.vector_to_matrix(self.weight).repeat(x.shape[0]//self.heads, 1, 1) 
		output = torch.bmm(x, W)
		repeated_bias = self.bias.repeat(x.shape[0]//self.heads, 1) # [h, t] repeated to [b * h, t]
		repeated_bias = repeated_bias.unsqueeze(1).repeat(1, x.shape[1], 1) # repeated to [b * h, e, t]
		output += repeated_bias
		return output

class MixerHead(nn.Module):

	def __init__(self, dim, length, hidden_dim, n_heads):
		super().__init__()
		self.n_heads = n_heads
		self.proj_head = nn.ModuleList(
			[nn.Linear(dim, hidden_dim)
			for i in range(n_heads)]
			)

		self.convs = nn.ModuleList(
			[nn.Conv1d(length, length, 1)
			for i in range(n_heads)]
			)

		self.out_proj = nn.Linear(dim, dim)
		self.GeLU = nn.GELU()

	def forward(self, x: torch.tensor):
		hidden_layer = []

		for head in range(self.n_heads):
			weight = self.convs[head].weight
			masked_weight = torch.tril(rearrange(weight, 'f d p -> p f d'))
			masked_weight = rearrange(masked_weight, 'p f d -> f d p').contiguous()
			
			projection = self.proj_head[head](x)
			conv_projection = F.conv1d(projection, masked_weight, bias=self.convs[head].bias)
			hidden_layer.append(conv_projection)

		# concatenate and project multi-headed output
		hidden_layer = torch.cat(hidden_layer, dim=2)
		hidden_layer = self.out_proj(hidden_layer)
		return hidden_layer

class MixerBlock(nn.Module):

	def __init__(self, dim, length=512, expand_conv=False, kernel=1, n_heads=0, mixer_residual=True, mlp_residual=True, recurrent=False, r_decay=1.):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			if n_heads > 0:
				if recurrent:
					decay = r_decay < 1.
					self.conv = HeadedRepeatCausalLinear(dim, n_heads, decay=decay, decay_constant=r_decay)
				else:
					self.conv = MixerHead(dim, length, dim//n_heads, n_heads)
			elif n_heads == 0:
				self.conv = nn.Conv1d(length, length, kernel, padding='same')
		self.n_heads = n_heads
		self.expand_conv = expand_conv
		self.mixer_residual = mixer_residual
		self.mlp_residual = mlp_residual

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		# for CLM training, apply lower triangular mask to convolution weights
		if self.n_heads==0 and self.expand_conv:
			rearranged_shape = rearrange(self.conv[0].weight, 'f d p -> f (d p)').shape
			mask = torch.tril(torch.ones(rearranged_shape)).to(device)
			applied_mask = rearrange(self.conv[0].weight, 'f d p -> f (d p)') * mask
			self.conv[0].weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

			rearranged_shape = rearrange(self.conv[2].weight, 'f d p -> f (d p)').shape
			mask = torch.tril(torch.ones(rearranged_shape)).to(device)

			applied_mask = rearrange(self.conv[2].weight, 'f d p -> f (d p)') * mask
			self.conv[2].weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

		elif self.n_heads==0:
			masked_conv = torch.tril(rearrange(self.conv.weight, 'f d p -> p f d'))
			self.conv.weight.data = rearrange(masked_conv, 'p f d -> f d p').contiguous()

		residual = x
		x = self.seq_layernorm(x)
		x = self.conv(x)
		if self.mixer_residual:
			x += residual
		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x)
		if self.mlp_residual:
			x += residual
		return x


class LanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, length, tie_weights=False, n_heads=0, kernel=1, recurrent=False, r_decay=1.):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = length,
				expand_conv = False,
				n_heads = n_heads,
				kernel = kernel,
				recurrent=recurrent,
				r_decay=r_decay
				)
			for i in range(depth)]
			)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		if tie_weights:
			self.wte.weight = self.lm_head.weight
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
		if inputs_embeds is None:
			x = input_ids
			x = self.wte(x)
		else:
			x = inputs_embeds
		for i, block in enumerate(self.mixerblocks):
			x = block(x)
		output = self.lm_head(x)
		
		output = rearrange(output, 'b t e -> b e t')
		if labels is not None:
			if labels.dim() > 2:
				labels = rearrange(labels, 'b p t -> b (p t)')
			shift_logits = output[..., :-1].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			loss = self.cel(shift_logits, shift_labels)
		else:
			loss = 0
		return loss, output

class InverterMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, length, tie_weights=False, n_heads=0, kernel=1):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = length,
				expand_conv = False,
				n_heads = n_heads,
				kernel = kernel
				)
			for i in range(depth)]
			)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		if tie_weights:
			self.wte.weight = self.lm_head.weight
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
		if inputs_embeds is None:
			x = input_ids
			x = self.wte(x)
		else:
			x = inputs_embeds
		for i, block in enumerate(self.mixerblocks):
			x = block(x)
		output = self.lm_head(x)
		
		output = rearrange(output, 'b t e -> b e t')
		if labels is not None:
			if labels.dim() > 2:
				labels = rearrange(labels, 'b p t -> b (p t)')
			loss = self.cel(output, labels)
		else:
			loss = 0
		return loss, output

class SplitLanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, length, tie_weights=False, n_heads=0, kernel=1, split_layer=None, mlp_residual=True, mixer_residual=True, recurrent=False, r_decay=1.):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = length,
				expand_conv = False,
				n_heads = n_heads,
				kernel = kernel,
				mlp_residual = mlp_residual,
				mixer_residual = mixer_residual,
				recurrent=recurrent,
				r_decay=r_decay
				)
			for i in range(depth)]
			)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		if tie_weights:
			self.wte.weight = self.lm_head.weight
		self.cel = nn.CrossEntropyLoss()
		self.split_layer = split_layer
		if recurrent:
			self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, HeadedRepeatCausalLinear):
				# Kaiming He initialization for Swish activation
				nn.init.kaiming_normal_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)

	def forward(self, input_ids, labels=None, **kwargs):
		x = input_ids
		x = self.wte(x)

		split_output = None
		for i, block in enumerate(self.mixerblocks): 
			if i == self.split_layer:
				split_output = x
			x = block(x)

		output = self.lm_head(x)
		output = rearrange(output, 'b t e -> b e t')

		if labels is not None:
			if labels.dim() > 2:
				labels = rearrange(labels, 'b p t -> b (p t)')
			shift_logits = output[..., :-1].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			loss = self.cel(shift_logits, shift_labels)
		else:
			loss = 0

		return loss, split_output, output


def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		params = parameter.numel()
		table.add_row([name, params])
		total_params += params
	print(table)
	print(f"Total Trainable Params: {total_params}")
	return total_params
