import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

import dgl
import dgl.nn.pytorch as dglnn
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.base import DGLError




class GATv2_Layer(nn.Module):
	# single graph attention network V2 layer

	def __init__(self, inp_feats, out_feats, n_heads, Leaky_ReLU_slope=0.2):
		super().__init__()
		self.n_heads = n_heads
		self.in_src_feats, self.in_dst_feats = expand_as_pair(inp_feats)
		self.e = nn.Parameter(torch.FloatTensor(size=(1,n_heads,out_feats)))
		self.Leaky_ReLU = nn.LeakyReLU(Leaky_ReLU_slope)
		self.fc_src = nn.Linear(
				self.in_src_feats, out_feats * n_heads, bias=True)
		self.fc_dst = nn.Linear(
				self.in_dst_feats, out_feats * n_heads, bias=True)
		if self.in_dst_feats != out_feats:
				self.res_fc = nn.Linear(
					self.in_dst_feats, n_heads * out_feats, bias=True)
		else:
			self.res_fc = Identity()
		self.feat_dropout = nn.Dropout(0)
		self.attn_dropout = nn.Dropout(0)
		self.out_feats = out_feats
		self.reset_parameters()

	def reset_parameters(self):
		gain = nn.init.calculate_gain('relu')
		# init linear layer for source 
		nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
		# init bias
		nn.init.constant_(self.fc_src.bias, 0)
		nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
		nn.init.constant_(self.fc_dst.bias, 0)
		nn.init.xavier_normal_(self.e, gain=gain)
		nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
		nn.init.constant_(self.res_fc.bias, 0)

	def forward(self, graph, feat, get_attention=False):
		with graph.local_scope():
			
			if isinstance(feat, tuple):
				h_src = self.feat_dropout(feat[0])
				h_dst = self.feat_dropout(feat[1])
				feat_src = self.fc_src(h_src).view(-1, self.n_heads, self.out_feats)
				feat_dst = self.fc_dst(h_dst).view(-1, self.n_heads, self.out_feats)
			else:
				h_src = h_dst = self.feat_dropout(feat)
				feat_src = self.fc_src(h_src).view(
					-1, self.n_heads, self.out_feats)
				if self.share_weights:
					feat_dst = feat_src
				else:
					feat_dst = self.fc_dst(h_src).view(
						-1, self.n_heads, self.out_feats)
				if graph.is_block:
					feat_dst = feat_src[:graph.number_of_dst_nodes()]
			graph.srcdata.update({'el': feat_src})# (num_src_edge, num_heads, out_dim)
			graph.dstdata.update({'er': feat_dst})
			graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
			e = self.Leaky_ReLU(graph.edata.pop('e'))# (num_src_edge, num_heads, out_dim)
			e = (e * self.e).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
			# compute softmax
			graph.edata['a'] = self.attn_dropout(edge_softmax(graph, e)) # (num_edge, num_heads)
			# message passing
			graph.update_all(fn.u_mul_e('el', 'a', 'm'),
							 fn.sum('m', 'ft'))
			rst = graph.dstdata['ft']
			# residual
			if self.res_fc is not None:
				resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self.out_feats)
				rst = rst + resval

			if get_attention:
				return rst, graph.edata['a']
			else:
				return rst



class GATv2_Model(nn.Module):
	def __init__(self, in_feats, n_hidden, n_classes, n_layers, num_heads,args):
		super().__init__()
		self.args = args
		self.n_layers = n_layers
		self.n_hidden = n_hidden
		self.n_classes = n_classes
		self.layers = nn.ModuleList()
		self.layers.append(
			GATv2_Layer(
				(in_feats, in_feats),
				n_hidden,
				n_heads=num_heads
			)
		)
		for i in range(1, n_layers - 1):
			self.layers.append(
				GATv2_Layer(
					(n_hidden * num_heads, n_hidden * num_heads),
					n_hidden,
					n_heads=num_heads
				)
			)
		self.layers.append(
			GATv2_Layer(
				(n_hidden * num_heads, n_hidden * num_heads),
				n_classes,
				n_heads=num_heads
			)
		)

	def forward(self, blocks, x):
		h = x
		for l, (layer, block) in enumerate(zip(self.layers, blocks)):
			# We need to first copy the representation of nodes on the RHS from the
			# appropriate nodes on the LHS.
			# Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
			# would be (num_nodes_RHS, D)
			h_dst = h[: block.num_dst_nodes()]
			# Then we compute the updated representation on the RHS.
			# The shape of h now becomes (num_nodes_RHS, D)
			if l < self.n_layers - 1:
				h = layer(block, (h, h_dst)).flatten(1)
			else:
				h = layer(block, (h, h_dst))
		h = h.mean(1)
		return h.log_softmax(dim=-1)

	def inference(self, g, x, num_heads, device):
		"""
		Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
		g : the entire graph.
		x : the input of entire node set.
		The inference code is written in a fashion that it could handle any number of nodes and
		layers.
		"""
		# During inference with sampling, multi-layer blocks are very inefficient because
		# lots of computations in the first few layers are repeated.
		# Therefore, we compute the representation of all nodes layer by layer.  The nodes
		# on each layer are of course splitted in batches.
		# TODO: can we standardize this?
		for l, layer in enumerate(self.layers):
			if l < self.n_layers - 1:
				y = torch.zeros(
					g.num_nodes(),
					self.n_hidden * num_heads
					if l != len(self.layers) - 1
					else self.n_classes,
				)
			else:
				y = torch.zeros(
					g.num_nodes(),
					self.n_hidden
					if l != len(self.layers) - 1
					else self.n_classes,
				)

			sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
			dataloader = dgl.dataloading.DataLoader(
				g,
				torch.arange(g.num_nodes()),
				sampler,
				batch_size=self.args.batch_size,
				shuffle=True,
				drop_last=False,
				num_workers=self.args.num_workers,
			)

			for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
				block = blocks[0].int().to(device)

				h = x[input_nodes].to(device)
				h_dst = h[: block.num_dst_nodes()]
				if l < self.n_layers - 1:
					h = layer(block, (h, h_dst)).flatten(1)
				else:
					h = layer(block, (h, h_dst))
					h = h.mean(1)
					h = h.log_softmax(dim=-1)

				y[output_nodes] = h.cpu()

			x = y
		return y.to(device)


# ORIGINAL GAT CODE FROM DGL - used for testing against original GAT

class GAT(nn.Module):
	def __init__(
		self, in_feats, n_hidden, n_classes, n_layers, num_heads, activation
	):
		super().__init__()
		self.n_layers = n_layers
		self.n_hidden = n_hidden
		self.n_classes = n_classes
		self.layers = nn.ModuleList()
		self.layers.append(
			dglnn.GATConv(
				(in_feats, in_feats),
				n_hidden,
				num_heads=num_heads,
				activation=activation,
			)
		)
		for i in range(1, n_layers - 1):
			self.layers.append(
				dglnn.GATConv(
					(n_hidden * num_heads, n_hidden * num_heads),
					n_hidden,
					num_heads=num_heads,
					activation=activation,
				)
			)
		self.layers.append(
			dglnn.GATConv(
				(n_hidden * num_heads, n_hidden * num_heads),
				n_classes,
				num_heads=num_heads,
				activation=None,
			)
		)

	def forward(self, blocks, x):
		h = x
		for l, (layer, block) in enumerate(zip(self.layers, blocks)):
			# We need to first copy the representation of nodes on the RHS from the
			# appropriate nodes on the LHS.
			# Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
			# would be (num_nodes_RHS, D)
			h_dst = h[: block.num_dst_nodes()]
			# Then we compute the updated representation on the RHS.
			# The shape of h now becomes (num_nodes_RHS, D)
			if l < self.n_layers - 1:
				h = layer(block, (h, h_dst)).flatten(1)
			else:
				h = layer(block, (h, h_dst))
		h = h.mean(1)
		return h.log_softmax(dim=-1)

	def inference(self, g, x, num_heads, device):
		"""
		Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
		g : the entire graph.
		x : the input of entire node set.
		The inference code is written in a fashion that it could handle any number of nodes and
		layers.
		"""
		# During inference with sampling, multi-layer blocks are very inefficient because
		# lots of computations in the first few layers are repeated.
		# Therefore, we compute the representation of all nodes layer by layer.  The nodes
		# on each layer are of course splitted in batches.
		# TODO: can we standardize this?
		for l, layer in enumerate(self.layers):
			if l < self.n_layers - 1:
				y = torch.zeros(
					g.num_nodes(),
					self.n_hidden * num_heads
					if l != len(self.layers) - 1
					else self.n_classes,
				)
			else:
				y = torch.zeros(
					g.num_nodes(),
					self.n_hidden
					if l != len(self.layers) - 1
					else self.n_classes,
				)

			sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
			dataloader = dgl.dataloading.DataLoader(
				g,
				torch.arange(g.num_nodes()),
				sampler,
				batch_size=args.batch_size,
				shuffle=True,
				drop_last=False,
				num_workers=args.num_workers,
			)

			for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
				block = blocks[0].int().to(device)

				h = x[input_nodes].to(device)
				h_dst = h[: block.num_dst_nodes()]
				if l < self.n_layers - 1:
					h = layer(block, (h, h_dst)).flatten(1)
				else:
					h = layer(block, (h, h_dst))
					h = h.mean(1)
					h = h.log_softmax(dim=-1)

				y[output_nodes] = h.cpu()

			x = y
		return y.to(device)