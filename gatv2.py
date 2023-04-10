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

# GATv2 Layer Module -> currently not optimized for efficiency* and single-head module. Combined later in Multihead module
# Other supporting codes such as the Multi-neighbourhood Sampling and Training codes are referenced from DGL*


class GATv2_Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATv2_Layer, self).__init__()
        self.fc = nn.Linear(in_dim*2, out_dim, bias=False)
        self.attn_fc = nn.Linear(out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """
        to reinit parameters in dgl style
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # GATv2 attention fn
        # concat h_i, h_j -> W[h_i || h_j] -> a^{T} * LeakyRELU(W[h_i || h_j])
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        z2 = self.fc(z2)
        a = F.leaky_relu(self.attn_fc(z2), negative_slope=0.2)
        return {"e": a}

    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self,g, h):
        h_src, h_dst = h
        g.ndata["z"] = h[:g.num_dst_nodes()]
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop("h")

class GATv2_MultiHead_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GATv2_MultiHead_Layer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATv2_Layer(in_dim, out_dim))

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        return torch.cat(head_outs, dim=1)


class GATv2_Model(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, args):
        super().__init__()
        self.args = args
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(
            GATv2_MultiHead_Layer(
                in_feats, in_feats, 2
            )
        )
        for i in range(1, n_layers - 1):
            self.layers.append(
                GATv2_MultiHead_Layer(
                    n_hidden, n_hidden, 2
                )
            )
        self.layers.append(
            GATv2_MultiHead_Layer(
                n_hidden, n_classes, 2
            )
        )

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[: block.num_dst_nodes()]
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