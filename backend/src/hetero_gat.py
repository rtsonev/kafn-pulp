import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv


class HeteroGAT(nn.Module):
    """
    A multi-layer GAT architecture for heterogeneous data, focusing on doc->doc, doc->term, term->doc edges.
    """

    def __init__(self, in_dim, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.5):
        """
        Constructor

        Args:
            in_dim (int): Dimension of input features (doc.x dimension).
            hidden_dim (int, optional): Dimension of each head's output in GATConv (before concatenation). Defaults to 64.
            num_layers (int, optional): Number of GAT layers to stack. Layers of HeteroConv, each containing GATConv for doc->doc, doc->term, and term->doc. Defaults to 2
            num_heads (int, optional): Number of attention heads in each GATConv. By default, PyG's GATConv concatenates attention heads. Defaults to 4.
            dropout (float, optional): Dropout probability applied after each layer. Defaults to 0.5.
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Layer 1: HeteroConv with doc->doc, doc->term, term->doc GATConv
        self.conv1 = HeteroConv(
            {
                ("doc", "to", "doc"): GATConv(
                    in_dim,
                    hidden_dim,
                    heads=num_heads,
                    add_self_loops=False,
                    concat=True,
                ),
                ("doc", "to", "term"): GATConv(
                    in_dim,
                    hidden_dim,
                    heads=num_heads,
                    add_self_loops=False,
                    concat=True,
                ),
                ("term", "to", "doc"): GATConv(
                    in_dim,
                    hidden_dim,
                    heads=num_heads,
                    add_self_loops=False,
                    concat=True,
                ),
            },
            aggr="sum",
        )

        # Optional layer 2
        self.conv2 = None
        if num_layers >= 2:
            self.conv2 = HeteroConv(
                {
                    ("doc", "to", "doc"): GATConv(
                        hidden_dim * num_heads,
                        hidden_dim,
                        heads=num_heads,
                        add_self_loops=False,
                        concat=True,
                    ),
                    ("doc", "to", "term"): GATConv(
                        hidden_dim * num_heads,
                        hidden_dim,
                        heads=num_heads,
                        add_self_loops=False,
                        concat=True,
                    ),
                    ("term", "to", "doc"): GATConv(
                        hidden_dim * num_heads,
                        hidden_dim,
                        heads=num_heads,
                        add_self_loops=False,
                        concat=True,
                    ),
                },
                aggr="sum",
            )

        # Final linear classifier from the doc node's last-layer representation -> 1 logit per doc node (binary classification).
        final_dim = hidden_dim * num_heads if self.conv2 else hidden_dim * num_heads
        self.classify = nn.Linear(final_dim, 1)

    def forward(self, data: HeteroData):
        """
        Forward pass through 1 or 2 GAT layers, then classification.

        Args:
            data (HeteroData): The PyG heterogeneous data object.

        Returns:
            out_dict(dict): Dictionary with node features after the final GAT layer (by node type).
            doc_logits(torch.Tensor): The predicted logits for doc nodes
        """
        x_dict, edge_index_dict, edge_attr_dict = (
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict,
        )

        # Layer 1
        out1 = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        for nt in out1:
            out1[nt] = F.elu(out1[nt])  # activation
            out1[nt] = F.dropout(out1[nt], p=self.dropout, training=self.training)

        # If only 1 layer classification is from out1 only
        if self.conv2 is not None:
            # Layer 2
            out2 = self.conv2(out1, edge_index_dict, edge_attr_dict)
            for nt in out2:
                out2[nt] = F.elu(out2[nt])
                out2[nt] = F.dropout(out2[nt], p=self.dropout, training=self.training)
            doc_logits = self.classify(out2["doc"]).squeeze(-1)
            return out2, doc_logits
        else:
            doc_logits = self.classify(out1["doc"]).squeeze(-1)
            return out1, doc_logits


def nx_to_heterodata_full(H: nx.Graph, embed_dim: int):
    """
    Converts the entire graph (all nodes, docs and terms) into a PyG HeteroData object. Assume each node in H has a node_type attribute (doc or term).

    Args:
        H (nx.Graph): Expanded network containing doc and term nodes, with doc->doc and doc->term edges.
        embed_dim (int): Embedding dimentions

    Returns:
        _type_: _description_
    """
    hetero_data = HeteroData()
    docs = []
    terms = []
    for n, attrs in H.nodes(data=True):
        if attrs.get("node_type") == "doc":
            docs.append(n)
        else:
            terms.append(n)
    docs = sorted(docs)
    terms = sorted(terms)
    doc_map = {d: i for i, d in enumerate(docs)}
    term_map = {t: i for i, t in enumerate(terms)}
    hetero_data["doc"].x = torch.zeros((len(docs), embed_dim), dtype=torch.float)
    hetero_data["term"].x = torch.zeros((len(terms), embed_dim), dtype=torch.float)

    doc_doc_src, doc_doc_dst, doc_doc_w = [], [], []
    doc_term_src, doc_term_dst, doc_term_w = [], [], []
    term_doc_src, term_doc_dst, term_doc_w = [], [], []

    for u, v, edata in H.edges(data=True):
        w = edata.get("weight", 1.0)
        ut = H.nodes[u]["node_type"]
        vt = H.nodes[v]["node_type"]
        if ut == "doc" and vt == "doc":
            doc_doc_src.append(doc_map[u])
            doc_doc_dst.append(doc_map[v])
            doc_doc_w.append(w)
        elif ut == "doc" and vt == "term":
            doc_term_src.append(doc_map[u])
            doc_term_dst.append(term_map[v])
            doc_term_w.append(w)
        elif ut == "term" and vt == "doc":
            term_doc_src.append(term_map[u])
            term_doc_dst.append(doc_map[v])
            term_doc_w.append(w)
    if len(doc_doc_src) > 0:
        hetero_data["doc", "to", "doc"].edge_index = torch.tensor(
            [doc_doc_src, doc_doc_dst], dtype=torch.long
        )
        hetero_data["doc", "to", "doc"].edge_attr = torch.tensor(
            doc_doc_w, dtype=torch.float
        )
    else:
        hetero_data["doc", "to", "doc"].edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )
        hetero_data["doc", "to", "doc"].edge_attr = torch.empty((0,), dtype=torch.float)
    if len(doc_term_src) > 0:
        hetero_data["doc", "to", "term"].edge_index = torch.tensor(
            [doc_term_src, doc_term_dst], dtype=torch.long
        )
        hetero_data["doc", "to", "term"].edge_attr = torch.tensor(
            doc_term_w, dtype=torch.float
        )
    else:
        hetero_data["doc", "to", "term"].edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )
        hetero_data["doc", "to", "term"].edge_attr = torch.empty(
            (0,), dtype=torch.float
        )
    if len(term_doc_src) > 0:
        hetero_data["term", "to", "doc"].edge_index = torch.tensor(
            [term_doc_src, term_doc_dst], dtype=torch.long
        )
        hetero_data["term", "to", "doc"].edge_attr = torch.tensor(
            term_doc_w, dtype=torch.float
        )
    else:
        hetero_data["term", "to", "doc"].edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )
        hetero_data["term", "to", "doc"].edge_attr = torch.empty(
            (0,), dtype=torch.float
        )
    hetero_data["doc"].docs = torch.tensor(docs, dtype=torch.long)
    return hetero_data, docs, terms
