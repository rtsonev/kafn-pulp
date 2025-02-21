import numpy as np
import nltk
import yake
import torch
import pickle
import joblib

from model_config import ModelConfig
from hetero_gat import HeteroGAT, nx_to_heterodata_full
from gensim.models import Doc2Vec
from typing import List, Tuple


class FakeNewsDetector:
    """
    Loads pretrained data and gives prediction for a new text
    """

    def __init__(self, model_config_path: str):
        """
        Constructor

        Args:
            model_config_path (str): Path to the model configuration file
        """
        self.config = ModelConfig.load(model_config_path)
        if self.config.doc2vec_filename:
            self.doc2vec_model = Doc2Vec.load(self.config.doc2vec_filename)
        if self.config.tfidf_model_filename:
            self.tfidf_model = joblib.load(self.config.tfidf_model_filename)
        with open(self.config.hetero_graph_filename, "rb") as f:
            self.H_all = pickle.load(f)
        self.train_doc_embs = np.load(self.config.doc_emb_path)
        self.embed_dim = self.train_doc_embs.shape[1]
        self.num_train = self.train_doc_embs.shape[0]
        self.doc_embeddings_dict = {
            i: self.train_doc_embs[i] for i in range(self.num_train)
        }
        self.model = HeteroGAT(
            in_dim=self.embed_dim,
            hidden_dim=self.config.gat_hidden_dim,
            num_layers=self.config.gat_num_layers,
            num_heads=self.config.gat_num_heads,
            dropout=self.config.gat_dropout,
        )
        self.model.load_state_dict(
            torch.load(self.config.model_filename, map_location=self.config.device)
        )
        self.model.to(self.config.device)
        self.model.eval()

    def predict_text(self, text: List[str]) -> List[Tuple[int, float]]:
        """
        For unseen texts - calculate embeddings, add them to the global graph and do forward pass to get a predictions.

        Args:
            text (List[str]): List of new texts to predict

        Returns:
             predictions (List[Tuple[int, float]]): A list of (pred_label, pred_prob) for each text in texts.
        """
        # get embeddings
        if self.config.doc2vec_filename:
            new_doc_embs = []
            for txt in text:
                tokens = nltk.word_tokenize(txt.lower())
                emb = self.doc2vec_model.infer_vector(tokens).astype(np.float32)
                new_doc_embs.append(emb)
            new_doc_embs = np.array(new_doc_embs, dtype=np.float32)
        else:
            X = self.tfidf_model.vectorizer.transform(text)
            new_doc_embs = X.toarray().astype(np.float32)

        # No mutation of original graph
        H_eval = self.H_all.copy()
        doc_embeddings_dict_eval = dict(self.doc_embeddings_dict)

        # Add new nodes
        start_new_doc_id = max(H_eval.nodes()) + 1
        new_doc_ids = range(start_new_doc_id, start_new_doc_id + len(text))
        for i, new_id in enumerate(new_doc_ids):
            H_eval.add_node(new_id, node_type="doc")
            doc_embeddings_dict_eval[new_id] = new_doc_embs[i]

        # Build edges for new dosc
        norms = np.linalg.norm(self.train_doc_embs, axis=1) + 1e-10
        for i, new_id in enumerate(new_doc_ids):
            new_doc_emb = new_doc_embs[i]
            new_norm = np.linalg.norm(new_doc_emb) + 1e-10
            sims = np.dot(self.train_doc_embs, new_doc_emb) / (norms * new_norm)
            top_k = 50
            topk_idx = np.argsort(sims)[::-1][:top_k]
            topk_sims = sims[topk_idx]
            sum_sim = float(np.sum(topk_sims) + 1e-10)
            for doc_i, sim_val in zip(topk_idx, topk_sims):
                w = sim_val / sum_sim
                H_eval.add_edge(new_id, doc_i, weight=w)
                H_eval.add_edge(doc_i, new_id, weight=w)

        # Add term edges with yake
        extractor = yake.KeywordExtractor(
            lan=self.config.yake_lan,
            n=self.config.yake_n,
            dedupLim=self.config.yake_dedup_lim,
        )
        term2node = {
            node_id: attrs["term_str"]
            for node_id, attrs in H_eval.nodes(data=True)
            if attrs.get("node_type") == "term"
        }
        inv_term2node = {v: k for k, v in term2node.items()}
        for i, new_id in enumerate(new_doc_ids):
            kw = extractor.extract_keywords(text[i])
            kw.sort(key=lambda x: x[1])
            kw = kw[: self.config.top_yake]
            terms = [r[0] for r in kw]
            matched_terms = []
            for t in terms:
                if t in inv_term2node:
                    matched_terms.append(inv_term2node[t])
            if matched_terms:
                w_term = 1.0 / len(matched_terms)
                for t_id in matched_terms:
                    H_eval.add_edge(new_id, t_id, weight=w_term)
                    H_eval.add_edge(t_id, new_id, weight=w_term)

        hetero_data, doc_list, _ = nx_to_heterodata_full(
            H_eval, embed_dim=self.embed_dim
        )
        doc_feats = [doc_embeddings_dict_eval[d] for d in doc_list]
        doc_feats = np.array(doc_feats, dtype=np.float32)
        hetero_data["doc"].x = torch.from_numpy(doc_feats)
        hetero_data = hetero_data.to(self.config.device)

        # Forward pass
        with torch.no_grad():
            _, doc_logits = self.model(hetero_data)
            doc_probs = torch.sigmoid(doc_logits).cpu().numpy()

        # Get list of predictions
        predictions = []
        for i, new_id in enumerate(new_doc_ids):
            idx_in_doc_list = doc_list.index(new_id)
            prob = float(doc_probs[idx_in_doc_list])
            label = int(prob > 0.5)
            predictions.append((label, prob))

        return predictions
