import os
import numpy as np
import pandas as pd
import nltk
import yake
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from joblib import Parallel, delayed
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import fractional_matrix_power
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv

# Download necessary NLTK data if not already present
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


############################################################
# --- 0. Data & Utilities ---
############################################################

def evaluation_metrics(
    y_true, y_pred, y_proba=None, title=None, labels=["real", "fake"]
):
    """
    Prints out classification metrics (Precision, Recall, F1-score) 
    and confusion matrix for a given set of predictions.

    Parameters
    ----------
    y_true : array-like
        True labels for test set (0 => real, 1 => fake).
    y_pred : array-like
        Predicted labels (0 => real, 1 => fake).
    y_proba : array-like, optional
        Predicted probabilities (or decision function values) 
        for the positive class. Used to compute ROC-AUC and PR-AUC.
    title : str, optional
        A title to display with the metrics.
    labels : list of str
        The class labels in order. Default is ["real", "fake"].
    """
    print(f"\n{title}")
    # Print classification report (precision, recall, f1-score, support)
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

    # Display confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title(title)
    plt.show()

    # If probabilities are given, also show ROC-AUC and PR-AUC
    if y_proba is not None:
        roc_auc = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
    print("\n")


class KAFNData:
    """
    This class encapsulates the entire dataset and allows for a
    transductive learning approach:
      - We first do a train/test split for final evaluation.
      - Among the train portion, only a fraction of the 'fake' documents
        receive the label pu_label=1. The rest remain unlabeled (pu_label=-1).
      - All 'real' docs are forced to pu_label=-1 because we assume 
        "fake" is our positive/interesting class for PU learning.

    Attributes
    ----------
    fraction_fake_labeled : float
        Fraction of fake documents (in the training set) that are actually labeled 
        as 'fake' (pu_label=1).
    df_all : pd.DataFrame
        The entire dataset after the train/test split, containing columns:
            - 'full_text': text of the document
            - 'binary_label': 0 => real, 1 => fake
            - 'split': "train" or "test"
            - 'pu_label': -1 => unlabeled, 1 => labeled fake
    num_docs : int
        Total number of documents in df_all
    """

    def __init__(
        self,
        csv_path,
        random_state=42,
        fraction_fake_labeled=0.3,
        test_size=0.2,
    ):
        """
        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing 'full_text' and 'binary_label' columns.
        random_state : int
            Random seed for reproducible train/test splits.
        fraction_fake_labeled : float
            Fraction of fake docs (in training set) that will receive a positive label (pu_label=1).
        test_size : float
            Fraction of dataset for the test set, typically 0.2 => 20%.
        """
        self.fraction_fake_labeled = fraction_fake_labeled

        # Read CSV. If lines have issues, skip them with on_bad_lines="skip".
        df = pd.read_csv(csv_path, on_bad_lines="skip")

        # Train/test split for final evaluation.
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["binary_label"],  # preserve class distribution
            random_state=random_state,
        )

        # Combine train/test back into a single df, with a split indicator.
        self.df_all = pd.concat([train_df, test_df], ignore_index=True)
        self.df_all["split"] = "test"
        self.df_all.loc[self.df_all.index[: len(train_df)], "split"] = "train"
        self.num_docs = len(self.df_all)

        # Initialize pu_label => -1 for all docs.
        # Then we will selectively set pu_label=1 for a fraction of fake docs in the train set.
        self.df_all["pu_label"] = -1

        # Identify all fake docs in the train set.
        train_fake_inds = self.df_all.index[
            (self.df_all["split"] == "train") & (self.df_all["binary_label"] == 1)
        ].tolist()

        # Number of labeled "fake" docs we want to keep.
        n_label = int(fraction_fake_labeled * len(train_fake_inds))
        if n_label < 1 and len(train_fake_inds) > 0:
            n_label = 1  # ensure at least 1 is labeled if there's at least 1 fake doc

        # Randomly pick that many from the train fake docs to label them as pu_label=1
        if len(train_fake_inds) > 0:
            labeled_inds = np.random.choice(
                train_fake_inds, size=n_label, replace=False
            )
            self.df_all.loc[labeled_inds, "pu_label"] = 1

    def get_doc_text(self, idx):
        """
        Returns the full text for document with index idx.
        """
        return self.df_all.iloc[idx]["full_text"]

    def get_pu_label(self, idx):
        """
        Returns the PU label (1 => labeled fake, -1 => unlabeled).
        """
        return self.df_all.iloc[idx]["pu_label"]

    def get_binary_label(self, idx):
        """
        Returns the ground truth binary label (0 => real, 1 => fake).
        """
        return self.df_all.iloc[idx]["binary_label"]

    def is_train_doc(self, idx):
        """
        Returns True if doc index idx belongs to the training set.
        """
        return self.df_all.iloc[idx]["split"] == "train"

    def is_test_doc(self, idx):
        """
        Returns True if doc index idx belongs to the test set.
        """
        return self.df_all.iloc[idx]["split"] == "test"


def baseline(data: KAFNData):
    """
    A simple baseline approach using TF-IDF and MultinomialNB to illustrate 
    how we might train on the entire training set with pseudo-labels:
      - pu_label==1 => positive (fake)
      - pu_label==-1 => negative (real or unlabeled fake)
    Then evaluate using the true binary labels on the test set.
    """

    # Partition the data into train/test based on 'split' column.
    train_df = data.df_all[data.df_all["split"] == "train"]
    test_df = data.df_all[data.df_all["split"] == "test"]

    # X_tr => training texts, y_tr => pseudo-labels (1 => labeled fake, 0 => otherwise)
    X_tr = train_df["full_text"]
    y_tr = train_df["pu_label"].apply(lambda x: 1 if x == 1 else 0)

    # X_tst => test texts, y_tst => real binary labels (0 => real, 1 => fake)
    X_tst = test_df["full_text"]
    y_tst = test_df["binary_label"]

    # We define a pipeline: TF-IDF + MultinomialNB
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=5000,  # up to 5000 features for TF-IDF
                    ngram_range=(1, 2),  # unigrams + bigrams
                ),
            ),
            (
                "classifier",
                MultinomialNB(
                    class_prior=[0.5, 0.5]  # Prior class probabilities for NB
                ),
            ),
        ]
    )

    # Fit on the pseudo-labeled training data
    pipeline.fit(X_tr, y_tr)

    # Predict on the test set
    y_pr = pipeline.predict(X_tst)

    # Also get predicted probabilities for scoring
    proba = pipeline.predict_proba(X_tst)
    y_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

    # Print metrics
    title = f"=== Baseline with TfidfVectorizer and MultinomialNB, Labeled fraction {data.fraction_fake_labeled} ==="
    evaluation_metrics(y_tst, y_pr, y_proba, title=title)


############################################################
# --- 1. Doc2Vec Embedding ---
############################################################

class Doc2VecEmb:
    """
    This class wraps around Gensim's Doc2Vec model for document embeddings.
    """

    def __init__(self, vector_size=100, window=5, min_count=2, epochs=20, workers=4):
        """
        Parameters
        ----------
        vector_size : int
            Dimensionality of the embedding vectors.
        window : int
            Maximum distance between the current and predicted word within a sentence.
        min_count : int
            Ignores all words with total frequency lower than this.
        epochs : int
            Number of training epochs.
        workers : int
            Number of worker threads to train the model.
        """
        self.model = None
        self.vsize = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers

    def train(self, data: KAFNData):
        """
        Trains the Doc2Vec model on the entire corpus (all docs).
        Each doc is tokenized, lower-cased, and assigned a unique tag (the doc ID).
        """
        docs = []
        for i in range(data.num_docs):
            tokens = word_tokenize(data.get_doc_text(i).lower())
            docs.append(TaggedDocument(tokens, [i]))

        model = Doc2Vec(
            documents=docs,
            vector_size=self.vsize,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=self.workers,
        )
        self.model = model

    def get_vector(self, i):
        """
        Retrieve the learned embedding vector for doc i.
        """
        return self.model.dv[i]


class TFIDFEmb:
    """
    A simple TF-IDF-based "embedding" class.
    We compute TF-IDF vectors for each document and store them in memory.
    """

    def __init__(
        self, max_features=10000, ngram_range=(1, 1), min_df=2, stop_words="english"
    ):
        """
        Parameters
        ----------
        max_features : int
            Maximum number of features (vocabulary size) in TF-IDF.
        ngram_range : tuple
            The lower and upper boundary of the n-grams to be extracted.
        min_df : int
            Ignore terms that appear in fewer than min_df documents.
        stop_words : str or list
            Stop words to remove from the text. 
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            # min_df=min_df, # optionally uncomment if needed
            stop_words=stop_words,
        )
        self.doc_vectors = None  # Will store the dense np.array for each doc

    def train(self, data):
        """
        Fit TF-IDF to the entire corpus and transform documents into TF-IDF vectors.
        The resulting vectors are converted to dense arrays and stored.
        """
        texts = [data.get_doc_text(i) for i in range(data.num_docs)]
        tfidf_matrix = self.vectorizer.fit_transform(texts)  # shape: (num_docs, num_features)
        self.doc_vectors = tfidf_matrix.toarray()  # Convert to dense for easier use later.

    def get_vector(self, i):
        """
        Get the TF-IDF vector for document i.
        """
        return self.doc_vectors[i]


############################################################
# --- 2. Build Weighted Doc-Doc Graph + Normalization ---
############################################################

def build_doc_graph(data: KAFNData, emb_model: Doc2VecEmb, k=5):
    """
    Build a graph of documents, each doc as a node.
    Edges connect each doc to its top-k nearest neighbors by cosine similarity.

    Steps:
      1) Compute embeddings for each doc (N x embedding_dim).
      2) Compute NxN cosine similarity matrix.
      3) For each doc i, pick top-k similar docs and create edges i->n in both directions.
      4) Normalize the edge weights from each doc so they sum to 1 (row-stochastic).

    Parameters
    ----------
    data : KAFNData
        The dataset object.
    emb_model : Doc2VecEmb or TFIDFEmb
        An embedding model with a get_vector(i) method.
    k : int
        Number of neighbors to connect per doc.

    Returns
    -------
    G : nx.Graph
        A NetworkX graph with doc nodes and weighted edges.
    """
    N = data.num_docs
    # 1) Gather embeddings
    all_emb = np.array([emb_model.get_vector(i) for i in range(N)])
    # 2) Cosine similarity
    sim = cosine_similarity(all_emb)

    G = nx.Graph()
    # Add nodes for each doc (labeled with node_type="doc")
    for i in range(N):
        G.add_node(i, node_type="doc")

    # 3) For each doc, pick top-k neighbors
    for i in range(N):
        row = sim[i].copy()
        row[i] = -999  # exclude self from top-k
        nbrs = np.argsort(row)[::-1][:k]  # indices of top-k largest similarities
        for n in nbrs:
            G.add_edge(i, n, weight=sim[i, n])
            G.add_edge(n, i, weight=sim[i, n])

    # 4) Normalize each doc's outgoing edges to sum=1
    for i in range(N):
        neighbors = list(G.neighbors(i))
        total_w = sum(G[i][nb]["weight"] for nb in neighbors)
        if total_w > 0:
            for nb in neighbors:
                old_w = G[i][nb]["weight"]
                G[i][nb]["weight"] = old_w / total_w

    return G


############################################################
# --- 3. Katz + Iterative PU-LP ---
############################################################

def compute_katz_matrix(G, alpha=0.01):
    """
    Compute the Katz similarity matrix for graph G, using:
      Katz = (I - alpha*A)^(-1) - I
    where A is the adjacency (here weighted) matrix.

    We use 'fractional_matrix_power' or direct inversion. If the matrix is singular,
    we catch LinAlgError and return None.

    Parameters
    ----------
    G : nx.Graph
        Graph with N nodes.
    alpha : float
        Katz damping factor controlling how much longer paths contribute.

    Returns
    -------
    katz_mat : np.array of shape (N, N) or None
        The Katz similarity matrix, or None if inversion fails.
    idx_map : dict
        Mapping from node ID to row index in the matrix.
    """
    # Sort nodes to get consistent ordering
    nodes = sorted(G.nodes())
    N = len(nodes)
    # Build adjacency matrix from the graph's 'weight'
    A = np.zeros((N, N), dtype=float)
    idx_map = {n: i for i, n in enumerate(nodes)}

    for u, v, data_e in G.edges(data=True):
        A[idx_map[u], idx_map[v]] = data_e["weight"]

    I = np.eye(N)
    try:
        mat = fractional_matrix_power(I - alpha * A, -1) - I
        return mat, idx_map
    except np.linalg.LinAlgError:
        print("Katz inversion failed.")
        return None, idx_map


def iterative_pu_lp(katz_mat, idx_map, pos_indices, unl_indices, m_iter=3, lam=1.0):
    """
    Iterative Positive-Unlabeled label propagation.
    We have:
      - pos_indices => IDs of docs that are known positives (pu_label=1)
      - unl_indices => IDs of docs that are unlabeled (pu_label=-1)
      - katz_mat => NxN Katz similarity matrix
      - idx_map => dictionary mapping doc ID to row index in katz_mat.

    Steps:
      1) Initialize pos_set = set(pos_indices), unl_set = set(unl_indices).
      2) For m_iter iterations:
         a) R_I': pick top (lam/m_iter * len(pos_set)) docs from unl_set
            sorted by average Katz similarity to pos_set => newly found positives
         b) Move them from unl_set to pos_set, and accumulate them in R_I_tot
         c) From the updated unl_set, pick bottom (lam/m_iter * len(pos_set)) docs
            by similarity => newly found negatives => R_N_tot
         d) Remove those from unl_set as well.
      3) Return the new sets R_I (potential fakes) and R_N (potential reals),
         plus leftover unlabeled docs that remain.

    Returns
    -------
    R_I_tot : list
        Doc IDs newly assigned to the positive (fake) set.
    R_N_tot : list
        Doc IDs newly assigned to the negative (real) set.
    leftover : list
        Doc IDs still unlabeled after the iterations.
    """
    pos_set = set(pos_indices)
    unl_set = set(unl_indices)

    R_I_tot = set()
    R_N_tot = set()

    def mean_sim(u_id, pos_ids):
        """
        Compute the average Katz similarity from doc u_id to all docs in pos_ids.
        """
        if not pos_ids:
            return 0.0
        row_u = idx_map[u_id]
        sims = [katz_mat[row_u, idx_map[p_id]] for p_id in pos_ids]
        return np.mean(sims) if len(sims) > 0 else 0.0

    for _ in range(m_iter):
        subset_size = int((lam / m_iter) * len(pos_set))
        # Ensure at least 1 if pos_set is not empty
        if subset_size < 1 and len(pos_set) > 0:
            subset_size = 1

        # a) Rank unl_set by average sim to pos_set, pick top subset_size => R_I'
        unl_scores = [(u, mean_sim(u, pos_set)) for u in unl_set]
        unl_scores.sort(key=lambda x: x[1], reverse=True)
        R_I_prime = [x[0] for x in unl_scores[:subset_size]]

        # Move them from unl_set to pos_set
        for r in R_I_prime:
            unl_set.remove(r)
        R_I_tot.update(R_I_prime)
        pos_set.update(R_I_prime)

        # b) Now rank the updated unl_set by average sim to pos_set, pick bottom subset_size => R_N'
        unl_scores2 = [(u, mean_sim(u, pos_set)) for u in unl_set]
        unl_scores2.sort(key=lambda x: x[1])
        R_N_prime = [x[0] for x in unl_scores2[:subset_size]]

        for r in R_N_prime:
            unl_set.remove(r)
        R_N_tot.update(R_N_prime)

    return list(R_I_tot), list(R_N_tot), list(unl_set)


############################################################
# --- 4. YAKE doc->term edges + filtering + normalization ---
############################################################

def extract_yake_keywords(text, top_k, yake_lan, yake_n, yake_dedup_lim):
    """
    Extract top_k keywords from a given text using YAKE.
    Sort them by ascending score (lower is better).
    Return only the top_k.
    """
    extractor = yake.KeywordExtractor(lan=yake_lan, n=yake_n, dedupLim=yake_dedup_lim)
    kw = extractor.extract_keywords(text)
    kw.sort(key=lambda x: x[1])  # ascending => best first
    return kw[:top_k]


def build_doc_term_edges(
    G_doc,
    data: KAFNData,
    top_k=5,
    yake_lan="en",
    yake_n=1,
    yake_dedup_lim=0.9,
    n_jobs=32,
):
    """
    Incorporate keywords (terms) into the doc graph, forming a doc-term bipartite structure.

    Steps:
      1) Extract top_k keywords for each doc using YAKE in parallel.
      2) Keep only terms appearing in >=2 docs.
      3) Create 'term' nodes for those valid terms.
      4) Create edges doc->term, weighted by YAKE-based normalization (Equation (2)).
      5) Normalize again so each node's edges sum to 1 (Equation (3)).

    Parameters
    ----------
    G_doc : nx.Graph
        Original doc-doc graph.
    data : KAFNData
        The dataset.
    top_k : int
        Number of keywords to extract per doc.
    yake_lan : str
        Language for YAKE (e.g. "en", "pt").
    yake_n : int
        n-gram size for YAKE keywords (1 => single word).
    yake_dedup_lim : float
        Deduplication threshold for YAKE.
    n_jobs : int
        Number of parallel jobs (CPU cores) to use in YAKE extraction.

    Returns
    -------
    H : nx.Graph
        Expanded network containing doc and term nodes, with doc->doc and doc->term edges.
    """
    N = data.num_docs
    # Create a new graph H that will include both doc and term nodes
    H = nx.Graph()

    # 1) Copy doc nodes from G_doc
    for i in range(N):
        H.add_node(i, node_type="doc")
    # Copy doc->doc edges
    for u, v, d in G_doc.edges(data=True):
        H.add_edge(u, v, weight=d["weight"])

    # 2) Extract YAKE keywords in parallel
    all_texts = [data.get_doc_text(i) for i in range(N)]
    parallel = Parallel(n_jobs=n_jobs, backend="loky")
    all_kw_lists = parallel(
        delayed(extract_yake_keywords)(text, top_k, yake_lan, yake_n, yake_dedup_lim)
        for text in all_texts
    )

    # 3) Summarize term usage
    term_usage = {}
    raw_doc_terms = [[] for _ in range(N)]
    for i, kw in enumerate(all_kw_lists):
        for ts, sc in kw:
            raw_doc_terms[i].append((ts, sc))
            term_usage.setdefault(ts, set()).add(i)

    # 4) Keep only terms used by >=2 docs
    valid_terms = {t for t, used_by in term_usage.items() if len(used_by) >= 2}

    # 5) Create new term nodes
    next_id = N
    term_map = {}
    for t in valid_terms:
        term_map[t] = next_id
        H.add_node(next_id, node_type="term")
        next_id += 1

    # 6) Create doc->term edges with normalized weights
    for i in range(N):
        doc_term_list = []
        for ts, sc in raw_doc_terms[i]:
            if ts not in valid_terms:
                continue

            # Weighted formula from the provided Equation (2)
            # if score > 1 => weight=0, else => w_val= score*(1 - (1/(1+score)))
            if sc > 1:
                w_val = 0.0
            else:
                w_val = sc * (1 - (1 / (1 + sc)))

            if w_val > 0:
                doc_term_list.append((ts, w_val))

        # We then normalize the doc->term edges so their sum=1
        denom = sum(x[1] for x in doc_term_list)
        if denom > 0:
            for ts, w in doc_term_list:
                norm_w = w / denom
                tid = term_map[ts]
                H.add_edge(i, tid, weight=norm_w)
                H.add_edge(tid, i, weight=norm_w)

    # 7) Finally, we re-normalize edges (Equation (3)), ensuring 
    #    each node's edges sum to 1, covering both doc->doc and doc->term edges.
    for node in H.nodes():
        neighbors = list(H.neighbors(node))
        total_w = sum(H[node][nb]["weight"] for nb in neighbors)
        if total_w > 0:
            for nb in neighbors:
                H[node][nb]["weight"] /= total_w

    return H


############################################################
# --- 5. Convert Nx => HeteroData ---
############################################################

def nx_to_heterodata(H, R_I, R_N, pos_indices, embed_dim=100):
    """
    Convert the doc-term Nx graph into a torch_geometric HeteroData object.

    Node types: "doc", "term"
    Relations: 
       doc->doc, doc->term, term->doc (no term->term).

    We also store the doc label:
      label=1 if doc in (pos_indices union R_I),
      label=0 if doc in R_N,
      label=-1 otherwise (i.e., still unlabeled after PU-LP).

    Finally, we allocate x vectors of size embed_dim for docs, 
    and placeholders for terms as well. We'll later fill doc x 
    with real embeddings (Doc2Vec or TF-IDF). 
    term.x is left as zeros or also later updated if needed by a GNEE-like approach.
    """
    # Separate doc and term nodes
    docs, terms = [], []
    for n, dattr in H.nodes(data=True):
        if dattr["node_type"] == "doc":
            docs.append(n)
        else:
            terms.append(n)
    docs.sort()
    terms.sort()

    # Create doc_label array
    # doc_label: 1 => doc in pos_indices or R_I
    #            0 => doc in R_N
    #            -1 => leftover
    doc_label_arr = []
    pos_set = set(pos_indices)
    R_I_set = set(R_I)
    R_N_set = set(R_N)
    for d in docs:
        if d in pos_set or d in R_I_set:
            doc_label_arr.append(1)
        elif d in R_N_set:
            doc_label_arr.append(0)
        else:
            doc_label_arr.append(-1)
    doc_label_arr = np.array(doc_label_arr)

    doc_map = {d: i for i, d in enumerate(docs)}
    term_map = {t: i for i, t in enumerate(terms)}

    # Initialize x
    doc_x = np.zeros((len(docs), embed_dim), dtype=np.float32)
    term_x = np.zeros((len(terms), embed_dim), dtype=np.float32)

    # Prepare edge lists
    doc_doc_src, doc_doc_dst, doc_doc_attr = [], [], []
    doc_term_src, doc_term_dst, doc_term_attr = [], [], []
    term_doc_src, term_doc_dst, term_doc_attr = [], [], []

    for u, v, edat in H.edges(data=True):
        w = edat["weight"]
        ut = H.nodes[u]["node_type"]
        vt = H.nodes[v]["node_type"]

        if ut == "doc" and vt == "doc":
            doc_doc_src.append(doc_map[u])
            doc_doc_dst.append(doc_map[v])
            doc_doc_attr.append(w)
        elif ut == "doc" and vt == "term":
            doc_term_src.append(doc_map[u])
            doc_term_dst.append(term_map[v])
            doc_term_attr.append(w)
        elif ut == "term" and vt == "doc":
            term_doc_src.append(term_map[u])
            term_doc_dst.append(doc_map[v])
            term_doc_attr.append(w)
        # No term->term edges in this approach

    # Build HeteroData object
    hetero_data = HeteroData()

    # doc node feature matrix, label array, original doc IDs
    hetero_data["doc"].x = torch.FloatTensor(doc_x)
    hetero_data["doc"].labels = torch.LongTensor(doc_label_arr)
    hetero_data["doc"].docs = torch.LongTensor(docs)

    # term node feature matrix
    hetero_data["term"].x = torch.FloatTensor(term_x)

    # doc->doc edges
    hetero_data["doc", "to", "doc"].edge_index = torch.tensor(
        [doc_doc_src, doc_doc_dst], dtype=torch.long
    )
    hetero_data["doc", "to", "doc"].edge_attr = torch.FloatTensor(doc_doc_attr)

    # doc->term edges
    hetero_data["doc", "to", "term"].edge_index = torch.tensor(
        [doc_term_src, doc_term_dst], dtype=torch.long
    )
    hetero_data["doc", "to", "term"].edge_attr = torch.FloatTensor(doc_term_attr)

    # term->doc edges
    hetero_data["term", "to", "doc"].edge_index = torch.tensor(
        [term_doc_src, term_doc_dst], dtype=torch.long
    )
    hetero_data["term", "to", "doc"].edge_attr = torch.FloatTensor(term_doc_attr)

    return hetero_data, docs, terms


def fill_doc_embeddings(hetero_data: HeteroData, docs, data: KAFNData, emb_model):
    """
    Fill hetero_data["doc"].x with the actual document embeddings 
    computed by the emb_model (Doc2Vec or TF-IDF).

    Parameters
    ----------
    hetero_data : HeteroData
        The PyG heterogeneous data object.
    docs : list of int
        The sorted list of doc IDs.
    data : KAFNData
        Our data object (not used here except for passing to emb_model).
    emb_model : Doc2VecEmb or TFIDFEmb
        The embedding model with get_vector(d) method.
    """
    doc_x_list = []
    for d in docs:
        emb = emb_model.get_vector(d)
        doc_x_list.append(emb)

    doc_x_arr = np.vstack(doc_x_list).astype(np.float32)
    hetero_data["doc"].x = torch.tensor(doc_x_arr)


############################################################
# --- 6. Multi-Head, Multi-Layer Hetero GAT ---
############################################################

class HeteroGAT(nn.Module):
    """
    A multi-layer GAT architecture for heterogeneous data, 
    focusing on doc->doc, doc->term, term->doc edges.

    We define:
      - (num_layers) layers of HeteroConv, each containing GATConv 
        for doc->doc, doc->term, and term->doc.
      - multi-head attention with user-defined heads. 
        By default, PyG's GATConv concatenates attention heads.
      - a final linear layer to produce a single logit per doc node (binary classification).
    """

    def __init__(self, in_dim, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.5):
        """
        Parameters
        ----------
        in_dim : int
            Dimension of input features (doc.x dimension).
        hidden_dim : int
            Dimension of each head's output in GATConv (before concatenation).
        num_layers : int
            Number of GAT layers to stack.
        num_heads : int
            Number of attention heads in each GATConv.
        dropout : float
            Dropout probability applied after each layer.
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

        # Final linear classifier from the doc node's last-layer representation -> 1 logit
        final_dim = hidden_dim * num_heads  # dimension after the last GAT layer
        self.classify = nn.Linear(final_dim, 1)

    def forward(self, data: HeteroData):
        """
        Forward pass through 1 or 2 GAT layers, then classification.

        Parameters
        ----------
        data : HeteroData
            The PyG heterogeneous data object.

        Returns
        -------
        out_dict : dict
            Dictionary with node features after the final GAT layer (by node type).
        doc_logits : torch.Tensor
            The predicted logits for doc nodes (shape: [num_docs]).
        """
        x_dict, edge_index_dict, edge_attr_dict = (
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict,
        )

        # --- Layer 1 ---
        out1 = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        for nt in out1:
            out1[nt] = F.elu(out1[nt])  # apply activation
            out1[nt] = F.dropout(out1[nt], p=self.dropout, training=self.training)

        # If we only have 1 layer, we classify from out1["doc"]
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


############################################################
# --- 7. Full Pipeline Function ---
############################################################

def kafn_pulp_pipeline(
    data: KAFNData,
    ###############################
    # Doc2Vec parameters
    ###############################
    embed_dim=100,
    doc2vec_window=5,
    doc2vec_min_count=2,
    doc2vec_epochs=20,
    doc2vec_workers=os.cpu_count(),
    ###############################
    # Graph / Label Propagation
    ###############################
    k=5,  # top-k neighbors in doc-doc
    alpha_katz=0.01,  # Katz alpha
    m_iter=3,  # number of PU-LP iterations
    m_factor=1.0,  # lam (lambda) in iterative_pu_lp
    ###############################
    # YAKE parameters
    ###############################
    top_yake=5,
    yake_lan="en",
    yake_n=1,
    yake_dedup_lim=0.9,
    n_jobs=os.cpu_count(),
    ###############################
    # Multi-layer, multi-head GAT parameters
    ###############################
    gat_hidden_dim=64,
    gat_num_layers=2,
    gat_num_heads=4,
    gat_dropout=0.5,
    ###############################
    # Training parameters
    ###############################
    num_epochs=30,
    lr=1e-3,
    weight_decay=1e-4,  # L2 regularization
    device="cpu",
):
    """
    Full pipeline that:
      1) Trains Doc2Vec on the entire corpus.
      2) Builds a doc->doc similarity graph using top-k neighbors.
      3) Computes Katz matrix for global similarity, runs iterative PU label propagation.
      4) Extracts keywords via YAKE, adds doc->term edges, normalizes everything.
      5) Converts Nx to PyG HeteroData.
      6) Fills doc embeddings from the trained Doc2Vec model.
      7) Constructs and trains a multi-layer multi-head HeteroGAT.
      8) Evaluates on test set using the real labels.

    Parameters
    ----------
    data : KAFNData
        Data wrapper containing both train and test sets, including partial labels for train.
    embed_dim : int
        Size of the Doc2Vec embeddings.
    doc2vec_window : int
        Window size for Doc2Vec model.
    doc2vec_min_count : int
        Minimum frequency for words in Doc2Vec training.
    doc2vec_epochs : int
        Epochs to train the Doc2Vec model.
    doc2vec_workers : int
        Number of worker threads for Doc2Vec training.
    k : int
        Number of neighbors in doc->doc graph.
    alpha_katz : float
        Katz alpha controlling damping factor.
    m_iter : int
        Number of iterations for iterative PU-LP.
    m_factor : float
        The lam (lambda) factor used in iterative PU-LP for controlling set sizes.
    top_yake : int
        Number of keywords to extract per doc via YAKE.
    yake_lan : str
        Language code for YAKE ("en", "pt", etc.).
    yake_n : int
        n-gram size for YAKE keyword extraction.
    yake_dedup_lim : float
        Deduplication threshold in YAKE.
    n_jobs : int
        Parallelization for YAKE extraction.
    gat_hidden_dim : int
        Dimension for GAT hidden layers (per head).
    gat_num_layers : int
        How many GAT layers to stack.
    gat_num_heads : int
        Number of attention heads in each GAT layer.
    gat_dropout : float
        Dropout probability used in GAT layers.
    num_epochs : int
        Number of training epochs for the GAT model.
    lr : float
        Learning rate for optimizer.
    weight_decay : float
        L2 regularization factor.
    device : str
        "cpu" or "cuda". If "cuda" is available, GPU is used.
    """
    print("=== Training Doc2Vec ===")
    emb_model = Doc2VecEmb(
        vector_size=embed_dim,
        window=doc2vec_window,
        min_count=doc2vec_min_count,
        epochs=doc2vec_epochs,
        workers=doc2vec_workers,
    )
    emb_model.train(data)

    print("=== Building doc-doc graph ===")
    G_doc = build_doc_graph(data, emb_model, k=k)

    print("=== Computing Katz matrix ===")
    katz_mat, idx_map = compute_katz_matrix(G_doc, alpha=alpha_katz)
    if katz_mat is None:
        print("Katz failed, aborting.")
        return

    # Separate doc IDs based on pu_label
    pos_indices = [i for i in range(data.num_docs) if data.get_pu_label(i) == 1]
    unl_indices = [i for i in range(data.num_docs) if data.get_pu_label(i) == -1]

    print("=== Iterative PU Label Propagation ===")
    R_I, R_N, leftover = iterative_pu_lp(
        katz_mat, idx_map, pos_indices, unl_indices, m_iter=m_iter, lam=m_factor
    )
    R_I = set(R_I)
    R_N = set(R_N)
    print(f"  => Found R_I={len(R_I)}, R_N={len(R_N)}, leftover={len(leftover)}")

    print("=== Building doc-term edges ===")
    H_all = build_doc_term_edges(
        G_doc,
        data,
        top_k=top_yake,
        yake_lan=yake_lan,
        yake_n=yake_n,
        yake_dedup_lim=yake_dedup_lim,
        n_jobs=n_jobs,
    )

    print("=== Converting Nx -> HeteroData ===")
    hetero_data, doc_ids, term_ids = nx_to_heterodata(
        H_all, R_I, R_N, pos_indices, embed_dim=embed_dim
    )

    print("=== Filling doc embeddings ===")
    fill_doc_embeddings(hetero_data, doc_ids, data, emb_model)

    # Move data to chosen device (CPU or GPU)
    device_ = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    hetero_data = hetero_data.to(device_)

    print("=== Building Multi-Layer HeteroGAT model ===")
    in_dim = hetero_data["doc"].x.size(1)
    model = HeteroGAT(
        in_dim=in_dim,
        hidden_dim=gat_hidden_dim,
        num_layers=gat_num_layers,
        num_heads=gat_num_heads,
        dropout=gat_dropout,
    ).to(device_)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # We'll only compute loss on doc nodes that have label >=0
    doc_labels = hetero_data["doc"].labels
    labeled_mask = doc_labels >= 0
    labeled_y = doc_labels[labeled_mask].float().to(device_)

    print("=== Training ===")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        _, doc_logits = model(hetero_data)
        doc_logits_labeled = doc_logits[labeled_mask]
        loss = criterion(doc_logits_labeled, labeled_y)
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")

    print("=== Evaluation on test set ===")
    model.eval()
    with torch.no_grad():
        _, doc_logits = model(hetero_data)
        preds = (doc_logits.sigmoid() > 0.5).long().cpu().numpy()

    # Evaluate only on test docs. Gather test doc IDs, then collect predictions and true labels.
    test_docs = []
    test_true = []
    test_pred = []
    for i, did in enumerate(doc_ids):
        if data.is_test_doc(did):
            test_docs.append(did)
            true_lbl = data.get_binary_label(did)
            pred_lbl = preds[i]
            test_true.append(true_lbl)
            test_pred.append(pred_lbl)

    title = f"\n=== Keywords attention PU-LP with Doc2Vec, Labeled fraction {data.fraction_fake_labeled} ==="
    evaluation_metrics(test_true, test_pred, title=title)


def kafn_pulp_pipeline_tfidf(
    data: KAFNData,
    ###############################
    # TF-IDF parameters
    ###############################
    max_features=10000,
    ngram_range=(1, 1),
    min_df=2,
    stop_words="english",
    ###############################
    # Graph / Label Propagation
    ###############################
    k=5,
    alpha_katz=0.01,
    m_iter=3,
    m_factor=1.0,
    ###############################
    # YAKE parameters
    ###############################
    top_yake=5,
    yake_lan="en",
    yake_n=1,
    yake_dedup_lim=0.9,
    n_jobs=os.cpu_count(),
    ###############################
    # GAT parameters
    ###############################
    gat_hidden_dim=64,
    gat_num_layers=2,
    gat_num_heads=4,
    gat_dropout=0.5,
    ###############################
    # Training parameters
    ###############################
    num_epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    device="cpu",
):
    """
    Similar pipeline but using TF-IDF instead of Doc2Vec.
    """
    # 1) TF-IDF
    print("=== Computing TF-IDF ===")
    tfidf_model = TFIDFEmb(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words=stop_words,
    )
    tfidf_model.train(data)

    # 2) Build doc->doc graph (top-k neighbors) using TF-IDF vectors
    print("=== Building doc-doc graph ===")
    G_doc = build_doc_graph(data, tfidf_model, k=k)

    # 3) Katz + iterative PU-LP
    print("=== Computing Katz matrix ===")
    katz_mat, idx_map = compute_katz_matrix(G_doc, alpha=alpha_katz)
    if katz_mat is None:
        print("Katz failed, aborting.")
        return

    pos_indices = [i for i in range(data.num_docs) if data.get_pu_label(i) == 1]
    unl_indices = [i for i in range(data.num_docs) if data.get_pu_label(i) == -1]

    print("=== Iterative PU Label Propagation ===")
    R_I, R_N, leftover = iterative_pu_lp(
        katz_mat, idx_map, pos_indices, unl_indices, m_iter=m_iter, lam=m_factor
    )
    R_I = set(R_I)
    R_N = set(R_N)
    print(f"  => Found R_I={len(R_I)}, R_N={len(R_N)}, leftover={len(leftover)}")

    # 4) doc-term edges via YAKE
    print("=== Building doc-term edges ===")
    H_all = build_doc_term_edges(
        G_doc,
        data,
        top_k=top_yake,
        yake_lan=yake_lan,
        yake_n=yake_n,
        yake_dedup_lim=yake_dedup_lim,
        n_jobs=n_jobs,
    )

    # 5) Convert Nx -> HeteroData, using TF-IDF dimension
    print("=== Converting Nx -> HeteroData ===")
    hetero_data, doc_ids, term_ids = nx_to_heterodata(
        H_all,
        R_I,
        R_N,
        pos_indices,
        embed_dim=tfidf_model.doc_vectors.shape[1],
    )

    # 6) Fill doc features with TF-IDF vectors
    print("=== Filling doc embeddings ===")
    doc_x_list = []
    for d in doc_ids:
        emb = tfidf_model.get_vector(d)
        doc_x_list.append(emb)
    doc_x_arr = np.vstack(doc_x_list).astype(np.float32)
    hetero_data["doc"].x = torch.tensor(doc_x_arr)

    # 7) GAT
    device_ = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    hetero_data = hetero_data.to(device_)

    print("=== Building Multi-Layer HeteroGAT model ===")
    in_dim = hetero_data["doc"].x.size(1)
    model = HeteroGAT(
        in_dim=in_dim,
        hidden_dim=gat_hidden_dim,
        num_layers=gat_num_layers,
        num_heads=gat_num_heads,
        dropout=gat_dropout,
    ).to(device_)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    doc_labels = hetero_data["doc"].labels
    labeled_mask = doc_labels >= 0
    labeled_y = doc_labels[labeled_mask].float().to(device_)

    print("=== Training ===")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        _, doc_logits = model(hetero_data)
        doc_logits_labeled = doc_logits[labeled_mask]
        loss = criterion(doc_logits_labeled, labeled_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")

    # 8) Evaluate on test set
    print("=== Evaluation on test set ===")
    model.eval()
    with torch.no_grad():
        _, doc_logits = model(hetero_data)
        preds = (doc_logits.sigmoid() > 0.5).long().cpu().numpy()

    test_docs = []
    test_true = []
    test_pred = []
    for i, did in enumerate(doc_ids):
        if data.is_test_doc(did):
            test_docs.append(did)
            test_true.append(data.get_binary_label(did))
            test_pred.append(preds[i])

    title = f"\n=== Keywords attention PU-LP with TFIDF, Labeled fraction {data.fraction_fake_labeled} ==="
    evaluation_metrics(test_true, test_pred, title=title)


############################################################
# --- 8. Testing functions with parameter details ---
############################################################

def test_baseline(data: KAFNData):
    """
    Runs the simple baseline approach:
      - TF-IDF on full training data with pseudo-labels 
      - MultinomialNB for classification 
      - Evaluate on test set.

    No parameter arguments here, it just calls baseline(data) directly.
    """
    baseline(data)


def test_pulp_doc2vec_pipeline(data: KAFNData):
    """
    Runs the full KAFN-PULP pipeline with Doc2Vec embeddings 
    and a GAT-based classifier.

    Explanation of Parameters:
      - embed_dim=100: 
          Dimensionality of Doc2Vec embeddings.
      - doc2vec_window=5: 
          The max skip distance between the current and predicted word in Doc2Vec.
      - doc2vec_min_count=2: 
          Words with total frequency less than 2 are ignored.
      - doc2vec_epochs=30: 
          Train Doc2Vec for 30 epochs (slightly reduced from default).
      - doc2vec_workers=os.cpu_count(): 
          Use all CPU cores for parallel training.
      - k=5: 
          Each doc is connected to top-5 neighbors by cosine similarity.
      - alpha_katz=0.02: 
          Katz damping factor, slightly higher to consider longer paths.
      - m_iter=3: 
          Number of iterations in iterative PU-LP.
      - m_factor=1.0: 
          The lam factor controlling R_I and R_N expansions.
      - top_yake=5: 
          Extract 5 keywords from each doc with YAKE.
      - yake_lan="en": 
          Language used by YAKE is English.
      - yake_n=1: 
          Extract single-word keywords (n-grams of size 1).
      - yake_dedup_lim=0.9: 
          Deduplication threshold in YAKE for merging similar terms.
      - n_jobs=os.cpu_count(): 
          Parallel jobs for YAKE. 
      - gat_hidden_dim=64: 
          Hidden layer dimension for GAT (per head).
      - gat_num_layers=2: 
          We have 2 stacked GAT layers.
      - gat_num_heads=2: 
          Each GAT layer has 2 attention heads.
      - gat_dropout=0.5: 
          50% dropout after each GAT layer.
      - num_epochs=50: 
          Train the GAT for 50 epochs.
      - lr=1e-3: 
          Learning rate for Adam.
      - weight_decay=1e-4: 
          L2 regularization strength.
      - device="cuda": 
          Attempt to run on GPU if available, else fallback to CPU.
    """
    kafn_pulp_pipeline(
        data,
        # Doc2Vec
        embed_dim=100,
        doc2vec_window=5,
        doc2vec_min_count=2,
        doc2vec_epochs=30,  # fewer epochs
        doc2vec_workers=os.cpu_count(),
        # Graph / Katz / PU-LP
        k=5,
        alpha_katz=0.02,  # increase alpha to consider longer paths
        m_iter=3,
        m_factor=1.0,
        # YAKE
        top_yake=5,
        yake_lan="en",
        yake_n=1,  # single-word keywords only
        yake_dedup_lim=0.9,
        n_jobs=os.cpu_count(),
        # GAT
        gat_hidden_dim=64,  # smaller hidden dimension
        gat_num_layers=2,
        gat_num_heads=2,  # fewer heads
        gat_dropout=0.5,
        # Training
        num_epochs=50,
        lr=1e-3,
        weight_decay=1e-4,
        device="cuda",
    )


def test_pulp_tfidfvectorizer_pipeline(data: KAFNData):
    """
    Runs the full KAFN-PULP pipeline with TF-IDF embeddings 
    and a GAT-based classifier.

    Explanation of Parameters:
      - max_features=5000: 
          Restrict TF-IDF vocabulary to 5000 most frequent terms.
      - ngram_range=(1, 3): 
          Use unigrams, bigrams, and trigrams in TF-IDF.
      - min_df=2: 
          Ignore terms appearing in fewer than 2 docs.
      - stop_words="english": 
          Remove common English stop words.
      - k=5: 
          Connect each doc with its top-5 neighbors in doc->doc graph.
      - alpha_katz=0.02: 
          Katz alpha slightly increased for more global influence.
      - m_iter=3: 
          Number of PU-LP iterations.
      - m_factor=1.0: 
          The lam factor for the expansions in PU-LP.
      - top_yake=5: 
          5 keywords per doc from YAKE.
      - yake_lan="en": 
          English language for YAKE.
      - yake_n=1: 
          Single-word keywords.
      - yake_dedup_lim=0.9: 
          Deduplicate similar terms in YAKE.
      - n_jobs=os.cpu_count(): 
          Parallel CPU jobs for keyword extraction.
      - gat_hidden_dim=64: 
          Hidden dimension in each GATConv for each head.
      - gat_num_layers=2: 
          Stacking 2 GAT layers.
      - gat_num_heads=4: 
          4 attention heads.
      - gat_dropout=0.5: 
          50% dropout after each layer.
      - num_epochs=30: 
          Train GAT for 30 epochs.
      - lr=1e-3: 
          Adam learning rate.
      - weight_decay=1e-4: 
          L2 regularization.
      - device="cuda": 
          Use GPU if available, else CPU.
    """
    kafn_pulp_pipeline_tfidf(
        data,
        # TF-IDF parameters
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
        stop_words="english",
        # Graph / Katz / PU-LP
        k=5,
        alpha_katz=0.02,
        m_iter=3,
        m_factor=1.0,
        # YAKE
        top_yake=5,
        yake_lan="en",
        yake_n=1,
        yake_dedup_lim=0.9,
        n_jobs=os.cpu_count(),
        # GAT
        gat_hidden_dim=64,
        gat_num_layers=2,
        gat_num_heads=4,
        gat_dropout=0.5,
        # Training
        num_epochs=30,
        lr=1e-3,
        weight_decay=1e-4,
        device="cuda",
    )
