import os
import json


class ModelConfig:
    """
    Utility class. Holds configuration for model used in kafn pulp pipeline
    """

    def __init__(
        self,
        # Doc2Vec params
        embed_dim=100,
        doc2vec_window=5,
        doc2vec_min_count=2,
        doc2vec_epochs=20,
        doc2vec_workers=os.cpu_count(),
        # TF-IDF parameters
        max_features=10000,
        ngram_range=(1, 1),
        min_df=2,
        stop_words="english",
        # Graph / Label Propagation
        k=5,
        alpha_katz=0.01,
        m_iter=3,
        m_factor=1.0,
        # YAKE params
        top_yake=5,
        yake_lan="en",
        yake_n=1,
        yake_dedup_lim=0.9,
        n_jobs=os.cpu_count(),
        # Multi-layer, multi-head GAT params
        gat_hidden_dim=64,
        gat_num_layers=2,
        gat_num_heads=4,
        gat_dropout=0.5,
        # Training parameters:
        num_epochs=30,
        lr=1e-3,
        weight_decay=1e-4,
        # Device:
        device="cpu",
        pipeline_data_dir=f"model/0.0/",
        result_dir="result/",
        artifact_prefix="kafn_pulp_doc2vec",
    ):
        """_summary_

        Args:
            embed_dim (int, optional): Size of the Doc2Vec embeddings. Defaults to 100.
            doc2vec_window (int, optional): Window size for Doc2Vec model. Defaults to 5.
            doc2vec_min_count (int, optional): Minimum frequency for words in Doc2Vec training. Defaults to 2.
            doc2vec_epochs (int, optional): Epochs to train the Doc2Vec model. Defaults to 20.
            doc2vec_workers (_type_, optional): Number of worker threads for Doc2Vec training. Defaults to os.cpu_count().

            max_features (int, optional): Maximum number of features (vocabulary size) in TF-IDF. Defaults to 10000.
            ngram_range (tuple, optional): The lower and upper boundary of the n-grams to be extracted. Defaults to (1, 1).
            min_df (int, optional): Ignore terms that appear in fewer than min_df documents. Defaults to 2.
            stop_words (str, optional): Stop words to remove from the text. Defaults to "english".

            k (int, optional): Number of neighbors in doc->doc graph. Defaults to 5.
            alpha_katz (float, optional): Katz alpha controlling damping factor. Defaults to 0.01.
            m_iter (int, optional): Number of iterations for iterative PU-LP. Defaults to 3.
            m_factor (float, optional): The lambda factor used in iterative PU-LP for controlling set sizes. Defaults to 1.0.
            top_yake (int, optional): Number of keywords to extract per doc via YAKE. Defaults to 5.
            yake_lan (str, optional): Language code for YAKE ("en", "fr", etc.). Defaults to "en".
            yake_n (int, optional): n-gram size for YAKE keyword extraction. Defaults to 1.
            yake_dedup_lim (float, optional): Deduplication threshold in YAKE. Defaults to 0.9.
            n_jobs (_type_, optional): Parallelization for YAKE extraction. Defaults to os.cpu_count().
            gat_hidden_dim (int, optional): Dimension for GAT hidden layers (per head). Defaults to 64.
            gat_num_layers (int, optional): How many GAT layers to stack. Defaults to 2.
            gat_num_heads (int, optional): Number of attention heads in each GAT layer. Defaults to 4.
            gat_dropout (float, optional): Dropout probability used in GAT layers. Defaults to 0.5.
            num_epochs (int, optional): Number of training epochs for the GAT model. Defaults to 30.
            lr (_type_, optional): Learning rate for optimizer. Defaults to 1e-3.
            weight_decay (_type_, optional): L2 regularization factor. Defaults to 1e-4.
            device (str, optional): "cpu" or "cuda". If "cuda" is available, GPU is used. Defaults to "cpu". "mps" not supported.
            pipeline_data_dir (_type_, optional): Directory where pipeline data will be saved. Defaults to f"model/0.0/".
            result_dir (str, optional): Directory where results will be saved . Defaults to "result/".
            artifact_prefix (str, optional): Model prefix. Defaults to "kafn_pulp_doc2vec".
        """

        self.embed_dim = embed_dim
        self.doc2vec_window = doc2vec_window
        self.doc2vec_min_count = doc2vec_min_count
        self.doc2vec_epochs = doc2vec_epochs
        self.doc2vec_workers = doc2vec_workers
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.stop_words = stop_words
        self.k = k
        self.alpha_katz = alpha_katz
        self.m_iter = m_iter
        self.m_factor = m_factor
        self.top_yake = top_yake
        self.yake_lan = yake_lan
        self.yake_n = yake_n
        self.yake_dedup_lim = yake_dedup_lim
        self.n_jobs = n_jobs
        self.gat_hidden_dim = gat_hidden_dim
        self.gat_num_layers = gat_num_layers
        self.gat_num_heads = gat_num_heads
        self.gat_dropout = gat_dropout
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.pipeline_data_dir = pipeline_data_dir
        self.result_dir = result_dir
        self.artifact_prefix = artifact_prefix
        os.makedirs(os.path.dirname(pipeline_data_dir), exist_ok=True)
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        self.config_filename = (
            f"{self.pipeline_data_dir}{self.artifact_prefix}_model_config.json"
        )
        self.model_filename = f"{self.pipeline_data_dir}{self.artifact_prefix}.pth"
        self.graph_filename = (
            f"{self.pipeline_data_dir}{self.artifact_prefix}_graph.pkl"
        )
        self.hetero_graph_filename = (
            f"{self.pipeline_data_dir}{self.artifact_prefix}_hetero_graph.pkl"
        )
        self.doc2vec_filename = (
            f"{self.pipeline_data_dir}{self.artifact_prefix}_gensim.bin"
        )
        self.tfidf_model_filename = (
            f"{self.pipeline_data_dir}{self.artifact_prefix}_tfidf.pkl"
        )
        self.doc_emb_path = (
            f"{self.pipeline_data_dir}{self.artifact_prefix}_embeddings.npy"
        )

    @classmethod
    def load(cls, filename: str):
        """
        Loads model configuration from a file

        Args:
            filename (str): _description_

        Returns:
            _type_: _description_
        """
        with open(filename, "r") as f:
            config = json.load(f)
        instance = cls()
        instance.__dict__.update(config)
        return instance
