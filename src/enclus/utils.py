import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.neighbors
from clu import metrics
from flax import linen as nn
from flax import struct
from flax.training import train_state
from jax import random
import scipy.sparse


class MultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        batch_size = x.shape[0]
        
        # Ensure that the input dimensions match
        if x.shape[-1] != self.num_heads * self.head_dim:
            x = nn.Dense(self.num_heads * self.head_dim)(x)
        
        # Project onto query, key, and value
        qkv = nn.Dense(3 * self.num_heads * self.head_dim)(x)
        qkv = jnp.reshape(qkv, (batch_size, 3, self.num_heads, self.head_dim))
        queries, keys, values = jnp.transpose(qkv, (1, 0, 2, 3))
        
        # Calculate attention scores
        scale = jnp.sqrt(self.head_dim)
        scores = jnp.matmul(queries, jnp.transpose(keys, (0, 2, 1))) / scale
        attention = nn.softmax(scores)
        
        # attention = nn.Dropout(
        #     rate=self.dropout_rate
        # )(attention, deterministic=not training,rng=None)
        
        # compute output
        output = jnp.matmul(attention, values)
        output = jnp.reshape(output, (batch_size, -1))
        
        return output

class TransformerBlock(nn.Module):
    """Transformer Block"""
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # MultiHeadAttention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate
        )(x, training)
        
        # The first residual connection and layer normalization
        x = x + attention_output
        x = nn.LayerNorm()(x)
        
        # Feedforward network
        mlp_output = nn.Sequential([
            nn.Dense(self.mlp_dim),
            nn.gelu,
            # nn.Dropout(rate=self.dropout_rate,deterministic=not training),
            nn.Dense(x.shape[-1])
        ])(x)
        
        # The second residual connection and layer normalization
        x = x + mlp_output
        x = nn.LayerNorm()(x)
        
        return x
    

class FeedForward(nn.Module):
    """Feedforward network with an optional attention mechanism"""
    n_layers: int
    n_neurons: int
    n_output: int
    num_heads: int = 4
    head_dim: int = 64
    dropout_rate: float = 0.1
    use_attention: bool = True

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Ensure the input dimension is a multiple of head_dim
        n_neurons = self.n_neurons
        if self.use_attention:
            # Adjust the number of neurons to be a multiple of head_dim
            n_neurons = self.num_heads * self.head_dim
        
        # Initial dense layer
        x = nn.Dense(
            features=n_neurons,
            dtype=jnp.float32,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.LayerNorm(dtype=jnp.float32)(x)

        # Intermediate layers
        for _ in range(self.n_layers - 1):
            residual = x
            
            if self.use_attention:
                # Attention block
                x = TransformerBlock(
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    mlp_dim=n_neurons * 4,
                    dropout_rate=self.dropout_rate
                )(x, training)
            
            # Feedforward layer
            x = nn.Dense(
                features=n_neurons,
                dtype=jnp.float32,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros_init(),
            )(x)
            x = nn.leaky_relu(x)
            
            # Residual connection
            x = x + residual
            x = nn.LayerNorm(dtype=jnp.float32)(x)

        # Output layer
        output = nn.Dense(
            features=self.n_output,
            dtype=jnp.float32,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )(x)

        return output


class CVAE(nn.Module):
    """Conditional Variational Autoencoder with Attention Mechanism"""
    n_layers: int
    n_neurons: int
    n_latent: int
    n_output_exp: int
    n_output_cov: int
    num_heads: int = 4
    head_dim: int = 64
    dropout_rate: float = 0.1
    use_attention: bool = True

    def setup(self):
        self.encoder = FeedForward(
            n_layers=self.n_layers,
            n_neurons=self.n_neurons,
            n_output=self.n_latent * 2,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            use_attention=self.use_attention
        )

        self.decoder_exp = FeedForward(
            n_layers=self.n_layers,
            n_neurons=self.n_neurons,
            n_output=self.n_output_exp,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            use_attention=self.use_attention
        )

        self.decoder_cov = FeedForward(
            n_layers=self.n_layers,
            n_neurons=self.n_neurons,
            n_output=self.n_output_cov,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            use_attention=self.use_attention
        )

    def __call__(self, x, mode="spatial", key=random.key(0), training: bool = True):
        conf_const = 0 if mode == "spatial" else 1
        conf_neurons = jax.nn.one_hot(
            conf_const * jnp.ones(x.shape[0], dtype=jnp.int8), 2, dtype=jnp.float32
        )

        x_conf = jnp.concatenate([x, conf_neurons], axis=-1)
        enc_mu, enc_logstd = jnp.split(self.encoder(x_conf, training), 2, axis=-1)

        key, subkey = random.split(key)
        z = enc_mu + random.normal(key=subkey, shape=enc_logstd.shape) * jnp.exp(
            enc_logstd
        )
        z_conf = jnp.concatenate([z, conf_neurons], axis=-1)

        dec_exp = self.decoder_exp(z_conf, training)

        if mode == "spatial":
            dec_cov = self.decoder_cov(z_conf, training)
            return (enc_mu, enc_logstd, dec_exp, dec_cov)
        return (enc_mu, enc_logstd, dec_exp)


@struct.dataclass
class Metrics(metrics.Collection):
    """
    :meta private:
    """

    enc_loss: metrics.Average
    dec_loss: metrics.Average
    enc_corr: metrics.Average


class TrainState(train_state.TrainState):
    """
    :meta private:
    """

    metrics: Metrics


def MatSqrt(Mats):
    """
    :meta private:
    """

    e, v = np.linalg.eigh(Mats)
    e = np.where(e < 0, 0, e)
    e = np.sqrt(e)

    m, n = e.shape
    diag_e = np.zeros((m, n, n), dtype=e.dtype)
    diag_e.reshape(-1, n**2)[..., :: n + 1] = e
    return np.matmul(np.matmul(v, diag_e), v.transpose([0, 2, 1]))



def BatchKNN(data, batch, k):
    """
    :meta private:
    """

    kNNGraphIndex = np.zeros(shape=(data.shape[0], k))

    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]

        batch_knn = sklearn.neighbors.kneighbors_graph(
            data[val_ind], n_neighbors=k, mode="connectivity", n_jobs=-1
        ).tocoo()
        batch_knn_ind = np.reshape(
            np.asarray(batch_knn.col), [data[val_ind].shape[0], k]
        )
        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]

    return kNNGraphIndex.astype("int")


def CalcCovMats(spatial_data, kNN, genes, spatial_key="spatial", batch_key=-1):
    """
    Calculate the covariance matrix for each spot in spatial transcriptomics data.

    Parameters:
    - spatial_data (AnnData): AnnData object containing spatial transcriptomics data.
    - kNN (int): Number of nearest neighbors to consider for each spot.
    - genes (list or np.ndarray): List of genes to use for covariance calculation.
    - spatial_key (str, optional): Key in .obsm storing spatial coordinates, default "spatial".
    - batch_key (str or int, optional): If -1, ignore batch information; otherwise, use the specified .obs column for batch grouping, default -1.

    Returns:
    - CovMats (np.ndarray): Array of covariance matrices with shape (N, G, G), where N is the number of spots and G is the number of genes.
    """

    # Apply log transformation to selected gene expression data to reduce skewness and amplify small differences
    ExpData = np.log(spatial_data[:, genes].X + 1)

    if batch_key == -1:
        # When ignoring batch information, build a global k-nearest neighbors graph
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key],  # Spatial coordinates
            n_neighbors=kNN,                  # Number of neighbors
            mode="connectivity",              # Return connectivity only, no distances
            n_jobs=-1,                        # Use all available CPU cores
        ).tocoo()  # Convert to COO sparse matrix

        # Reshape the graph's column indices into a 2D array where each row corresponds to a spot and its k neighbors
        kNNGraphIndex = np.reshape(
            np.asarray(kNNGraph.col),
            [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
    else:
        # When considering batch information, use a custom BatchKNN function for within-batch neighbor computation
        kNNGraphIndex = BatchKNN(
            spatial_data.obsm[spatial_key],  # Spatial coordinates
            spatial_data.obs[batch_key],      # Batch labels
            kNN                               # Number of neighbors
        )

    # Compute the weighted distance matrix:
    # - ExpData.mean(axis=0)[None, None, :] computes the mean expression across all spots (shape (1, 1, G))
    # - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]] retrieves each spot's kNN expression data (shape (N, kNN, G))
    DistanceMatWeighted = (
        ExpData.mean(axis=0)[None, None, :]       # Global mean expression for each gene
        - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]]  # Difference from neighbor expression
    )

    # Calculate the covariance matrices by matrix-multiplying and normalizing:
    # - Transpose to shape (N, G, kNN), multiply by (N, kNN, G), then divide by (kNN - 1)
    CovMats = np.matmul(
        DistanceMatWeighted.transpose([0, 2, 1]),  # Shape (N, G, kNN)
        DistanceMatWeighted                       # Shape (N, kNN, G)
    ) / (kNN - 1)  # Resulting shape (N, G, G)

    # Add a small positive-definite perturbation to ensure numerical stability and positive definiteness
    CovMats = CovMats + CovMats.mean() * 1e-5 * np.expand_dims(
        np.identity(CovMats.shape[-1]), axis=0  # Identity matrix of shape (1, G, G)
    )

    # Return the computed covariance matrices
    return CovMats


def niche_cell_type(
    spatial_data, kNN, spatial_key="spatial", cell_type_key="cell_type", batch_key=-1
):
    """
    :meta private:
    """

    from sklearn.preprocessing import OneHotEncoder

    if batch_key == -1:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key],
            n_neighbors=kNN,
            mode="connectivity",
            n_jobs=-1,
        ).tocoo()
        knn_index = np.reshape(
            np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
    else:
        knn_index = BatchKNN(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN
        )

    one_hot_enc = OneHotEncoder().fit(
        np.asarray(list(set(spatial_data.obs[cell_type_key]))).reshape([-1, 1])
    )
    cell_type_one_hot = (
        one_hot_enc.transform(
            np.asarray(spatial_data.obs[cell_type_key]).reshape([-1, 1])
        )
        .reshape([spatial_data.obs["cell_type"].shape[0], -1])
        .todense()
    )

    cell_type_niche = pd.DataFrame(
        cell_type_one_hot[knn_index].sum(axis=1),
        index=spatial_data.obs_names,
        columns=list(one_hot_enc.categories_[0]),
    )
    return cell_type_niche

def compute_covet(
    spatial_data, k=8, g=64, genes=[], spatial_key="spatial", batch_key=-1
):
    """
    Compute niche covariance matrices for spatial data, run with scenclus.compute_covet

    :param spatial_data: (anndata) spatial data, with an obsm indicating spatial location of spot/segmented cell
    :param k: (int) number of nearest neighbours to define niche (default 8)
    :param g: (int) number of HVG to compute COVET representation on (default 64)
    :param genes: (list of str) list of genes to keep for niche covariance (default []
    :param spatial_key: (str) obsm key name with physical location of spots/cells (default 'spatial')
    :param batch_key: (str) obs key name of batch/sample of spatial data (default 'batch' if in spatial_data.obs, else -1)

    :return COVET: niche covariance matrices
    :return COVET_SQRT: matrix square-root of niche covariance matrices for approximate OT
    :return CovGenes: list of genes selected for COVET representation
    """
    if g == -1:
        CovGenes = spatial_data.var_names
    else:
        if "highly_variable" not in spatial_data.var.columns:
            if 'log' in spatial_data.layers.keys():
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")
            elif('log1p' in spatial_data.layers.keys()):
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log1p")
            elif(spatial_data.X.min() < 0):
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g)
            else:
                spatial_data.layers["log"] = np.log(spatial_data.X + 1)
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")

        CovGenes = np.asarray(spatial_data.var_names[spatial_data.var.highly_variable])
        if len(genes) > 0:
            CovGenes = np.union1d(CovGenes, genes)

    if batch_key not in spatial_data.obs.columns:
        batch_key = -1

    COVET = CalcCovMats(
        spatial_data, k, genes=CovGenes, spatial_key=spatial_key, batch_key=batch_key
    )
    # print(COVET.shape)
    COVET_SQRT = MatSqrt(COVET)
    return (
        COVET.astype("float32"),
        COVET_SQRT.astype("float32"),
        np.asarray(CovGenes).astype("str"),
    )
