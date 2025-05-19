
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import scanpy as sc
import sklearn.neighbors
import tensorflow_probability.substrates.jax as jax_prob
from flax import linen as nn
from jax import jit, random
from tqdm import trange

from enclus._dists import (
    KL,
    AOT_Distance,
    log_nb_pdf,
    log_normal_pdf,
    log_pos_pdf,
    log_zinb_pdf,
)

from enclus.utils import CVAE, Metrics, TrainState, compute_covet, niche_cell_type


class ENCLUS:
    def __init__(
        self,
        spatial_data,
        sc_data,
        spatial_key="spatial",
        batch_key="batch",
        num_layers=3,
        num_neurons=1024,
        latent_dim=512,
        k_nearest=16,
        num_cov_genes=64,
        cov_genes=[],
        num_HVG=2048,
        sc_genes=[],
        spatial_dist="pois",
        sc_dist="nb",

        spatial_coeff=1.0,
        sc_coeff=1.0,
        cov_coeff=1.0,
        kl_coeff=0.1,
        log_input=0.1,
        stable_eps=1e-6,

        adaptive_weights=False,
        early_stopping=True,
        patience=30,

        cluster_reg_coeff=0.1,
        distance_metric="euclidean",
        n_clusters=6,
        tau=0.1,
        alpha=2.0,
        gamma=0.1,

        num_heads=8,
        head_dim=64,
        use_attention=True,
        dropout_rate=0.1,
    ):
        self.cluster_reg_coeff = cluster_reg_coeff
        self.distance_metric = distance_metric
        self.adaptive_weights = adaptive_weights
        self.early_stopping = early_stopping
        self.patience = patience

        self.loss_weights = {
            'spatial': 1.0,
            'sc': 1.0,
            'cov': 1.0,
            'kl': 1.0,
            'cluster': 1.0,
        }

        self.spatial_data = spatial_data[:, np.intersect1d(spatial_data.var_names, sc_data.var_names)]
        self.sc_data = sc_data

        if "highly_variable" not in self.sc_data.var.columns:
            if 'log' in self.sc_data.layers.keys():
                sc.pp.highly_variable_genes(self.sc_data, n_top_genes=num_HVG, layer="log")
            elif('log1p' in self.sc_data.layers.keys()):
                sc.pp.highly_variable_genes(self.sc_data, n_top_genes=num_HVG, layer="log1p")
            elif(self.sc_data.X.min() < 0):
                sc.pp.highly_variable_genes(self.sc_data, n_top_genes=num_HVG)
            else:
                sc_data.layers["logp"] = np.log(self.sc_data.X + 1)
                sc.pp.highly_variable_genes(self.sc_data, n_top_genes=num_HVG, layer="logp")

        sc_genes_keep = np.union1d(
            self.sc_data.var_names[self.sc_data.var.highly_variable], self.spatial_data.var_names
        )
        if len(sc_genes) > 0:
            sc_genes_keep = np.union1d(sc_genes_keep, sc_genes)
            
        # print("sc_genes_keep shape:",sc_genes_keep.shape)

        if self.sc_data.raw is None:
            self.sc_data.raw = self.sc_data

        self.sc_data = self.sc_data[:, sc_genes_keep]

        self.overlap_genes = np.asarray(
            np.intersect1d(self.spatial_data.var_names, self.sc_data.var_names)
        )
        self.non_overlap_genes = np.asarray(
            list(set(self.sc_data.var_names) - set(self.spatial_data.var_names))
        )
        
        self.spatial_data = self.spatial_data[:, list(self.overlap_genes)]
        self.sc_data = self.sc_data[
            :, list(self.overlap_genes) + list(self.non_overlap_genes)
        ]
        print("sc_data shape and st_data shape:",self.sc_data.X.shape,self.spatial_data.X.shape)
        if batch_key not in spatial_data.obs.columns:
            batch_key = -1

        self.k_nearest = k_nearest
        self.spatial_key = spatial_key
        self.batch_key = batch_key
        self.cov_genes = cov_genes
        self.num_cov_genes = num_cov_genes
        self.gamma = gamma  # Weight for clustering loss
        self.tau = tau  # Scaling parameter for Deep K-Means
        self.alpha = alpha  # Exponent for Deep K-Means
        self.n_clusters = n_clusters
    

        # print("Computing Niche Covariance Matrices")

        (
            self.spatial_data.obsm["COVET"],
            self.spatial_data.obsm["COVET_SQRT"],
            self.CovGenes,
        ) = compute_covet(
            self.spatial_data,
            self.k_nearest,
            self.num_cov_genes,
            self.cov_genes,
            spatial_key=self.spatial_key,
            batch_key=self.batch_key,
        )

        self.overlap_num = self.overlap_genes.shape[0]
        self.cov_gene_num = self.spatial_data.obsm["COVET_SQRT"].shape[-1]
        self.full_trans_gene_num = self.sc_data.shape[-1]

        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.latent_dim = latent_dim

        self.spatial_dist = spatial_dist
        self.sc_dist = sc_dist

        self.dist_size_dict = {"pois": 1, "nb": 2, "zinb": 3, "norm": 1}

        self.exp_dec_size = (
            self.dist_size_dict[self.sc_dist] * self.sc_data.shape[-1]
            + (self.dist_size_dict[self.spatial_dist] - 1) * self.spatial_data.shape[-1]
        )

        self.spatial_coeff = spatial_coeff
        self.sc_coeff = sc_coeff
        self.cov_coeff = cov_coeff
        self.kl_coeff = kl_coeff

        if self.sc_dist == "norm" or self.spatial_dist == "norm" or self.spatial_data.X.min()<0 or self.sc_data.X.min()<0:
            self.log_input = -1
        else:
            self.log_input = log_input

        self.eps = stable_eps

        print("Initializing CVAE")

        self.model = CVAE(
            n_layers=self.num_layers,
            n_neurons=self.num_neurons,
            n_latent=self.latent_dim,
            n_output_exp=self.exp_dec_size,
            n_output_cov=int(self.cov_gene_num * (self.cov_gene_num + 1) / 2),
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )

        print("Finished Initializing ENCLUS")
    

    def deep_kmeans_loss(self, Z):
        # Compute distances
        if self.distance_metric == "euclidean":
            distances = self.tau * jnp.sum((Z[:, None, :] - self.cluster_centers[None, :, :]) ** 2, axis=-1)
        else:  # cosine similarity
            Z_normalized = Z / jnp.linalg.norm(Z, axis=-1, keepdims=True)
            C_normalized = self.cluster_centers / jnp.linalg.norm(self.cluster_centers, axis=-1, keepdims=True)
            distances = -self.tau * jnp.dot(Z_normalized, C_normalized.T)
        
        distances = jnp.clip(distances, a_min=self.eps, a_max=None)
        
        # Compute soft assignments
        logits = -distances
        logits = logits - jnp.max(logits, axis=1, keepdims=True)
        q = jnp.exp(logits)
        q = q / (jnp.sum(q, axis=1, keepdims=True) + self.eps)
        q = jnp.power(q, self.alpha)
        q = q / (jnp.sum(q, axis=1, keepdims=True) + self.eps)
        
        # Compute clustering loss
        weighted_distances = distances * q
        clustering_loss = jnp.mean(jnp.sum(weighted_distances, axis=1))
        
        # Add cluster balance regularization
        cluster_sizes = jnp.mean(q, axis=0)
        cluster_reg = -jnp.mean(jnp.log(cluster_sizes + self.eps))
        
        return clustering_loss + self.cluster_reg_coeff * cluster_reg


    def _initialize_cluster_centers(self, state):
        """Initialize cluster centers"""
        key = random.key(0)
        key1, key2 = random.split(key)
        # Obtain initial latent representations
        spatial_enc_mu, _, _, _ = state.apply_fn(
            {"params": state.params},
            x=self.inp_log_fn(self.spatial_data.X),
            mode="spatial",
            key=key1,
        )
        sc_enc_mu, _, _ = state.apply_fn(
            {"params": state.params},
            x=self.inp_log_fn(self.sc_data[:, self.spatial_data.var_names].X),
            mode="sc",
            key=key2,
        )

        # Convert to numpy arrays
        sc_latent = np.asarray(sc_enc_mu)
        spatial_latent = np.asarray(spatial_enc_mu)
        combined_latent = np.concatenate([sc_latent, spatial_latent], axis=0)
        
        # Initialize cluster centers using K-means++
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=30, init='k-means++')
        kmeans.fit(combined_latent)
        self.cluster_centers = jnp.array(kmeans.cluster_centers_)

        
    def _get_batch_indices(self, key1, key2, batch_size):
        """Get batch indices"""
        batch_spatial_ind = random.choice(
            key=key1,
            a=self.spatial_data.shape[0],
            shape=[batch_size],
            replace=False,
        )
        batch_sc_ind = random.choice(
            key=key2,
            a=self.sc_data.shape[0],
            shape=[batch_size],
            replace=False,
        )
        return batch_spatial_ind, batch_sc_ind

    def _get_batch_data(self, batch_indices):
        """Get batch data"""
        batch_spatial_ind, batch_sc_ind = batch_indices
        
        batch_spatial_exp = self.spatial_data.X[batch_spatial_ind]
        batch_spatial_cov = self.spatial_data.obsm["COVET_SQRT"][batch_spatial_ind]
        batch_sc_exp = self.sc_data.X[batch_sc_ind]
        
        return batch_spatial_exp, batch_spatial_cov, batch_sc_exp

    def _update_loss_weights(self, metrics_history):
        """Update loss weights"""
        # Compute moving average of each loss term
        avg_metrics = np.mean(metrics_history, axis=0)
        
        # Compute relative weights
        total_loss = np.sum(np.abs(avg_metrics))
        if total_loss > 0:
            weights = {
                'spatial': 2.0 * total_loss / (6.0 * np.abs(avg_metrics[1]) + 1e-8),
                'sc': 2.0 * total_loss / (6.0 * np.abs(avg_metrics[0]) + 1e-8),
                'cov': 2.0 * total_loss / (6.0 * np.abs(avg_metrics[2]) + 1e-8),
                'kl': 2.0 * total_loss / (6.0 * np.abs(avg_metrics[3]) + 1e-8),
                'cluster': 2.0 * total_loss / (6.0 * np.abs(avg_metrics[4]) + 1e-8),
            }
            
            # Clip weights to prevent numerical instability
            for k in weights:
                weights[k] = np.clip(weights[k], 0.1, 10.0)
            
            # Smooth update
            for k in weights:
                self.loss_weights[k] = 0.9 * self.loss_weights[k] + 0.1 * weights[k]

    def _print_training_progress(self, metrics, tq):
        """print training progress"""
        metric_names = ['sc', 'spatial', 'cov', 'kl', 'cluster', 
                        ]
        print_statement = ""
        # if self.adaptive_weights:
        weight_statement = " | " + " ".join([f"{k}_w: {v:.2f}" for k, v in self.loss_weights.items()])
        print_statement += weight_statement
        
        tq.set_description(print_statement)
            
        tq.refresh()

    def inp_log_fn(self, x):
        """
        :meta private:
        """

        if self.log_input > 0:
            return jnp.log(x + self.log_input)
        return x

    def mean_sc(self, sc_inp):
        """
        :meta private:
        """

        sc_inp = sc_inp[:, : self.dist_size_dict[self.sc_dist] * self.sc_data.shape[-1]]
        if self.sc_dist == "zinb":
            sc_r, sc_p, sc_d = jnp.split(sc_inp, 3, axis=-1)
            return nn.softplus(sc_r) * jnp.exp(sc_p) * (1 - nn.sigmoid(sc_d))
        if self.sc_dist == "nb":
            sc_r, sc_p = jnp.split(sc_inp, 2, axis=-1)
            return nn.softplus(sc_r) * jnp.exp(sc_p)
        if self.sc_dist == "pois":
            sc_l = sc_inp
            return sc_l
        if self.sc_dist == "norm":
            sc_l = sc_inp
            return sc_l

    def mean_spatial(self, spatial_inp):
        """
        :meta private:
        """

        if self.spatial_dist == "zinb" or self.spatial_dist == "nb":
            spatial_inp = jnp.concatenate(
                [
                    spatial_inp[:, : self.spatial_data.shape[-1]],
                    spatial_inp[
                        :,
                        -(self.dist_size_dict[self.spatial_dist] - 1)
                        * self.spatial_data.shape[-1] :,
                    ],
                ],
                axis=-1,
            )
        else:
            spatial_inp = spatial_inp[:, : self.spatial_data.shape[-1]]

        if self.spatial_dist == "zinb":
            spatial_r, spatial_p, spatial_d = jnp.split(spatial_inp, 3, axis=-1)
            return (
                nn.softplus(spatial_r)
                * jnp.exp(spatial_p)
                * (1 - nn.sigmoid(spatial_d))
            )
        if self.spatial_dist == "nb":
            spatial_r, spatial_p = jnp.split(spatial_inp, 2, axis=-1)
            return nn.softplus(spatial_r) * jnp.exp(spatial_p)
        if self.spatial_dist == "pois":
            spatial_l = spatial_inp
            return spatial_l
        if self.spatial_dist == "norm":
            spatial_l = spatial_inp
            return spatial_l

    def factor_sc(self, sc_inp, dec_exp):
        """
        :meta private:
        """

        sc_neurons = dec_exp[
            :, : self.dist_size_dict[self.sc_dist] * self.sc_data.shape[-1]
        ]

        if self.sc_dist == "zinb":
            sc_r, sc_p, sc_d = jnp.split(sc_neurons, 3, axis=-1)
            sc_like = jnp.mean(
                log_zinb_pdf(sc_inp, nn.softplus(sc_r) + self.eps, sc_p, sc_d)
            )
        if self.sc_dist == "nb":
            sc_r, sc_p = jnp.split(sc_neurons, 2, axis=-1)
            sc_like = jnp.mean(log_nb_pdf(sc_inp, nn.softplus(sc_r) + self.eps, sc_p))
        if self.sc_dist == "pois":
            sc_l = sc_neurons
            sc_like = jnp.mean(log_pos_pdf(sc_inp, nn.softplus(sc_l) + self.eps))
        if self.sc_dist == "norm":
            sc_l = sc_neurons
            sc_like = jnp.mean(log_normal_pdf(sc_inp, sc_l))
        return sc_like

    def factor_spatial(self, spatial_inp, dec_exp):
        """
        :meta private:
        """

        if self.spatial_dist == "zinb" or self.spatial_dist == "nb":
            spatial_neurons = jnp.concatenate(
                [
                    dec_exp[:, : self.spatial_data.shape[-1]],
                    dec_exp[
                        :,
                        -(self.dist_size_dict[self.spatial_dist] - 1)
                        * self.spatial_data.shape[-1] :,
                    ],
                ],
                axis=-1,
            )
        else:
            spatial_neurons = dec_exp[:, : self.spatial_data.shape[-1]]

        if self.spatial_dist == "zinb":
            spatial_r, spatial_p, spatial_d = jnp.split(spatial_neurons, 3, axis=-1)
            spatial_like = jnp.mean(
                log_zinb_pdf(
                    spatial_inp, nn.softplus(spatial_r) + self.eps, spatial_p, spatial_d
                )
            )
        if self.spatial_dist == "nb":
            spatial_r, spatial_p = jnp.split(spatial_neurons, 2, axis=-1)
            spatial_like = jnp.mean(
                log_nb_pdf(spatial_inp, nn.softplus(spatial_r) + self.eps, spatial_p)
            )
        if self.spatial_dist == "pois":
            spatial_l = spatial_neurons
            spatial_like = jnp.mean(
                log_pos_pdf(spatial_inp, nn.softplus(spatial_l) + self.eps)
            )
        if self.spatial_dist == "norm":
            spatial_l = spatial_neurons
            spatial_like = jnp.mean(log_normal_pdf(spatial_inp, spatial_l))
        return spatial_like

    def grammian_cov(self, dec_cov):
        """
        :meta private:
        """

        dec_cov = jax_prob.math.fill_triangular(dec_cov)
        return jnp.matmul(dec_cov, dec_cov.transpose([0, 2, 1]))

    def create_train_state(self, key=random.key(0), init_lr=0.0001, decay_steps=4000):
        """
        :meta private:
        """

        key, subkey1, subkey2 = random.split(key, num=3)
        params = self.model.init(
            rngs={"params": subkey1},
            x=self.inp_log_fn(self.spatial_data.X[0:1]),
            mode="spatial",
            key=subkey2,
        )["params"]

        lr_sched = optax.exponential_decay(init_lr, decay_steps, 0.5, staircase=True)
        tx = optax.adam(lr_sched)  #

        return TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx, metrics=Metrics.empty()
        )

    @partial(jit, static_argnums=(0,))
    def train_step(self, state, spatial_inp, spatial_COVET, sc_inp, key=random.key(0)):
        key, subkey1, subkey2 = random.split(key, num=3)

        def loss_fn(params):
            spatial_enc_mu, spatial_enc_logstd, spatial_dec_exp, spatial_dec_cov = state.apply_fn(
                {"params": params},
                x=self.inp_log_fn(spatial_inp),
                mode="spatial",
                key=subkey1,
            )
            
            sc_enc_mu, sc_enc_logstd, sc_dec_exp = state.apply_fn(
                {"params": params},
                x=self.inp_log_fn(sc_inp[:, : spatial_inp.shape[-1]]),
                mode="sc",
                key=subkey2,
            )

            spatial_exp_like = self.factor_spatial(spatial_inp, spatial_dec_exp)
            sc_exp_like = self.factor_sc(sc_inp, sc_dec_exp)
            spatial_cov_like = jnp.mean(AOT_Distance(spatial_COVET, self.grammian_cov(spatial_dec_cov)))
            kl_div = jnp.mean(KL(spatial_enc_mu, spatial_enc_logstd)) + jnp.mean(KL(sc_enc_mu, sc_enc_logstd))
            combined_latent = jnp.concatenate([sc_enc_mu,spatial_enc_mu], axis=0)
            clustering_loss = self.deep_kmeans_loss(combined_latent)

            w = self.loss_weights
            loss = (
                -w['spatial'] * self.spatial_coeff * spatial_exp_like
                - w['sc'] * self.sc_coeff * sc_exp_like
                - w['cov'] * self.cov_coeff * spatial_cov_like
                + w['kl'] * self.kl_coeff * kl_div
                + w['cluster'] * self.gamma * clustering_loss
            )

            metrics = [
                sc_exp_like,
                spatial_exp_like,
                spatial_cov_like,
                kl_div,
                clustering_loss,
            ]
            
            return loss, metrics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, (loss, metrics)

    
    def train(
        self,
        training_steps=1000,
        batch_size=1024,
        verbose=16,
        init_lr=0.0001,
        decay_steps=5000,
        key=random.key(0),
    ):

        batch_size = min(self.sc_data.shape[0], min(self.spatial_data.shape[0], batch_size))
        
        key, subkey = random.split(key)
        state = self.create_train_state(subkey, init_lr=init_lr, decay_steps=decay_steps)
        self.params = state.params
        
        print("Initializing cluster centers...")
        self._initialize_cluster_centers(state)
        
        tq = trange(training_steps, leave=True, desc="")
        best_loss = float('inf')
        patience_counter = 0
        metrics_history = []
        
        for training_step in tq:
            key, subkey1, subkey2 = random.split(key, num=3)
            batch_indices = self._get_batch_indices(subkey1, subkey2, batch_size)
            batch_data = self._get_batch_data(batch_indices)
            
            state, (loss, metrics) = self.train_step(state, *batch_data, key=key)
            metrics_history.append(metrics)
            
            if loss < best_loss:
                best_loss = loss
                best_params = state.params
                patience_counter = 0
            else:
                patience_counter += 1
            
            if self.early_stopping and patience_counter >= self.patience:
                print("\nEarly stopping triggered")
                self.params = best_params
                break
            
            if self.adaptive_weights and training_step % verbose == 0:
                self._update_loss_weights(metrics_history[-verbose:])
            
            if training_step % verbose == 0:
                self._print_training_progress(metrics, tq)
        
        self.latent_rep()

    
    # @partial(jit, static_argnums=(0,))
    def model_encoder(self, x):
        """
        :meta private:
        """

        return self.model.bind({"params": self.params}).encoder(x)

    # @partial(jit, static_argnums=(0,))
    def model_decoder_exp(self, x):
        """
        :meta private:
        """

        return self.model.bind({"params": self.params}).decoder_exp(x)

    # @partial(jit, static_argnums=(0,))
    def model_decoder_cov(self, x):
        """
        :meta private:
        """
        return self.model.bind({"params": self.params}).decoder_cov(x)

    def encode(self, x, mode="spatial", max_batch=64):
        """
        :meta private:
        """

        conf_const = 0 if mode == "spatial" else 1
        conf_neurons = jax.nn.one_hot(
            conf_const * jnp.ones(x.shape[0], dtype=jnp.int8), 2, dtype=jnp.float32
        )

        x_conf = jnp.concatenate([self.inp_log_fn(x), conf_neurons], axis=-1)

        if x_conf.shape[0] < max_batch:
            enc = jnp.split(self.model_encoder(x_conf), 2, axis=-1)[0]
        else:  # For when the GPU can't pass all point-clouds at once
            num_split = int(x_conf.shape[0] / max_batch) + 1
            x_conf_split = np.array_split(x_conf, num_split)
            enc = np.concatenate(
                [
                    jnp.split(self.model_encoder(x_conf_split[split_ind]), 2, axis=-1)[
                        0
                    ]
                    for split_ind in range(num_split)
                ],
                axis=0,
            )
        return enc

    def decode_exp(self, x, mode="spatial", max_batch=64):
        """
        :meta private:
        """

        conf_const = 0 if mode == "spatial" else 1
        conf_neurons = jax.nn.one_hot(
            conf_const * jnp.ones(x.shape[0], dtype=jnp.int8), 2, dtype=jnp.float32
        )

        x_conf = jnp.concatenate([x, conf_neurons], axis=-1)

        if mode == "spatial":
            if x_conf.shape[0] < max_batch:
                dec = self.mean_spatial(self.model_decoder_exp(x_conf))
            else:  # For when the GPU can't pass all point-clouds at once
                num_split = int(x_conf.shape[0] / max_batch) + 1
                x_conf_split = np.array_split(x_conf, num_split)
                dec = np.concatenate(
                    [
                        self.mean_spatial(
                            self.model_decoder_exp(x_conf_split[split_ind])
                        )
                        for split_ind in range(num_split)
                    ],
                    axis=0,
                )
        else:
            if x_conf.shape[0] < max_batch:
                dec = self.mean_sc(
                    self.model.bind({"params": self.params}).decoder_exp(x_conf)
                )
            else:  # For when the GPU can't pass all point-clouds at once
                num_split = int(x_conf.shape[0] / max_batch) + 1
                x_conf_split = np.array_split(x_conf, num_split)
                dec = np.concatenate(
                    [
                        self.mean_sc(self.model_decoder_exp(x_conf_split[split_ind]))
                        for split_ind in range(num_split)
                    ],
                    axis=0,
                )
        return dec

    def decode_cov(self, x, max_batch=64):
        """
        :meta private:
        """

        conf_const = 0
        conf_neurons = jax.nn.one_hot(
            conf_const * jnp.ones(x.shape[0], dtype=jnp.int8), 2, dtype=jnp.float32
        )

        x_conf = jnp.concatenate([x, conf_neurons], axis=-1)

        if x_conf.shape[0] < max_batch:
            dec = self.grammian_cov(self.model_decoder_cov(x_conf))
        else:  # For when the GPU can't pass all point-clouds at once
            num_split = int(x_conf.shape[0] / max_batch) + 1
            x_conf_split = np.array_split(x_conf, num_split)
            dec = np.concatenate(
                [
                    self.grammian_cov(self.model_decoder_cov(x_conf_split[split_ind]))
                    for split_ind in range(num_split)
                ],
                axis=0,
            )
        return dec

    def latent_rep(self):
        """
        Compute latent embeddings for spatial and single cell data, automatically performed after training

        :return: nothing, adds 'enclus_latent' self.spatial_data.obsm and self.spatial_data.obsm
        """

        self.spatial_data.obsm["enclus_latent"] = np.asarray(self.encode(
            self.spatial_data.X, mode="spatial"
        ))
        self.sc_data.obsm["enclus_latent"] = np.asarray(self.encode(
            self.sc_data[:, self.spatial_data.var_names].X, mode="sc"
        ))

    def impute_genes(self):
        """
        Impute full transcriptome for spatial data

        :return: nothing, adds 'imputation' to self.spatial_data.obsm
        """

        self.spatial_data.obsm["imputation"] = pd.DataFrame(
            self.decode_exp(self.spatial_data.obsm["enclus_latent"], mode="sc"),
            columns=self.sc_data.var_names,
            index=self.spatial_data.obs_names,
        )

        print(
            "Finished imputing missing gene for spatial data! See 'imputation' in obsm of ENCLUS.spatial_data"
        )

    def reconstruct_genes(self):
        """
        Impute full transcriptome for spatial data

        :return: nothing, adds 'imputation' to self.spatial_data.obsm
        """

        self.sc_data.obsm["reconstruct"] = pd.DataFrame(
            self.decode_exp(self.sc_data.obsm["enclus_latent"], mode="sc"),
            columns=self.sc_data.var_names,
            index=self.sc_data.obs_names,
        )

        print(
            "Finished reconstruct gene for sc data! See 'reconstruct' in obsm of enclus.sc_data"
        )

    def infer_niche_covet(self):
        """
        Predict COVET representation for single-cell data

        :return: nothing, adds 'COVET_SQRT' and 'COVET' to self.sc_data.obsm
        """

        self.sc_data.obsm["COVET_SQRT"] = np.asarray(self.decode_cov(
            self.sc_data.obsm["enclus_latent"]
        ))
        self.sc_data.obsm["COVET"] = np.matmul(
            self.sc_data.obsm["COVET_SQRT"], self.sc_data.obsm["COVET_SQRT"]
        )

    def infer_niche_celltype(self, cell_type_key="cell_type"):
        """
        Predict cell type abundence based one ENCLUS-inferred COVET representations

        :param cell_type_key: (string) key in spatial_data.obs where cell types are stored for environment composition (default 'cell_type')

        :return: nothing, adds 'niche_cell_type' to self.sc_data.obsm & self.spatial_data.obsm
        """

        self.spatial_data.obsm["cell_type_niche"] = niche_cell_type(
            self.spatial_data,
            self.k_nearest,
            spatial_key=self.spatial_key,
            cell_type_key=cell_type_key,
            batch_key=self.batch_key,
        )

        regression_model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5).fit(
            self.spatial_data.obsm["COVET_SQRT"].reshape(
                [self.spatial_data.shape[0], -1]
            ),
            self.spatial_data.obsm["cell_type_niche"],
        )

        sc_cell_type = regression_model.predict(
            self.sc_data.obsm["COVET_SQRT"].reshape([self.sc_data.shape[0], -1])
        )

        self.sc_data.obsm["cell_type_niche"] = pd.DataFrame(
            sc_cell_type,
            index=self.sc_data.obs_names,
            columns=self.spatial_data.obsm["cell_type_niche"].columns,
        )
