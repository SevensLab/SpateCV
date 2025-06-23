# SpateCV
SpateCV is a unified framework for the joint analysis of scRNA and scST data, enabling the imputation of unmeasured genes and data integration.

![SpateCV](https://github.com/SevensLab/SpateCV/blob/master/flowchart.png)

## Installation
This implementation is written in Python3 and relies on jax, flax, sklearn, scipy and scanpy. It is recommended to use a Python version = `3.9`, and a R version above `4.2`.

``` shell
$ git clone https://github.com/SevensLab/SpateCV.git
$ cd SpateCV
$ conda env create -f environment.yml
$ conda activate SpateCV
```

To install JAX, simply run the command:
```shell
$ pip install -U "jax[cuda12]"
```

## Setup
```shell
pip install -r requirement.txt
```

## Arguments
### baseline
The specific parameter settings follow those in the [Spatial Benchamark](https://github.com/QuKunLab/SpatialBenchmarking), and the baseline code is adapted from Spatial Benchmark, whose parameter settings also follow the default repositories of their respective models.  
The code for uniport is referenced in the example on its official website [Impute genes for MERFISH](https://uniport.readthedocs.io/en/latest/examples/MERFISH/MERFISH_impute.html).  
The code for stDiff is referenced in the example on its official website [demo](https://github.com/fdu-wangfeilab/stDiff/blob/master/demo.ipynb).  
### data preprocess
The datasets 1-12 in the benchmark test were all from the paper [Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type](https://www.nature.com/articles/s41592-022-01480-9). The raw data were initially processed in 'process/data_process.py' to convert the txt file to h5ad, and only the genes shared by both ST and scRNA-seq were retained. 
## Tutorials
We provide source codes for reproducing the SpateCV analysis in the main text in the `demos` directory.

+ [Mouse hypothalamic (MERFISH)](https://github.com/SevensLab/SpateCV/blob/master/demos/Mouse-hypothalamic-MERFISH.ipynb)
+ [Mouse hypothalamic integration (MERFISH)](https://github.com/SevensLab/SpateCV/blob/master/demos/integrate.ipynb)
+ [Ablation study](https://github.com/SevensLab/SpateCV/blob/master/demos/integrate-ablation.ipynb)
+ [Mouse MOp (MERFISH)](https://github.com/SevensLab/SpateCV/blob/master/demos/Mouse-MOp-MERFISH.ipynb)
+ [Mouse VISp (STARmap)](https://github.com/SevensLab/SpateCV/blob/master/demos/Mouse-VISp-STARmap.ipynb)

### Main function: enclus.ENCLUS()
**Key parameters includes:**
- **spatial_data**: AnnData matrix containing spatial transcriptomics data.
- **sc_data**: AnnData matrix containing single-cell RNA-seq data.
- **num_layers**: Number of layers in the neural network. Default: `3`
- **num_neurons**: Number of neurons per hidden layer. Default: `1024`
- **latent_dim**: Dimension of the latent embedding space. Default: `512`
- **k_nearest**: Number of nearest neighbors for constructing kNN graph. Default: `16`
- **num_HVG**: Number of highly variable genes selected for analysis. Default: `2048`
- **spatial_coeff**: Weight coefficient for spatial data loss term. Default: `1.0`
- **sc_coeff**: Weight coefficient for single-cell data loss term. Default: `1.0`
- **cov_coeff**: Weight coefficient for covariate loss term. Default: `1.0`
- **kl_coeff**: Weight coefficient for KL divergence loss. Default: `0.1`
- **patience**: Number of iterations with no improvement before stopping early. Default: `30`
- **n_clusters**: Number of clusters to detect. Default: `6`
- **tau**: Scaling parameter for Deep K-Means clustering. Default: `0.1`
- **num_heads**: Number of attention heads in multi-head attention modules. Default: `8`
- **head_dim**: Dimension of each attention head. Default: `64`
 

## Quick start 
```python
import enclus

enclus_model = enclus.ENCLUS(spatial_data = st_data, sc_data = sc_data,
                    num_layers=3,
                    num_neurons=1024,
                    latent_dim=512,
                    k_nearest=16,
                    num_cov_genes=64,
                    num_HVG=1024,
                    sc_genes=add_genes,
                    spatial_coeff=1,  
                    sc_coeff=1,  
                    kl_coeff=0.02,
                    n_clusters=10,
                    tau=0.1,
                    early_stopping=True,
                    patience=30,
                    num_heads=8,
                    head_dim=64,
                    )
#train model
enclus_model.train(training_steps=4628,
    batch_size=1024,
    verbose=100,
    init_lr=0.00001,
    decay_steps=4000)
    
enclus_model.impute_genes()
st_data.obsm['enclus_latent'] = enclus_model.spatial_data.obsm['enclus_latent']
sc_data.obsm['enclus_latent'] = enclus_model.sc_data.obsm['enclus_latent']
```
