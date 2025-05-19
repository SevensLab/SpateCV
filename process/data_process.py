import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
from process.data import reindex

adata_seq = sc.read('../DataUpload/Dataset5/scRNA_count.txt', sep = '\t', first_column_names = True).T
adata_spatial = sc.read('../DataUpload/Dataset5/Spatial_count.txt', sep = '\t')

# adata_spatial.obs_names_make_unique()
# adata_seq.obs_names_make_unique()
locations = sc.read('../DataUpload/Dataset5/Locations.txt', sep = '\t')
spatial_data = locations.X
adata_spatial.obsm['spatial'] = spatial_data

adata_spatial.var_names_make_unique()
adata_seq.var_names_make_unique()

sc.pp.filter_genes(adata_seq, min_cells=1)
sc.pp.filter_cells(adata_seq, min_genes=200)  
sc.pp.filter_genes(adata_spatial, min_cells=1)
sc.pp.filter_cells(adata_spatial, min_genes=10)  
adata_seq = reindex(adata_seq, adata_spatial.var_names)
sc.pp.filter_genes(adata_seq, min_cells=1)
sc.pp.filter_cells(adata_seq, min_genes=1) 

adata_spatial = reindex(adata_spatial, adata_seq.var_names)
sc.pp.filter_genes(adata_spatial, min_cells=1)
sc.pp.filter_cells(adata_spatial, min_genes=1) 

print(adata_spatial,adata_seq)

adata_seq.write('./datasets/sc/dataset5_seq_.h5ad')
adata_spatial.write('./datasets/sp/dataset5_spatial_.h5ad')

