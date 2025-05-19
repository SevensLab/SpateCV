
import scipy
import anndata as ad
import scanpy as sc
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse import issparse, csr
from anndata import AnnData
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler

# 定义分块大小常量
CHUNK_SIZE = 20000


def reindex(adata, genes, chunk_size=CHUNK_SIZE):
    """
    使用基因列表重新索引AnnData对象

    参数
    ----------
    adata : AnnData
        原始AnnData对象
    genes : list
        用于索引的基因列表
    chunk_size : int, optional
        将大数据分块处理的大小，默认值为CHUNK_SIZE

    返回
    ------
    AnnData
        重新索引后的AnnData对象
    """
    # 找出genes中在adata.var_names中的基因索引
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))

    if len(idx) == len(genes):
        # 如果所有基因都在adata中，直接按顺序重排
        adata = adata[:, genes]
    else:
        # 如果部分基因不在adata中，创建一个新的稀疏矩阵
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        # 分块处理数据以节省内存
        for i in range(new_X.shape[0] // chunk_size + 1):
            # 将每个块的数据填入新的矩阵中
            new_X[i * chunk_size:(i + 1) * chunk_size, idx] = adata[i * chunk_size:(i + 1) * chunk_size, genes[idx]].X
        # 创建一个新的AnnData对象
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var={'var_names': genes})
    return adata


def plot_hvg_umap(hvg_adata, color=['celltype'], path=None, save_filename=None):
    """
    绘制高变基因（HVG）的UMAP图

    参数
    ----------
    hvg_adata : AnnData
        包含高变基因的AnnData对象
    color : list, optional
        用于着色的元数据字段，默认值为['celltype']
    path : str, optional
        保存图像的路径
    save_filename : str, optional
        保存图像的文件名，不包括扩展名

    返回
    ------
    AnnData
        经过UMAP降维处理后的AnnData对象
    """
    # 设置绘图参数，如分辨率和图像大小
    sc.set_figure_params(dpi=80, figsize=(3, 3))  # type: ignore
    # 复制AnnData对象以避免修改原始数据
    hvg_adata = hvg_adata.copy()

    if save_filename:
        # 如果提供了保存文件名，设置保存路径和文件名
        sc.settings.figdir = path
        save = f'{save_filename}.pdf'
    else:
        save = None

    # 数据标准化，限制最大值为10
    sc.pp.scale(hvg_adata, max_value=10)
    # 计算主成分分析（PCA）
    sc.tl.pca(hvg_adata)
    # 计算邻居图，用于UMAP
    sc.pp.neighbors(hvg_adata, n_pcs=30, n_neighbors=30)
    # 进行UMAP降维
    sc.tl.umap(hvg_adata, min_dist=0.1)
    # 绘制UMAP图，并根据指定的元数据字段进行着色
    sc.pl.umap(hvg_adata, color=color, legend_fontsize=15, ncols=2, show=None, save=save)

    return hvg_adata


def get_data_loader(data_ary: np.ndarray,
                    cell_type: np.ndarray,
                    batch_size: int = 512,
                    is_shuffle: bool = True):
    """
    创建用于训练的DataLoader

    参数
    ----------
    data_ary : np.ndarray
        输入数据数组
    cell_type : np.ndarray
        细胞类型标签数组
    batch_size : int, optional
        每个批次的样本数，默认值为512
    is_shuffle : bool, optional
        是否打乱数据，默认值为True

    返回
    -------
    DataLoader
        PyTorch的DataLoader对象
    """
    # 将numpy数组转换为PyTorch张量，并指定数据类型为float32
    data_tensor = torch.from_numpy(data_ary.astype(np.float32))
    cell_type_tensor = torch.from_numpy(cell_type.astype(np.float32))
    # 创建一个TensorDataset，将数据和标签打包
    dataset = TensorDataset(data_tensor, cell_type_tensor)
    # 创建一个随机数生成器，指定设备为CUDA
    generator = torch.Generator(device='cuda')
    # 创建DataLoader，设置批次大小、是否打乱和生成器
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False, generator=generator
    )


def scale(adata):
    """
    对AnnData对象进行最大绝对值缩放

    参数
    ----------
    adata : AnnData
        输入的AnnData对象

    返回
    -------
    AnnData
        经过缩放处理后的AnnData对象
    """
    # 初始化最大绝对值缩放器
    scaler = MaxAbsScaler()
    # 对数据进行缩放，注意转置以匹配缩放器的输入要求
    normalized_data = scaler.fit_transform(adata.X.T).T
    # 将缩放后的数据赋值回AnnData对象
    adata.X = normalized_data
    return adata


def data_augment(adata: AnnData, fixed: bool, noise_std):
    """
    对AnnData对象进行数据增强，通过添加噪声

    参数
    ----------
    adata : AnnData
        输入的AnnData对象
    fixed : bool
        是否添加固定标准差的噪声
    noise_std : float
        噪声的标准差

    返回
    -------
    AnnData
        合并了增强数据的AnnData对象
    """
    # 复制AnnData对象以进行增强
    augmented_adata = adata.copy()
    gene_expression = adata.X

    if fixed:
        # 如果fixed为True，添加固定标准差的噪声
        augmented_adata.X = augmented_adata.X + np.full(gene_expression.shape, noise_std)
    else:
        # 否则，添加符合正态分布的随机噪声
        augmented_adata.X = augmented_adata.X + np.abs(np.random.normal(0, noise_std, gene_expression.shape))

    # 将原始数据与增强数据合并
    merge_adata = adata.concatenate(augmented_adata, join='outer')

    return merge_adata
