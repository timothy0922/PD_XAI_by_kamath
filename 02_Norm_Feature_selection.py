#%%
import scanpy as sc

import anndata as ad
import pandas as pd
import numpy as np
import harmonypy as hm

sc.settings.n_jobs = 10

#%%
DATA_DIR ='/data1/BMAI/june/PD_XAI/Data/'
adata = sc.read_h5ad(DATA_DIR+"kamath_qc.h5ad")
print(adata)
# %%
# 데이터 균등 샘플링(donor_id 기준)
sc.pp.subsample(adata, n_obs = 2000 * adata.obs['donor_id'].nunique())
print(f"샘플링 후: {adata.n_obs}")
print(adata.obs['Status'].value_counts())
# 샘플링 후: 28000개 
# Status
# Ctrl    16123
# PD      11877
#%%
# 정규화
# 정규화 전, raw data(Count matrix) 저장
# 정규화 후, raw data는 사라지게 됨
adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum = 1e4)
sc.pp.log1p(adata)
print("정규화 완료")

#%%
# Feature selection
# 논문에서 사용한 Feature selection: (1). HVG, (2). PCA, (3). NMF, (4). ETM
# HVG
sc.pp.highly_variable_genes(adata, flavor = 'seurat', n_top_genes = 2000, batch_key = 'donor_id')
print(f"HVG 수: {adata.var['highly_variable'].sum()}")

# PCA
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata, n_pcs = 50, log = True)
#%%
# Identification of Elbow point
variance_ratio = adata.uns['pca']['variance_ratio']
for i, var in enumerate(variance_ratio[:50]):
    consum = np.sum(variance_ratio[:i+1])
    print(f"PC{i+1:2d}:{var:.4f}(누적: {consum:.4f})")

# PC 50까지 누적해도 PC 100% 설명 불가능
# PC 11 이후로 완만하게 기울기가 내려가는 것을 볼 수 있음
# n_PCs : 15 ~ 20 사이로 설정
# %%
# PCA 결과 추출
pca_embedding = adata.obsm['X_pca'].astype(np.float64)
# Batch correction
harmony_out = hm.run_harmony(pca_embedding, adata.obs, 'donor_id')
print("Batch Correction")
#%%
adata.obsm['X_pca_harmony'] = harmony_out.Z_corr
print(f"X_pca_harmony shape: {adata.obsm['X_pca_harmony'].shape}")

# %%
sc.pp.neighbors(adata, use_rep = 'X_pca_harmony')
sc.tl.umap(adata)
# %%
sc.tl.leiden(adata, resolution = 0.3, flavor = 'igraph', n_iterations = 2)
print(f"클러스터 수: {adata.obs['leiden'].nunique()}")
sc.pl.umap(adata, color = ['Status','donor_id','leiden'], ncols = 3)
# %%
sc.tl.rank_genes_groups(adata, groupby = 'leiden', method = 'wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes = 5, sharely = False)
# %%
marker_genes = {
    'DA neuron': ['TH', 'SLC6A3', 'SLC18A2'],
    'Non-DA neuron': ['SNAP25', 'SYT1'],
    'Astrocyte': ['AQP4', 'GFAP'],
    'Oligodendrocyte': ['MBP', 'MOG'],
    'OPC': ['PDGFRA', 'VCAN'],
    'Microglia': ['CSF1R', 'CX3CR1'],
    'Endothelial': ['FLT1', 'CLDN5']
}

sc.pl.dotplot(adata, marker_genes, groupby = 'leiden', dendrogram = True)
# %%
cell_type_map = {
    '0': 'Microglia',
    '1': 'Non-DA neuron',
    '2': 'Non-DA neuron',
    '3': 'Oligodendrocyte',
    '4': 'Oligodendrocyte',
    '5': 'Non-DA neuron',
    '6': 'Endothelial',
    '7': 'Astrocyte',
    '8': 'Non-DA neuron',
    '9': 'Non-DA neuron',
    '10': 'DA neuron',
    '11': 'OPC',
    '12': 'Non-DA neuron',
    '13': 'Non-DA neuron',
    '14': 'Non-DA neuron',
    '15': 'DA neuron',
    '16': 'Non-DA neuron',
    '17': 'Non-DA neuron',
    '18': 'Low quality',
}

adata.obs['cell_type'] = adata.obs['leiden'].map(cell_type_map)
# %%
sc.pl.umap(adata, color='cell_type')
# %%
adata = adata[adata.obs['cell_type'] != 'Low quality'].copy()
print(f"Low quality 제거 후, 세포 수 : {adata.n_obs}")
# Oligodendrocyte    12506
# Non-DA neuron       7118
# Microglia           2397
# Astrocyte           2387
# DA neuron           1637
# OPC                  876
# Endothelial          875
print(adata.obs['cell_type'].value_counts())
print(f"전체 세포 수: {adata.n_obs}")
# 전체 세포 수: 27796
# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

colors = {
    'DA neuron':        '#FF7F0E',
    'Non-DA neuron':    '#8B4513',
    'OPC':              '#FF69B4',
    'Oligodendrocyte':  '#808080',
    'Microglia':        '#9370DB',
    'Endothelial':      '#2CA02C',
    'Astrocyte':        '#1F77B4',
}

order = ['Astrocyte', 'Endothelial', 'Microglia',
         'Oligodendrocyte', 'OPC', 'Non-DA neuron', 'DA neuron']

ct_status = adata.obs.groupby(['cell_type', 'Status']).size().unstack()
ct_prop = ct_status.div(ct_status.sum(axis=1), axis=0).loc[order]
ct_total = ct_status.sum(axis=1).loc[order]
ct_total_prop = ct_total / ct_total.sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(wspace=0.4)

# 왼쪽: n수 제거
bar_colors = [colors[ct] for ct in order]
ct_total_prop.plot(kind='barh', ax=axes[0], color=bar_colors)
axes[0].set_xlabel('Proportion of cells')
axes[0].set_title('Cell type composition')
axes[0].set_ylabel('')

# 오른쪽: stacked bar
left = pd.Series([0.0] * len(order), index=order)
for status, alpha in [('PD', 0.35), ('Ctrl', 1.0)]:
    vals = ct_prop[status]
    for i, ct in enumerate(order):
        axes[1].barh(i, vals[ct], left=left[ct],
                    color=colors[ct], alpha=alpha)
        left[ct] += vals[ct]

axes[1].set_xlabel('Proportion of cells')
axes[1].set_title('Ctrl vs PD proportion')
axes[1].set_ylabel('')
axes[1].set_yticks(range(len(order)))
axes[1].set_yticklabels(order)
axes[1].axvline(x=0.5, color='white', linestyle='--', linewidth=0.8)

# legend를 실제 색상과 일치하게
legend_elements = [
    Patch(facecolor='dimgray', alpha=0.35, label='PD'),
    Patch(facecolor='dimgray', alpha=1.0,  label='Control')
]
axes[1].legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('cell_type_composition.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
adata.write_h5ad('kamath_annotated.h5ad')
print("저장완료")
#%%
print(f"저장 완료: {adata.n_obs} cells x {adata.n_vars} genes")
print(adata.obs['cell_type'].value_counts())
print(adata.X.max())
print(adata.layers)
#%%
