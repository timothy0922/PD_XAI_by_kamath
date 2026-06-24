#%%
import scanpy as sc
import pandas as pd
import numpy as np
import os

DATA_DIR = '/data1/BMAI/june/PD_XAI/Data/'

adata = sc.read_h5ad(os.path.join(DATA_DIR, 'kamath_raw.h5ad'))
print(adata)
# 전체 요약 
# n_obs × n_vars = 320106 × 41625

# %%
# 기본 정보
print(f"shape: {adata.shape}")
print(f"세포 수: {adata.n_obs}")
print(f"유전자 수: {adata.n_vars}")

# 세포 정보
print(adata.obs.columns.tolist())
print(adata.obs.head())
print(adata.obs.describe())
print(adata.obs['Status'].value_counts())
# Status
# Ctrl    184762
# PD      135344
print(adata.obs['donor_id'].nunique())

# 유전자 정보
print(adata.var.columns.tolist())
print(adata.var.head())
#%%
print(type(adata.X))
print(adata.X.shape)
print(adata.X[:5,:5].toarray())

# %%
# 핵심 마커 유무
pd_genes = ['SNCA', 'LRRK2', 'TH', 'GPC6', 'AGTR1']
for gene in pd_genes:
    if gene in adata.var_names:
        print(f"{gene} 유전자는 데이터셋에 있음")
    else:
        print(f"{gene} 유전자는 데이터셋에 없음")

# donor by status
print(adata.obs.groupby(
    ['donor_id','Status']
).size().reset_index(name = 'cell_count'))
#%%
# 각 세포 별 발현량을 더하여 1차원 형태로 저장함
total_count = np.array(adata.X.sum(axis = 1)).flatten()
print(total_count)
print(f"세포당 UMI  평균: {total_count.mean():.1f}") # 세포당 UMI  평균: 13400.1
print(f"세포당 UMI  중앙값: {np.median(total_count):.1f}") # 세포당 UMI  중앙값: 7698.0
print(f"세포당 UMI  최소: {total_count.min():.1f}") # 세포당 UMI  최소: 600.0
print(f"세포당 UMI  최대: {total_count.max():.1f}") # 세포당 UMI  최대: 320062.0

# %%
# Kamath QC 기준
# 1. UMI < 650 제외
# 2. mt > 10% 제외
# 3. Doublet 제외

adata.var['mt']   = adata.var_names.str.startswith('MT-')

sc.pp.calculate_qc_metrics(adata, qc_vars = ['mt'], inplace = True, log1p = True)

print(f"전체 세포 수: {adata.n_obs}") # 전체 세포 수: 320106
print(f"UMI 650 미만인 세포: {(adata.obs['total_counts']<650).sum()}개") # UMI 650 미만인 세포: 70개
print(f"mt > 10% 인   세포: {(adata.obs['pct_counts_mt']>10).sum()}개") # mt > 10% 인   세포: 2개
# %%
# 데이터가 많을 때는 점을 제외(jitter = False), 혹은 샘플링 후 시각화 진행
sc.pl.violin(
    adata,
    ['n_genes_by_counts','total_counts','pct_counts_mt'],
    jitter = False,
    multi_panel = True
)
# %%
sc.pl.scatter(adata, 'total_counts','n_genes_by_counts', color = 'pct_counts_mt')
# %%
# 1. UMI < 650 제외
adata = adata[adata.obs['total_counts'] >= 650].copy()
print(f"UMI >= 650 후, 최소 UMI 수: {adata.obs['total_counts'].min():.0f}")
print(f"UMI >= 650 후, 세포 수    : {adata.n_obs}개")
# UMI >= 650 후, 세포 수    : 320036개

# 2. mt > 10% 제외
adata = adata[adata.obs['pct_counts_mt']<=10].copy()
print(f"MT 10% 초과 제외 후, 세포 수: {adata.n_obs}개")
# MT 10% 초과 제외 후, 세포 수: 320034개

# 3. Doublet 제외
# sc.pp.scrublet(adata, batch_key='donor_id')
# print(adata.obs['predicted_doublet'].value_counts())
# adata = [adata.obs['predicted_doublet'] == False].copy()
# print(f"Doublet 제거 후, 세포 수: {adata.n_obs}")
# %%
print(f"최종 QC 적용 후, 세포 수: {adata.n_obs}")
adata.write_h5ad(DATA_DIR+'kamath_qc.h5ad')
print("저장 완료")
# %%
