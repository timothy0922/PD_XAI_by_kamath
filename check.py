#%%
import scanpy as sc
import anndata as ad
import lime
import os
import numpy as np
import pandas as pd
# %%
# 데이터로드
DATA_DIR = "/data1/BMAI/june/PD_XAI/Data/"
list = ['Homo_bcd.tsv', 'Homo_features.tsv', 'Homo_matrix.mtx', 'METADATA_PD.tsv']
for f in list:
    file_path = os.path.join(DATA_DIR, f)
    size = os.path.getsize(file_path)
    print(f"File: {f}, Size: {size} bytes")


# %%
metadata = pd.read_csv(os.path.join(DATA_DIR, 'METADATA_PD.tsv'),
                         sep = '\t', header = 0, skiprows = [1], low_memory = False)
print(f"전체 세포 수: {len(metadata)}")
# 전체 세포 수: 1,494,413
# %%
# 메타데이터 컬럼 명
# 'species','species__ontology_label'
# 'disease','disease__ontology_label',
# 'organ', 'organ__ontology_label'
metadata['species__ontology_label'].value_counts()
# Homo sapiens       1,432,825
# Macaca fascicularis 44,779
# Tupaia belangeri    9,916
# Rattus norvegicus  6,893
metadata['disease__ontology_label'].value_counts()
# Normal       821,529
# PD           459,009
# LB Dementia  213,875
metadata['organ__ontology_label'].value_counts()
# substantia nigra pars compacta 1,418,359
# caudate nucleus                76,054
# %%
subset_metadata = metadata[
    (metadata['species__ontology_label'] == 'Homo sapiens')&
    (metadata['disease__ontology_label'] != 'Lewy body dementia')&
    (metadata['organ__ontology_label']== 'substantia nigra pars compacta')
].reset_index(drop = True)

print(f"subset 세포 수 : {len(subset_metadata)}")
# subset 세포 수: 1,142,896
print(subset_metadata['disease__ontology_label'].value_counts())
# Normal: 683,887
# PD: 459,009
print(f"donor 수 : {subset_metadata['donor_id'].nunique()}명")
# Donor: 14명
print(subset_metadata.groupby(
    ['donor_id','disease__ontology_label']
).size().reset_index(name = 'cell_count'))
# %%
import scipy.io

print("matrix 로딩 중...")
matrix = scipy.io.mmread(DATA_DIR + 'Homo_matrix.mtx').T.tocsr()
print(f"matrix shape: {matrix.shape}")

#%%
barcode = pd.read_csv(os.path.join(DATA_DIR, 'Homo_bcd.tsv'), sep = '\t', header = None)
print(f"barcode shape: {barcode.shape}")
feature = pd.read_csv(os.path.join(DATA_DIR,'Homo_features.tsv'), sep = '\t', header = None)
print(f"feature shape: {feature.shape}")
#%%
# 메트릭스, 바코드, 특징 데이터셋 통합
adata = ad.AnnData(X = matrix)
adata.obs_names = barcode[0].values
adata.var_names = feature[1].values # gene name
adata.var['feature_type'] = feature[2].values # gene expression
print(adata)
# %%

#%%
# 각 단계별로 나눠서 확인해볼게요
adata = ad.AnnData(X=matrix)
print(f"Step 1 완료: {adata.shape}")

adata.obs_names = barcode[0].values
print(f"Step 2 완료: obs_names 설정")

adata.var_names = feature[1].values
print(f"Step 3 완료: var_names 설정")

adata.var['feature_type'] = feature[2].values
print(f"Step 4 완료: feature_type 설정")

print(adata)