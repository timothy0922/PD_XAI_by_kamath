#%%
import pandas as pd
import numpy as np

import scanpy as sc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score, confusion_matrix, accuracy_score
import lime
import lime.lime_tabular
from scipy.sparse import issparse
from tqdm import tqdm
#%%
adata = sc.read_h5ad('/data1/BMAI/june/PD_XAI/Data/kamath_annotated.h5ad')
print(f"로드 완료: {adata.n_obs} cells * {adata.n_vars} genes")
# 27796 cells * 41625 genes
print(adata.obs['cell_type'].value_counts())
# cell_type
# Oligodendrocyte    12506
# Non-DA neuron       7118
# Microglia           2397
# Astrocyte           2387
# DA neuron           1637
# OPC                  876
# Endothelial          875
#%%
donor_status = adata.obs[['donor_id', 'Status']].drop_duplicates()
print(donor_status.sort_values('Status'))
# %%
# dataset을 donor_id를 기준으로 train/test를 분류함
# cell type이 아니라 donor_id를 기준으로 하는 이유는 donor 간 독립성을 보장하고, 밀접한 cell의 상호작용으로 인한 배치가 존재할 수 있음
control_donor  = ['3345','3346','3322','3298','6173','3482','4956','5610']
target_donor   = ['4560','4568','3887','1963','3873','2142']
# 논문에 따르면 train: target = 0.8:0.2
control_train, control_test = train_test_split(control_donor, test_size = 0.2, random_state=42)
pd_train, pd_test           = train_test_split(target_donor, test_size = 0.2, random_state=42)

train_donor = control_train + pd_train
test_donor = control_test + pd_test
print(f"train donor {len(train_donor)}명: {train_donor}")
# train: 10명
print(f"test donor {len(test_donor)}명: {test_donor}")
# test: 4명
# %%
cell_types = adata.obs['cell_type'].unique().tolist()
print(cell_types)
#%%
adata_dict = {}
for cell in cell_types:
    cell_adata = adata[adata.obs['cell_type']==cell]
    # donor 기준 split
    train_ad = cell_adata[cell_adata.obs['donor_id'].isin(train_donor)].copy()
    test_ad = cell_adata[cell_adata.obs['donor_id'].isin(test_donor)].copy()
    # Train에서만 HVG 진행
    # 논문 코드: sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # 이 코드로 돌리면, 발현량, 분산 기준으로 nHVG를 결정함
    sc.pp.highly_variable_genes(train_ad, min_mean=0.0125, max_mean=3, min_disp=0.5)
 #   sc.pp.highly_variable_genes(train_ad, n_top_genes = 2000, batch_key = 'donor_id')
    hvg_genes = train_ad.var_names[train_ad.var['highly_variable']]
    # Test에서는 Train HVG 활용
    train_ad = train_ad[:,hvg_genes].copy()
    test_ad = test_ad[:,hvg_genes].copy()

    adata_dict[cell] = {'train': train_ad, 'test': test_ad}
    print(f"{cell} → train: {train_ad.n_obs}cells / test: {test_ad.n_obs}cells / HVG: {train_ad.n_vars}genes")

# Microglia → train: 1648cells / test: 749cells / HVG: 5323genes
# Non-DA neuron → train: 4423cells / test: 2695cells / HVG: 7313genes
# Oligodendrocyte → train: 8488cells / test: 4018cells / HVG: 5653genes
# Endothelial → train: 433cells / test: 442cells / HVG: 5508genes
# Astrocyte → train: 1657cells / test: 730cells / HVG: 6558genes
# DA neuron → train: 1147cells / test: 490cells / HVG: 5663genes
# OPC → train: 603cells / test: 273cells / HVG: 5964genes
# %%
save_dir = '/data1/BMAI/june/PD_XAI/Data/ML_ready_author/'
cell_type = ['Astrocyte','DA_neuron','Endothelial','Microglia','Non-DA_neuron','Oligodendrocyte','OPC']
cell_adata = {}
for cell in cell_type:
    cell_adata[cell] = {
        'train' : sc.read_h5ad(f"{save_dir}{cell}_train.h5ad"),
        'test'  : sc.read_h5ad(f"{save_dir}{cell}_test.h5ad")
    }
    print(f"{cell} -> train: {cell_adata[cell]['train'].obs}, test: {cell_adata[cell]['test'].obs}")    
#%%
# MLP: (1). hidden node: 100, (2). ReLu activation, (3). Adam optimizer, (4) 500-iteration cap
kf = KFold(n_splits = 5, shuffle = True)
mlp_dict = {}

for cell, data in tqdm(cell_adata.items(), desc = 'Training_MLP'):
    print(f"\n -- {cell} -- ")
    train_ad = data['train']
    test_ad  = data['test']
    # scRNA-seq데이터는 주로 희소행렬 데이터
    # MLP에 넣기 위해서는 행렬로 변환 필요
    # 정답 레이블 만들기
    X_train = train_ad.X.toarray() if issparse(train_ad.X) else train_ad.X
    y_train = (train_ad.obs['Status']== 'PD').astype(int).values
    
    X_test  = test_ad.X.toarray()  if issparse(test_ad.X)  else test_ad.X
    y_test =  (test_ad.obs['Status']== 'PD').astype(int).values

    print(f"Train - control: {(y_train == 0).sum()}, PD: {(y_train == 1).sum()}")
    print(f"Test - control: {(y_test == 0).sum()}, PD: {(y_test == 1).sum()}")
    fold_score = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, Y_val = y_train[train_idx], y_train[val_idx]
        
        mlp_fold = MLPClassifier(
        hidden_layer_sizes = (100,),
        activation = 'relu',
        solver = 'adam',
        max_iter = 500,
        random_state = 42
    )
        mlp_fold.fit(X_tr, y_tr)
        score = balanced_accuracy_score(Y_val, mlp_fold.predict(X_val))
        fold_score.append(score)
        print(f" Fold {fold+1}: BA = {score:.3f}")
    print(f"5-Fold CV Balanced Accuracy: {np.mean(fold_score):.3f} ± {np.std(fold_score)}")

    mlp_val = MLPClassifier(
        hidden_layer_sizes = (100,),
        activation = 'relu',
        solver = 'adam',
        max_iter = 500, random_state = 42
    )

    mlp_val.fit(X_train, y_train)
    y_pred = mlp_val.predict(X_test)
    test_ba_val = balanced_accuracy_score(y_test, y_pred)
    print(f"Test Balanced Accuracy: {test_ba_val:.3f}")

    print(classification_report(y_test, y_pred, target_names=['Control', 'PD']))
    df_index = pd.DataFrame({
        'true_label': y_test,
        'predicted_labe': y_pred
    })
    df_index['cell_index'] = df_index.index

    mlp_dict[cell] = {
        'model': mlp_val,
        'cv_score': fold_score,
        'cell_index': df_index
    }
    
# %%
import os
save_dir = '/data1/BMAI/june/PD_XAI/Data/ML_ready_author/'
os.makedirs(save_dir, exist_ok=True)

for ct, data in adata_dict.items():
    # cell type 이름에서 공백 제거 (파일명용)
    ct_name = ct.replace(' ', '_')
    
    data['train'].write_h5ad(f"{save_dir}{ct_name}_train.h5ad")
    data['test'].write_h5ad(f"{save_dir}{ct_name}_test.h5ad")
    print(f"{ct} 저장 완료 → train: {data['train'].n_obs}, test: {data['test'].n_obs}")

print("\n모든 파일 저장 완료!")
# %%
import matplotlib.pyplot as plt
import numpy as np

cell_types = ['DA neuron', 'Non-DA neuron', 'OPC', 
              'Oligodendrocyte', 'Microglia', 'Endothelial', 'Astrocyte']

cv_ba  = [0.512, 0.983, 0.932, 0.982, 0.943, 0.752, 0.941]
test_ba = [0.500, 0.469, 0.941, 0.806, 0.878, 0.827, 0.963]

x = np.arange(len(cell_types))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width/2, cv_ba,  width, label='5-Fold CV BA', color='steelblue')
ax.bar(x + width/2, test_ba, width, label='Test BA',     color='tomato')

ax.set_ylabel('Balanced Accuracy')
ax.set_title('MLP Classification Performance by Cell Type')
ax.set_xticks(x)
ax.set_xticklabels(cell_types, rotation=15)
ax.set_ylim(0.3, 1.05)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label='Random (0.5)')
ax.legend()

plt.tight_layout()
plt.savefig('mlp_performance.png', dpi=150, bbox_inches='tight')
plt.show()
# %%

lime_result = {}
for cell_type, data in tqdm(cell_adata.items(), desc = 'LIME'):
    train_ad = data['train']
    test_ad  = data['test']

    X_train = train_ad.X.toarray() if issparse(train_ad.X) else train_ad.X
    X_test  = test_ad.X.toarray() if issparse(test_ad.X) else test_ad.X

    gene_name = train_ad.var_names.tolist()
    num_genes = len(gene_name)
    mlp       = mlp_dict[cell_type]['model']

    df_test = pd.DataFrame(X_test)
    df_test.index = range(len(X_test))
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        class_names   = ['Control', 'PD'],
        feature_names = gene_name,
        mode          = 'classification',
        discretize_continuous = False
    )

    df_lime = pd.DataFrame()
    df_lime[0] = []
    df_lime[1] = []
    df_lime['cell_index'] = []

    for i in tqdm(range(len(X_test)), desc=f'{cell_type} cells', leave=False):
        exp = explainer.explain_instance(
            df_test.values[i],
            mlp.predict_proba,
            num_features = num_genes
        )
        dd = exp.as_list()
        df2 = pd.DataFrame(dd)
        df2['cell_index'] = i
        df_lime = pd.concat([df_lime, df2])

    save_path = f'/data1/BMAI/june/PD_XAI/Data/LIME/{cell_type}_lime.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_lime.to_csv(save_path, index=False)
    lime_result[cell_type] = df_lime
    print(f"저장 완료: {save_path}")
# %%
ct = 'DA_neuron'
data = cell_adata[ct]
train_ad = data['train']
test_ad  = data['test']

X_train = train_ad.X.toarray() if issparse(train_ad.X) else train_ad.X
X_test  = test_ad.X.toarray()  if issparse(test_ad.X)  else test_ad.X

gene_names = train_ad.var_names.tolist()
num_genes  = len(gene_names)
mlp = mlp_dict[ct]['model']

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    class_names   = ['Control', 'PD'],
    feature_names = gene_names,
    mode          = 'classification',
    discretize_continuous = False
)

df_lime = pd.DataFrame()
df_lime[0] = []
df_lime[1] = []
df_lime['cell_index'] = []

for i in tqdm(range(len(X_test)), desc='DA neuron LIME'):
    exp = explainer.explain_instance(
        X_test[i],
        mlp.predict_proba,
        num_features = num_genes,
        num_samples  = 1000
    )
    dd = exp.as_list()
    df2 = pd.DataFrame(dd)
    df2['cell_index'] = i
    df_lime = pd.concat([df_lime, df2])
# %%
save_path = '/data1/BMAI/june/PD_XAI/Data/LIME/DA_neuron_lime.csv'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df_lime.to_csv(save_path, index=False)
print("저장 완료!")
# %%
ct = 'Astrocyte'
data = cell_adata[ct]
train_ad = data['train']
test_ad  = data['test']

X_train = train_ad.X.toarray() if issparse(train_ad.X) else train_ad.X
X_test  = test_ad.X.toarray()  if issparse(test_ad.X)  else test_ad.X

gene_names = train_ad.var_names.tolist()
num_genes  = len(gene_names)
mlp = mlp_dict[ct]['model']

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    class_names   = ['Control', 'PD'],
    feature_names = gene_names,
    mode          = 'classification',
    discretize_continuous = False
)

df_lime = pd.DataFrame()
df_lime[0] = []
df_lime[1] = []
df_lime['cell_index'] = []

for i in tqdm(range(len(X_test)), desc='DA neuron LIME'):
    exp = explainer.explain_instance(
        X_test[i],
        mlp.predict_proba,
        num_features = num_genes,
        num_samples  = 1000
    )
    dd = exp.as_list()
    df2 = pd.DataFrame(dd)
    df2['cell_index'] = i
    df_lime = pd.concat([df_lime, df2])
# %%
save_path = '/data1/BMAI/june/PD_XAI/Data/LIME/Astrocyte_neuron_lime.csv'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df_lime.to_csv(save_path, index=False)
print("저장 완료!")
# %%
lime_summary = df_lime.groupby(0)[1].mean().sort_values(ascending =False)
print(lime_summary.head(20))
# SLC7A11        0.003481
# CHI3L1         0.003396
# PPP1R3C        0.002708
# NMNAT2         0.002594
# CRYAB          0.002565 -> 논문과 동일
# HSPH1          0.002169
# ADAMTS9-AS2    0.002121
# C5orf17        0.002076
# HSPB1          0.001993
# DAAM1          0.001948
# RNA28S5        0.001902
# HSP90AA1       0.001871 -> 논문과 동일
# SMAD9          0.001849
# HBG2           0.001848
# MT2A           0.001834 -> 논문과 동일
# MAP2           0.001830
# ANGPT1         0.001828
# NUPR1          0.001796
# HSPA1A         0.001779 -> 논문과 동일
# MT1G           0.001764
# %%
from scipy import stats
lime_zscore = pd.Series(
    stats.zscore(lime_summary.values),
    index = lime_summary.index
)
fig, ax = plt.subplots(figsize=(12, 8))
lime_abs = lime_summary.abs()
zscore_abs = lime_zscore.abs()
top_genes = lime_zscore.abs().sort_values(ascending = False).head(20).index

# 전체 유전자 회색으로
ax.scatter(lime_abs.values, zscore_abs.values,
           color='lightgray', s=10, alpha=0.5, label='Other genes')

# 상위 유전자 강조
ax.scatter(lime_abs[top_genes].values, zscore_abs[top_genes].values,
           color='tomato', s=40, zorder=5, label='Top genes')

for gene in top_genes:
    ax.annotate(gene,
                xy=(lime_abs[gene], zscore_abs[gene]),
                fontsize=11, fontweight='bold', ha='left',
                xytext=(5, 0), textcoords='offset points')
ax.set_xlabel('Mean LIME score (absolute value)', fontsize=13)
ax.set_ylabel('LIME feature importance Z-score (absolute value)', fontsize=13)
ax.set_title('Astrocyte — LIME feature importance across all cells', fontsize=14)
ax.tick_params(labelsize=11)
ax.legend(fontsize=11)
ax.set_xlabel('Mean LIME score (absolute value)')
ax.set_ylabel('LIME feature importance Z-score (absolute value)')
ax.set_title('Astrocyte — LIME feature importance across all cells')
ax.legend()
plt.tight_layout()
plt.savefig('/data1/BMAI/june/PD_XAI/Figure/astrocyte_lime_all.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(12, 8))

# 강조할 유전자
highlight_genes = ['HSP90AA1', 'HSPA1A', 'MT2A', 'CRYAB', 'RIMS1']

# 전체 유전자 회색
ax.scatter(lime_abs.values, zscore_abs.values,
           color='lightgray', s=10, alpha=0.5, label='Other genes')

# 상위 유전자 연한 빨강
ax.scatter(lime_abs[top_genes].values, zscore_abs[top_genes].values,
           color='tomato', s=40, alpha=0.5, zorder=4, label='Top genes')

# 강조 유전자 별도 색상
for gene in highlight_genes:
    if gene in lime_abs.index:
        ax.scatter(lime_abs[gene], zscore_abs[gene],
                   color='gold', s=150, zorder=6,
                   edgecolors='black', linewidth=1.5)
        ax.annotate(gene,
                    xy=(lime_abs[gene], zscore_abs[gene]),
                    fontsize=12, fontweight='bold',
                    ha='left', color='black',
                    xytext=(8, 3), textcoords='offset points')

# 범례 수동 추가
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
           markersize=8, label='Other genes'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', 
           markersize=8, label='Top genes'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
           markeredgecolor='black', markersize=10, 
           label='PD-associated genes (Fiorini et al.)'),
]
ax.legend(handles=legend_elements, fontsize=10)

ax.set_xlabel('Mean LIME score (absolute value)', fontsize=13)
ax.set_ylabel('LIME feature importance Z-score (absolute value)', fontsize=13)
ax.set_title('Astrocyte — LIME feature importance\n(highlighted: PD-associated genes)', 
             fontsize=14)

plt.tight_layout()
plt.savefig('astrocyte_lime_highlight.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
