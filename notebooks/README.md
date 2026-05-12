# Notebooks

論文 Y の各種分析を Google Colab で実行するためのノートブックを格納します。

## ノートブック一覧

| ファイル | 内容 |
|---|---|
| `01_exploratory.ipynb` | 探索的データ分析(EDA):記述統計、領域別分布、相関ヒートマップ |
| `02_smm_visualization.ipynb` | SMM 構造推定の可視化:領域別パラメータプロット、適合度比較 |
| `03_robustness.ipynb` | 頑健性検定:外れ値除外、サブサンプル分析、代替仕様 |

## Google Colab での実行方法

各ノートブック冒頭に以下のセルを配置し、GitHub リポジトリをマウントしてから実行する:

```python
# Google Colab セットアップ(セル冒頭)
!git clone https://github.com/USERNAME/paper-y-replication.git
%cd paper-y-replication
!pip install -r requirements.txt

# データセットをアップロード(初回のみ)
from google.colab import files
uploaded = files.upload()  # hanrei_features_v3_3_2015_2024.csv を選択
!mv hanrei_features_v3_3_2015_2024.csv data/
```

## ローカル環境での実行方法

```bash
jupyter notebook notebooks/
```

## 注意事項

- ノートブック自体は本リポジトリにはプレースホルダーのみを含み、実コードは `src/` ディレクトリの各 Python スクリプトから移植して作成する想定です
- ノートブックは可視化・対話的分析のために設計され、本格的な再現は `src/run_all.py` の使用を推奨します
