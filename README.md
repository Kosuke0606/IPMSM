# IPMSM Torque Prediction and AIME Analysis

IPMSM のトポロジ画像からトルクを予測し、その予測根拠を AIME (Approximate Inverse Model Explanations) で可視化・解釈するための研究用リポジトリです。CNN ベースの回帰モデル学習、画像前処理、AIME のグローバル・ローカル重要度解析を、スクリプトと Jupyter Notebook で扱えるようにまとめています。

## 何ができるか

- `aime_xai/` に AIME 実装をパッケージ化
- IPMSM 画像とトルク CSV を対応付けて学習データを構築
- グレースケール画像を使ったトルク回帰モデルの学習
- AIME によるグローバル特徴重要度・ローカル特徴重要度の算出
- Notebook 上での可視化実験と解析結果の出力

## リポジトリ構成

```text
.
├── aime_xai/                  # AIME implementation
├── notebook/                  # Training / analysis notebooks
├── 20250425_data/             # Local dataset (git ignored)
│   ├── Material_d/            # RGB motor images
│   ├── Material_d_Gray/       # Grayscale motor images
│   └── TorqData/              # Torque CSV files
├── outputs/                   # Generated figures and artifacts
├── train_model.py             # Grayscale regression training script
├── main.py                    # Minimal entry point placeholder
└── pyproject.toml             # Python dependencies
```

## 主なファイル

- `train_model.py`
  - `224x224` のグレースケール画像を入力に、トルク平均値を回帰する学習スクリプトです。
  - 画像ファイル `SingleI_<i>_<j>.bmp` と CSV `Data_<i>_<j>.csv` を対応付け、CSV の先頭 30 行にある `T` 列の平均を目的変数にしています。
- `aime_xai/core.py`
  - AIME 本体です。
  - 通常の擬似逆行列ベースに加え、Huber loss + IRLS を使うロバスト近似逆演算子も選べます。
- `notebook/EM_AIME.ipynb`
  - RGB 画像ベースの学習・解析ノートブックです。
- `notebook/EM_AIME_Gray.ipynb`
  - グレースケール画像ベースの学習・解析ノートブックです。
- `notebook/RGB_to_Glay.ipynb`
  - RGB 画像をグレースケールへ一括変換する前処理ノートブックです。
- `notebook/AIME_for_ImageData (2).ipynb`
  - CIFAR-10 を例に AIME の考え方と可視化方法を確認するサンプルです。

## セットアップ

このプロジェクトは `pyproject.toml` ベースです。Python 3.12 を前提にしています。

```bash
uv sync
```

Notebook を起動する場合:

```bash
uv run jupyter lab
```

スクリプトを直接実行する場合:

```bash
uv run python train_model.py
```

## データ配置

`train_model.py` はリポジトリ直下の以下の構成を前提にしています。

```text
20250425_data/
├── Material_d/
├── Material_d_Gray/
└── TorqData/
```

補足:

- `20250425_data/` は `.gitignore` 対象なので、データそのものは Git 管理されません。
- Notebook には相対パスの違いがあるため、実行場所に応じて `IMG_BASE_PATH` や `CSV_BASE_PATH` を確認してください。
- 学習スクリプトは `SingleI_*.bmp` と `Data_*.csv` の対応が崩れると読み込みに失敗します。

## 使い方

### 1. 学習

グレースケール画像でトルク回帰モデルを学習します。

```bash
uv run python train_model.py
```

学習時には以下を行います。

- 乱数シード固定
- GPU 検出と TensorFlow / CuPy 初期化
- 画像と CSV の対応付け
- train / validation / test 分割
- VGG 風 1ch CNN による回帰学習
- 最良モデルの `.keras` 保存

### 2. Notebook 解析

Jupyter Lab で `notebook/` 配下を開き、以下の順で使うと流れを追いやすいです。

1. `RGB_to_Glay.ipynb`
2. `EM_AIME.ipynb` または `EM_AIME_Gray.ipynb`
3. `AIME_for_ImageData (2).ipynb`

### 3. AIME をコードから使う

```python
import numpy as np

from aime_xai import AIME

X = np.random.randn(100, 8)
Y = np.random.randn(100, 1)

explainer = AIME(use_huber=False).create_explainer(X, Y)

global_df = explainer.global_feature_importance_without_viz()
local_df = explainer.local_feature_importance_without_viz(
    x=X[0],
    y=np.array([1.0]),
)
```

利用できる主なメソッド:

- `create_explainer`
- `global_feature_importance`
- `global_feature_importance_without_viz`
- `local_feature_importance`
- `local_feature_importance_without_viz`

## 出力物

- `best_vgg_like_1channel_regressor.keras`
  - 学習済み最良モデル
- `outputs/`
  - 図や PDF などの生成物

## 注意点

- 現在の依存関係には `cupy-cuda12x` が含まれており、実質的に GPU 環境を前提とした構成です。
- `main.py` はまだプレースホルダで、実運用の中心は `train_model.py` と `notebook/` です。
- Notebook は実験履歴を含むため、セル出力やパス設定が環境依存になっている箇所があります。

## 今後の改善候補

- `main.py` から学習・解析を呼べる CLI の整備
- Notebook 依存の処理をスクリプトへ切り出し
- データパス設定の統一
- 学習済みモデル評価手順の明文化
- AIME の可視化結果保存フローの整理
