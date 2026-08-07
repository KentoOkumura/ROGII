# exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction

## 状態

- ルート: pf_beam
- 状態: submitted / not adopted
- CV: 9.427109596582213
- Public LB: 9.837
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-09
- 親実験: `exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline` の失敗切り分け

## 仮説

Connor Tynan の公開 notebook `ROGII K16 spline + kernel kNN + adaptive kappa` の deterministic v6 fallback は、exp206 の単純な `dTVT ~= a*dZ+b` 線形外挿よりも hidden tail に強い可能性がある。外部 weight 前提の v7/v8 は混ぜず、まず K16 spline / kernel kNN / adaptive kappa / ANCC near-strike / GR correction / U-projection を再現して単独性能を測る。

## 変更点

- 公開ソース `/tmp/kaggle-notebooks/connortynan-k16-versioned/rogii-k16-spline-kernel-knn-adaptive-kappa.py` の v6 fallback を `connortynan_k16_reproduction.py` に移植。
- train 側は target well を donor field と kappa fit から除外する group-safe KFold CV に変更。
- v7 neural committee と v8 LightGBM meta-layer は外部 weights 不在のため無効化。
- inference は full train field/kappa を fit して `submission.csv` を直接生成する。

## 検証方針

- Fold: 5-fold group-safe、well id SHA256 ordering
- Group: `well_id`
- Stratification: なし
- Leakage Check:
  - validation target well は donor field から除外。
  - validation target well は kappa fit から除外。
  - unknown suffix の true TVT は metrics のみに使用。
  - test inference は test true TVT を使用しない。

## 実行入口

- 学習 notebook: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train.ipynb`
- 推論 notebook: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 9.427109596582213 |
| Public LB | 9.837 |
| Private LB | - |

## 所見

### 良かった点

- Stage 1 は GPU / booster / 外部 weights なしで Kaggle train/inference まで実行できた。
- exp206 の線形 `a,b` 外挿とは別系統の geometry + local donor field 再現になっている。
- group-safe CV は exp206 v4 CV 52.507 より大幅に良い。

### 悪かった点

- group-safe kappa refit は公開 notebook の full-train fit より重いが、Kaggle train v1 は約 6 分で完了した。
- 公開 header の v8 スコア主張は外部 weights 前提の可能性があり、この実験の Stage 1 では評価対象外。

### リスク / 注意

- 公開 notebook の v6 fallback と同じく typewell GR correction を使うため、GR/typewell の欠損や外れ値に影響される。
- CV は target well を除外するため、public script の full-train kappa より厳しい条件になる。
- LB 判断前に blind weight search や v7/v8 代替モデルを追加しない。

## 次

- Stage 1 は Public LB 9.837 で不採用とし、v7/v8 external weights audit は必要なら別途判断する。
- Stage 2 は外部 `e2e2/stacker/gbm` weights の有無と必要入力を監査してから判断する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
