# 要件

## 依頼

正解 TVT 以外の train-side 情報と候補パス間の差・傾向から regime を作り、
傾向別の複数 expert を学習できるか検証する。最初の実装は、exp264 の global
dual-objective selector と混同しないよう、pairwise divergence fingerprint の
分離可能性を確認する Stage 0 監査に限定する。

## 制約

- Route: `ensemble`。
- 親は `exp264_exp263_candidate_confidence_dual_selector`、候補値の正は
  `exp263_last_anchor_better_candidate_confidence_pair_cache` とする。
- regime/gate 特徴には evaluation-tail の true TVT、target、error、oracle、candidate winner、
  exp264 outer-valid label を使わない。
- 候補パスの絶対 TVT は regime 特徴に使わず、primitive 6 本の相対差、傾き差、曲率差、
  crossing、順位安定性、confidence と非 TVT raw context だけを使う。
- primitive 6 本の 15 pair を正とし、線形結合である 5 pair candidate と fixed formula を
  pairwise fingerprint の独立成分として重複投入しない。
- 初回は 512-row block、K=3、outer-train-only robust scaling + KMeans を固定し、
  window/K/temperature の grid を行わない。
- Stage 0 は 0 variant、0 model config、0 fold training、0 booster。親/control 再学習なし。
- Stage 1 soft expert は Stage 0 guard 通過後の別承認とし、初回実装では無効化する。
- 再現性は `docs/06_reproducibility.md` に従い、stable seed、入力/特徴 schema/content SHA、
  centroid/assignment SHA、Kaggle package metadata を記録する。

## 受け入れ基準

- `docs/legacy/steering/`、実験ディレクトリ、`config.yaml`、Jupytext train/inference notebook、
  `SESSION_NOTES.md`、`result.md`、`metrics.json` が揃う。
- Stage 0 が exp263 の 6 primitive を fold ごとに読み、15 pair の target-free fingerprint を
  512-row block 単位で生成する。
- outer fold ごとに outer-train block だけで scaler/KMeans を fit し、outer-valid へ割り当てる。
- regime occupancy、centroid、fold stability、pairwise feature summary、exp264 global score の
  regime 別校正 readout を保存する。exp264 Stage B artifact が未完成の場合は fail-closed にする。
- Stage 1 昇格条件を機械可読に保存する: 各 regime が 4/5 folds で 100 wells 以上かつ
  block share 10%以上、centroid-matched assignment stability 70%以上、少なくとも2 regimeで
  best candidate family が異なるか global expected-error calibration bias 差が0.25 ft以上。
- Stage 0 の静的テスト、Jupytext test、`py_compile`、`ruff --select F821`、
  `validate-exp` が通る。
- deterministic anchor として扱う場合は feature content SHA、centroid/assignment SHA、
  Kaggle kernel versionを記録する。Stage 0ではmodel/prediction/submission SHAは対象外と明記する。

