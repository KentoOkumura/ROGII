# exp173_beam_topk_path_posterior_audit

## 状態

- ルート: pf_beam
- 状態: completed_train_side_negative_no_submit
- CV: 15.972927962
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-03
- 親実験: exp072_exp063_full_replay_feature_cache

## 仮説

GR shift-scan proxy ではなく、Beam search 本体が保持した top-K path と path cost を直接見ることで、top2 / topK posterior が候補生成や confidence diagnostic として使えるかを判断できる可能性がある。

## 変更点

- exp072 の train pseudo-tail row と fixed PF/Beam/likPF candidate cache を比較基準に使う。
- Beam search を再実行し、最終 beam の top-K path と累積 cost を復元する。
- top1/top2 commit、top-K weighted mean、固定温度 posterior mean、cost gap、entropy、path separation、top-K oracle headroom を保存する。
- LightGBM 学習、inference port、submission は行わない。

## 検証方針

- Fold: なし
- Group: well
- Stratification: なし
- Leakage Check: true TVT は scoring と oracle-headroom readout のみに使い、Beam generation / posterior temperature / feature generation には使わない。

## 実行入口

- 学習 notebook: `exp173_beam_topk_path_posterior_audit_train.ipynb`
- 推論 notebook: この実験では使わない
- Kaggle 準備: `make prepare-kaggle-notebooks EXP=exp173_beam_topk_path_posterior_audit EXTRA_ARGS="--notebook train --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| baseline `likpf_mean` RMSE | 11.594897672 |
| best posterior RMSE | 15.972927962 |
| best top-K oracle RMSE | 15.549454381 |
| Public LB | なし |
| Private LB | なし |

## 所見

### 良かった点

- Beam top-K path/cost の保存と posterior readout の導線は Kaggle v2 で完走した。

### 悪かった点

- best posterior は `likpf_mean` から RMSE +4.378030290 悪化し、top-K oracle でも +3.954556709 悪化した。

### リスク / 注意

- posterior mean は物理的に無効な中間 trajectory になり得る。
- train-side pseudo-tail で baseline に届かないため、raw-test parity、worst-well regression、mode separation bucket の追加確認には進めない。
- deterministic submission anchor ではない。

## 次

- inference port / submit はしない。この backlog は完了/不採用として閉じる。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
