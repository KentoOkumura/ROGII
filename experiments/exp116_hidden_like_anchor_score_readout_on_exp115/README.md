# exp116_hidden_like_anchor_score_readout_on_exp115

## 状態

- ルート: ml_model
- 状態: kaggle_train_v2_complete
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-24
- 親実験: exp115_hidden_like_spatial_holdout_from_ppt

## 仮説

exp115 の hidden-like holdout 上で既存 anchor を再採点し、通常 OOF で良い候補が空間的に hidden-like な subset で崩れないかを見る。

## 変更点

- 新規学習なし。
- exp115 split と既存 prediction / by-well metrics の deterministic merge。
- exp073 / exp098 は row-level prediction、exp092 は by-well metrics fallback で採点。

## 検証方針

- Fold: upstream OOF / train-side prediction をそのまま使用。
- Group: well。
- Stratification: exp115 の `verification_like_spatial` / `verification_like_typewell_purged`。
- Leakage Check: exp115 valid true TVT を特徴量や prior 生成に使わず、既存 prediction の評価だけに使う。

## 実行入口

- 学習 notebook: `exp116_hidden_like_anchor_score_readout_on_exp115_train.ipynb`
- 推論 notebook: `exp116_hidden_like_anchor_score_readout_on_exp115_inference.ipynb`
- Kaggle 準備: `make prepare-kaggle-notebooks EXP=exp116_hidden_like_anchor_score_readout_on_exp115 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp116-hidden-like-anchor-readout-train --title 'exp116 hidden like anchor readout train' --run-on-push --strict"`
- Kaggle kernel: `kentookumura/exp116-hidden-like-anchor-readout-train` v2 COMPLETE
- Kaggle output: `experiments/exp116_hidden_like_anchor_score_readout_on_exp115/kaggle/output/train_v2/`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| best verification_like_spatial | exp073 lgb2 RMSE 10.765221 |
| best verification_like_typewell_purged | exp073 lgb2 RMSE 10.725383 |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- loaded sources 3 / missing 0。
- exp073 / exp098 は row-level bucket と by-well readout まで生成済み。

### 悪かった点

- exp092 は row-level prediction が手元で空だったため by-well fallback。row-level bucket 比較は未完。

### リスク / 注意

- exp115 は exact hidden split ではない。LB 代替にしない。
- by-well fallback の exp092 と row-level の exp073/exp098 は粒度が違うため、bucket 比較では混ぜない。

## 次

- exp092 row-level prediction を取得できた場合だけ同 script で再実行する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
