# exp222_row_step_delta_target_ablation_on_exp148

## 状態

- ルート: ml_model
- 状態: completed_train_side_rejected_no_submit
- CV: 15.301575123885728
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-08
- 親実験: exp148_learned_likelihood_fulltrain_addonly_on_exp092

## 仮説

exp148 系の anchor residual target は各 row を `last_known_tvt` から独立に戻す。row-to-row step delta を学習し、推論・OOF 評価で well ごとに累積復元すれば、long-tail の形状一貫性が改善する可能性がある。一方で target 変更は過去に long-tail regression を起こしているため、まず lgb0 の CPU train-side ablation だけで反証する。

## 変更点

- exp148 の feature surface、U-projection feature、learned-likelihood feature は固定。
- target を `target_step_delta` に変更。
- 最初の unknown row は `TVT_i - last_known_tvt`、以降は `TVT_i - TVT_{i-1}`。
- OOF prediction は `last_known_tvt + cumsum(pred_step_delta)` で `pred_tvt` に戻して RMSE を計算。
- LightGBM は `lgb0` のみ、CPU deterministic mode、5 folds。
- inference / submit は未対応。train-side guard を通した場合だけ別途実装する。

## 検証方針

- Fold: 5-fold GroupKFold
- Group: `well`
- Stratification: なし
- Leakage Check: 前 row true TVT は label 作成だけに使い、feature には入れない。OOF prediction、valid/test true TVT、oracle best、true-error rank、評価 label は feature source にしない。
- Primary metric: 累積復元後 `pred_tvt` の RMSE
- Guard: distance bucket、worst-well、cumulative drift、exp115 hidden-like stress

## 実行入口

- 学習 notebook: `exp222_row_step_delta_target_ablation_on_exp148_train.ipynb`
- 推論 notebook: `exp222_row_step_delta_target_ablation_on_exp148_inference.ipynb` は scaffold のみ。提出候補化するまで使わない。
- Kaggle 準備: `make prepare-kaggle-notebooks EXP=exp222_row_step_delta_target_ablation_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp222-stepdelta-lgb0 --title 'exp222 stepdelta lgb0' --run-on-push --strict"`
- Kaggle URL: https://www.kaggle.com/code/kentookumura/exp222-stepdelta-lgb0
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 15.301575123885728 |
| Public LB | なし |
| Private LB | なし |

Kaggle train v3 は `kentookumura/exp222-stepdelta-lgb0` version 3 で完了した。fold RMSE は 14.576648 / 15.961225 / 14.807865 / 16.358125 / 14.716812。exp148 lgb0 の 8.599786 から +6.701789 悪化し、exp148 lgb_mean の 8.501281 から +6.800294 悪化した。

Distance bucket は `000_050` だけ 0.601446 と良いが、`1000_plus` が 16.933071 で exp148 lgb_mean bucket 9.325405 から +7.607666 悪化した。worst-well は `1b1eba53` RMSE 67.727455、`896d15b9` 58.371578、`81bf5923` 51.912468。cumulative drift でも final error が -69.507812、-84.615234、+78.115234 など大きく、recursive step-delta の累積誤差が主因。

## 所見

### 良かった点

- feature surface を固定した target-only ablation なので、exp148 との差分が読みやすい。
- `cumulative_drift.csv` を出すため、step target の蓄積誤差を well 単位で確認できる。
- v3 の assembly memory fix で Kaggle CPU 上の full train は完走した。

### 悪かった点

- v1 は入力確認セルで exp145 learned likelihood cache 378万行を全量ロードし、その後本処理でも再ロードしたため、学習開始前に Kaggle kernel が死亡した。
- v2 でも full feature assembly 中に学習前 memory death したため、v3 では full-frame `copy()`、巨大な一括 finite check、anchor merge、target sort の full feature copy を削減した。
- CPU lgb0 でも peak RSS は約 21.97 GB、elapsed は 3397.475 sec と重い。
- step delta の raw target RMSE は 0.036166 と小さく見えるが、well-wise cumsum 復元で TVT RMSE 15.301575 まで悪化した。
- long-tail 1000_plus と worst-well の cumulative drift が壊れた。

### リスク / 注意

- global RMSE が近くても、near row、worst-well、hidden-like stress、cumulative drift が悪化する場合は submit しない。
- この実験は deterministic submission anchor ではない。submission は生成しない。

## 次

1. row-step delta target は採用しない。lgb1/lgb2 展開、inference port、submit は行わない。
2. recursive delta prediction の後続は、直接予測ではなく drift diagnostics / posthoc guard の材料に限定する。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
