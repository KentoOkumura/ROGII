# exp133_gr_bimodal_match_ambiguity_detector

## 状態

- ルート: pf_beam
- 状態: completed_train_side_diagnostic_no_submit
- CV: 9.322479895503927 (`pred_exp092_lgb1` reference best)
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-26
- 親実験: `gr_bimodal_match_ambiguity_detector` backlog

## 仮説

GR score curve が flat または +/-15-25ft 周期 decoy で二峰化する row / well では、PF/Beam/likPF や exp092 の hard な mode commit が大外ししやすい。真値 mode を当てに行くのではなく、ambiguity / uncertainty / averaging weight として扱うことで、後続の exp092 confidence feature や near-row guard の材料になる。

## 変更点

- exp072 feature cache、exp073 OOF、exp092 OOF を固定入力として join する。
- 各候補 TVT 周辺の `[-25,-20,-15,0,15,20,25] ft` shifted GR score curve を作る。
- top1-top2 margin、peak count、peak spacing、shifted score gap、entropy、bimodality、flat flag を出す。
- mode commit / midpoint / likPF-midpoint blend は診断 proxy としてだけ評価する。
- LightGBM 学習、PF/Beam 再実行、inference port、submit は行わない。

## 検証方針

- Fold: upstream OOF predictions を使用
- Group: well
- Stratification: ambiguity flag、flat flag、distance bucket、margin / entropy / ambiguity quantile
- Leakage Check: true TVT は metrics と bucket readout のみに使い、score curve や threshold 作成には使わない

## 実行入口

- 学習 notebook: `exp133_gr_bimodal_match_ambiguity_detector_train.ipynb`
- 推論 notebook: `exp133_gr_bimodal_match_ambiguity_detector_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp133_gr_bimodal_match_ambiguity_detector EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp133-gr-bimodal-ambiguity-train --title 'exp133 gr bimodal ambiguity train' --run-on-push --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は smoke debug のみに限定する。

## 結果

- Kaggle train v2 完了。
- rows / wells: 3,783,989 / 773
- runtime: 1783.534 sec
- best candidate: `pred_exp092_lgb1` RMSE 9.322479895503927 / MAE 5.980980396 / within10 0.822047051
- ambiguity: ambiguous rate 0.566856563、flat rate 0.432845324、mean ambiguity score 0.488638610
- feature cache: 40 columns、decompressed SHA `acdb77139e7f73efda4d8fe5dc29799823c4e46a687af5030e70a9dbe0b8d50a`

## 所見

### 良かった点

- GR ambiguity detector を LightGBM 学習なしの train-side diagnostic として分離した。
- mode commit と midpoint proxy を診断用途に限定し、真値側 mode selection にしない設計にした。
- 40列の target-free GR ambiguity feature cache を保存した。

### 悪かった点

- `grbm_mode_commit_proxy` RMSE 73.891128、`grbm_midpoint_proxy` RMSE 66.623786 で直接 TVT proxy は大きく壊れた。
- `grbm_likpf_midpoint_blend` も RMSE 21.681648 で `likpf_mean` 11.594898 より悪く、補正や averaging policy としては不採用。
- ambiguous flag は exp092 が悪化する領域ではなく、ambiguous=1 の exp092 RMSE 9.244988 は ambiguous=0 の 9.422932 より良い。flat flag 側は exp092 RMSE 9.424383 とやや悪いが、直接 proxy はさらに悪い。

## リスク / 注意

- detector に真値側 mode を選ばせない。
- Public LB に合わせて decoy spacing や threshold を調整しない。
- flat score well と bimodal well を分けて読む。
- この実験単体では提出しない。

## 次

1. Direct mode commit / midpoint / likPF midpoint blend は閉じる。
2. 残す場合は 40列 feature cache を exp092 add-only feature として低優先で小さく評価する。
