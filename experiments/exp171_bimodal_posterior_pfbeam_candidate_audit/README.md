# exp171_bimodal_posterior_pfbeam_candidate_audit

## 状態

- ルート: `pf_beam`
- 状態: completed_train_side_rejected_no_submit
- CV: diagnostic_only
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-02
- 親/参照: `bimodal_posterior_pfbeam_candidate_audit` backlog、`exp133`、`exp167`、`exp170`、`exp072`

## 仮説

GR shift-scan surface に 6-30ft 程度離れた top2 mode がある場合、片方に hard commit するより、target-free cost 差から固定温度 posterior weight を作り `p * mode1 + (1 - p) * mode2` を出す方が、二峰性 well の大外しを抑える可能性がある。

exp133 の midpoint/proxy は大きく壊れたため、この実験では単純 midpoint を採用候補にせず、比較対象としてだけ残す。温度は config 固定で、same-OOF truth に合わせて選ばない。

## 変更点

- raw train well の known prefix から slope prior を作る。
- typewell GR と horizontal GR の local window cost surface を作り、top1 と 6-30ft 離れた top2 local minimum を抽出する。
- `posterior_temperatures` ごとに top2 cost から posterior mean 候補を作る。
- hard commit、top2 commit、midpoint、posterior mean、固定 exp072 PF/Beam 候補を同じ sampled rows で比較する。
- PF/Beam 再生成、ML 学習、candidate replacement、inference port、提出は行わない。

## 検証方針

- 検証面: train well hidden-tail と prefix-backtest sampled rows
- 主指標: candidate RMSE / MAE / within10
- 補助指標: commit 比 error gain、bimodal flag bucket、mode separation bucket、distance bucket、well 別 worst regression
- guard: near-row、longtail、bimodal detected rows、worst wells

## 実行入口

- 学習 notebook: `exp171_bimodal_posterior_pfbeam_candidate_audit_train.ipynb`
- 推論 notebook: `exp171_bimodal_posterior_pfbeam_candidate_audit_inference.ipynb`
- Kaggle 準備例:

```bash
make prepare-kaggle-notebooks EXP=exp171_bimodal_posterior_pfbeam_candidate_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp171-bimodal-posterior-pfbeam-train --title 'exp171 bimodal posterior pfbeam train' --run-on-push --strict"
```

## 所見

Kaggle train v1 は完了した。posterior / midpoint は hard commit を少し改善したが、fixed `likpf_mean` には大きく届かなかった。

- best fixed candidate: `likpf_mean` RMSE 11.471434 / MAE 6.989252 / within10 0.775439
- best posterior overall: `posterior_mean_t16` RMSE 76.698097 / MAE 39.759649 / within10 0.224286
- hidden_tail best posterior: `posterior_mean_t16` RMSE 102.301054 / MAE 50.511337 / within10 0.207900

direct replacement、PF/Beam likelihood 変更、inference port、submit は行わない。
