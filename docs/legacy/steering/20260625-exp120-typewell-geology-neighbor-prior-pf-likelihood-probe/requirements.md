# exp120 typewell geology neighbor prior PF likelihood probe requirements

## 目的

`typewell_geology_neighbor_prior_pf_likelihood_probe` を個別実験として実装する。typewell 側の `Geology` boundary / marker、GR regime change、exp065 native overlap neighbor prior が、PF/Beam candidate の likelihood として使える信号かを train pseudo-tail OOF で診断する。

## 背景

- exp065 は native row-lag overlap group を作成済みで、typewell GR の shift / trim で一致する group が多い。
- exp109 は native overlap neighbor prior を `likpf_mean` 後段補正として使うと global RMSE を改善したが、worst-well regression が残った。
- exp099 は multi-observation likelihood の oracle headroom は増やしたが、target-free top1 scorer としては崩壊した。
- 本実験は direct correction や submit 候補ではなく、PF likelihood 材料としての信号有無を読む。

## スコープ

- train pseudo-tail のみを対象にする。
- exp099 v2 wide feature cache を候補 TVT surface として使う。
- exp065 common typewell cluster assignments を neighbor pool として使う。
- typewell `Geology` がある場合は boundary を使い、ない場合は typewell GR changepoint を fallback marker として使う。
- horizontal GR regime change が強い row でのみ marker likelihood を効かせる。
- validation well 自身と同 fold valid wells の true TVT は neighbor source に使わない。

## 非スコープ

- hidden/test inference port。
- submission.csv 生成。
- full PF kernel の置換。
- ML route exp092 への add-only feature 化。

## 成功条件

- Kaggle train notebook で、baseline `likpf_mean`、existing candidates、`neighbor_drift_prior`、`marker_boundary_prior`、`marker_plus_neighbor_prior` の RMSE / MAE / within10 / bucket / by-well を保存できる。
- `marker_plus_neighbor_prior` が `likpf_mean` に対して改善するか、改善しない場合もどの bucket / well 条件で崩れるかが読める。
- worst-well regression が大きい場合は direct inference port 不可として記録する。

## リスク

- hidden/test horizontal には `Geology` がないため、marker は typewell 側の soft prior に限定する。
- GR regime change と geology boundary の意味が hidden test で同じとは限らない。
- exp109 と同じく global 改善しても by-well regression が残る可能性が高い。
- 候補 TVT surface を粒子 proxy として扱うため、真の PF particle likelihood とは異なる。
