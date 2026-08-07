# exp385_gr_typewell_likelihood_on_vector_drift_paths セッションノート

## 目的

exp384の複数vector-drift pathをGR/typewell likelihoodで周辺化し、
物理モデルをさらに0.50 ft以上改善できるか検証する。

## 現在の状態

- Route: `pf_beam`
- 状態: `design_closed_by_exp383_stage0_resource_fail`
- 実装・CV・LB: なし
- Notebook: template scaffold
- Kaggle package/push/run/inference/submission: なし

## コマンドログ

2026-07-24:

```bash
make new-steering EXP=exp385_gr_typewell_likelihood_on_vector_drift_paths
make new-exp EXP=exp385_gr_typewell_likelihood_on_vector_drift_paths
```

## 変更点

- exp384 candidate値を固定し、typewell projection、GR Student-t emission、
  exact forward-backward、posterior平均だけを追加する設計を確定。
- known-prefix Stage 0、circular control、full Stage 1 gateを固定。

## 予定実行量

- Stage 0: 1 likelihood audit / 5 folds / full decoder 0
- Stage 1: 別承認時のみ773 exact decoder well-runs
- fitted model / PF / Beam / booster: `0 / 0 / 0 / 0`
- exp384 control再実行: 0
- Kaggle CPU予定

## 再現性メモ

- RNGなし、stable well/candidate/window/state順
- exp384 candidate input SHAをhard pin
- typewell index、real/circular score、transition、posterior、prediction SHAを記録予定
- deterministic anchorはrerun一致まで主張しない

## 次のアクション

1. exp383 Stage 0 resource FAILにより、未実装・未実行で閉じる。

## 2026-07-25 ユーザー閉鎖判断

- ユーザーがexp383とその後続実験の閉鎖を明示確認した。
- exp385は実装、Kaggle run、inference、submissionを行わず、再開候補にも残さない。
2. PASS後も自動実装せず、ユーザー承認を得る。
3. Stage 0 PASS後にだけ773-well full run承認を得る。
