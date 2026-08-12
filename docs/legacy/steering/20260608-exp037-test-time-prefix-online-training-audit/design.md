# 設計

## アプローチ

exp036 の prefix audit scaffold を再利用し、補正ではなく fold 内で online training rows を追加して
LightGBM residual model を再学習する。各 fold でまず exp026 相当の base model を training wells のみで fit し、
control prediction は fixed `exp014_bucket_shrink_params` を適用する。次に validation wells の可視 prefix 内へ
疑似 cutoff を 1 個作り、cutoff 後かつ original visible prefix 以内の rows を online training rows として収集する。
base training rows に online rows を concat し、online rows だけ小さい sample weight を与えた model を候補別に fit する。
prediction は original visible prefix を使った hidden tail に限定し、control との差を評価する。

## 実験範囲

- 対象実験: `exp037_test_time_prefix_online_training_audit`
- Route: `ml_model`
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- 変更する変数: online rows の追加有無、online row weight、online cutoff quantile
- 固定する変数: exp026 feature set、LightGBM params、pseudo-tail selected variant、distance bucket shrink params

## リスク

- リークリスク: validation well の target を使うが、test-time に見える finite `TVT_input` prefix のみを使う。hidden tail と cutoff 前から見えない未来 prefix は使わない。
- CV/LB 不一致リスク: public/test の visible prefix 分布と train CV の prefix 分布がずれる可能性がある。original-fold と stable well-hash holdout selection を併記する。
- ランタイム/メモリリスク: online 候補ごとに fold 再学習が必要。`N_AUG_SPLITS=1`、少数 weight 候補、row cap で制限する。
