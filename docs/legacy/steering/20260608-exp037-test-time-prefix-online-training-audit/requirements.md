# 要件

## 依頼

`test_time_prefix_online_training_audit` を `exp037` として実装する。
exp026 の pseudo-tail + fixed bucket-shrink anchor に対し、test-time に見える
`TVT_input` prefix を小さい重みで追加学習する online training が clean CV で改善するか監査する。

## 制約

- Route: `ml_model`
- 親実験は `exp026_pseudo_tail_bucket_shrink_inference_submit` とする。
- base training recipe、feature set、LightGBM params、fixed bucket shrink は exp026 と同じにする。
- validation well の追加学習には、その well の finite `TVT_input` prefix から疑似 cutoff 後に見える rows だけを使う。
- 評価対象 tail rows と未来の `TVT_input` は追加学習に使わない。
- Public LB は候補選択に使わない。
- organizer 明確回答未確認の rules risk を記録し、clean CV で改善しない限り inference 化しない。

## 受け入れ基準

- steering docs、実験 config、train/inference notebook、audit script が exp037 名で揃っている。
- train notebook から online-training audit が実行できる。
- control、online weight 候補、original-fold selection、well-hash holdout selection の metrics/artifacts を保存する。
- `validate_experiment.py`、`py_compile`、`pytest` が通る。
