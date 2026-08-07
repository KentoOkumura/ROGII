# exp154_segment_level_dense_candidate_verifier_on_exp148

## 状態

Kaggle train v1 と inference v1 / submit 完了。Public LB 8.078 で exp148 Public LB 7.960 より悪化したため採用しない。

## 仮説

現 ML route submitted anchor の exp148 `lgb_mean` を固定 base にし、`tvt_dense` 系候補を well/segment 単位で低頻度に使えるかを診断する。exp135 / exp151 の失敗を踏まえ、全 row add-only や単純 gate ではなく、near guard と candidate path continuity guard を通した segment だけを評価する。

## 検証方針

- exp148 OOF prediction、exp073 OOF prediction、exp072 dense/PF/Beam feature cache を結合する。
- `target_tvt` は scoring と oracle readout のみに使う。
- verifier 条件は target-free な candidate disagreement、tail、near guard、candidate path continuity に限定する。
- dense 全体置換、dense-only submission、LightGBM 再学習は行わない。
- 評価は overall RMSE、within10、PF worst50、common PF+ML worst26、exp148 worst50、near-row、path continuity、worst-well regression、raw-test parity で行う。

## 所見

best `verifier_dense50_tail1500_q90_min80_clip10_a025` は exp148 base 8.501281 から 8.472280 へ -0.029002 改善した。PF worst50 は -0.480663、common PF+ML worst26 は -0.618472、exp148 worst50 は -0.292035 改善した。near `000_050` は変更なし。

一方で within10 は 0.856332 から 0.855105 へ小幅悪化し、最大 well regression は +2.287373。提出 ref `54142393` は Public LB 8.078 で、exp148 Public LB 7.960 から +0.118 悪化した。train-side の局所改善は Public LB に移らなかったため、ML route anchor には採用しない。

## 生成物

- `kaggle/output/train_v1/artifacts/exp154_segment_level_dense_candidate_verifier_on_exp148_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp154_segment_level_dense_candidate_verifier_on_exp148_by_well.csv`
- `kaggle/output/train_v1/artifacts/exp154_segment_level_dense_candidate_verifier_on_exp148_bucket_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp154_segment_level_dense_candidate_verifier_on_exp148_common_worst_metrics.csv`
- `kaggle/output/train_v1/artifacts/exp154_segment_level_dense_candidate_verifier_on_exp148_summary.json`
- `kaggle/output/inference_v1/submission.csv`
- `kaggle/output/inference_v1/exp154_segment_level_dense_candidate_verifier_on_exp148_inference_summary.json`
