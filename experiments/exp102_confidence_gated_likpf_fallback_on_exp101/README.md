# exp102_confidence_gated_likpf_fallback_on_exp101

## 状態

Kaggle train v2 完了。train-side audit として支持あり。

## 仮説

exp101 の row-wise selector は `likpf_mean` 単体を超えなかったが、error ranker の信頼度が極端に高い行だけ `pf_ancc` / `beam_mean` に切り替えれば、path switch と worst-well 悪化を抑えながら oracle headroom の一部を拾える可能性がある。

## 検証方針

exp101 の保存済み LightGBM booster と exp099 v2 feature cache を使って、OOF fold-safe に predicted error / binary probability / multiclass probability margin を復元する。

default は常に `likpf_mean`。`lgb_candidate_error_ranker` が `pf_ancc` または `beam_mean` を選び、信頼度条件と switch-rate cap を満たす行だけ切り替える。

## 判定

`likpf_mean` RMSE 11.594898 を基準に、RMSE、within10、switch rate、selection distribution、path switch、bucket metrics、worst-well 悪化を確認する。改善してもこの実験では提出候補化しない。

## 所見

best gate は `gate_error_margin_sr050_d020_std000020`。`likpf_mean` baseline RMSE 11.594898 から 11.561206 へ `-0.033692` 改善した。switch は 5.0% に制限され、`pf_ancc` 3.961%、`beam_mean` 1.039% だけを採用する。

ただし within10 は 0.772807 から 0.771966 へ小さく悪化し、worst-well / bucket でも改善と悪化が混在する。このまま inference port はせず、continuity / raw-test parity / worst-well guard の follow-up に回す。
