# 要件

## 依頼

`confidence_gate_continuity_rawtest_parity` バックログを実装する。`exp102` の confidence gate と `exp112` の expected-error gate は train-side で小改善したが、within10、worst-well、raw-test 再生成の根拠が弱い。保存済み OOF 予測を同じ shared surface で比較し、continuity / raw-test parity / worst-well guard を通して、直接 inference port へ進めるか ML feature / 診断へ戻すかを判断できる形にする。

## 制約

- Route: `pf_beam`
- 親実験: `exp102_confidence_gated_likpf_fallback_on_exp101`
- 診断親: `exp112_learned_pf_likelihood_weight_or_feature_followup`、`exp124_projection_confidence_error_map`
- 新規モデル学習はしない。保存済み OOF prediction の posthoc audit とする。
- exp112 の OOF は 155 wells subset のため、exp102 との公平比較は configured variants がそろう `shared_exp102_exp112_oof_rows` に限定する。
- `tvt_dense` high-drift gate は optional input とし、exp124 などで gate prediction artifact が存在する場合だけ比較対象へ入れる。
- `target_tvt` / `true_tvt` は評価、bucket、worst-well guard にだけ使う。
- この実装だけでは inference port / submission は作らない。
- Kaggle Notebook 実行を正とする。ローカル notebook 実行は明示的な smoke debug に限定する。
- 再現性: exp125 自体は RNG なし。gzip 入力は decompressed content SHA を記録する。raw-test hidden regeneration は未実施として記録する。

## 受け入れ基準

- `experiments/exp125_confidence_gate_continuity_rawtest_parity/` に config、train/inference notebook、補助スクリプト、記録ファイルがある。
- train notebook で exp102 / exp112 OOF prediction を読み、shared surface の metrics を保存できる。
- by-well regression、distance / tail-rank bucket、continuity summary、common worst metrics、raw-test parity checklist を生成する。
- dense/high-drift gate prediction がない場合も optional missing として終了できる。
- inference notebook は no-submission summary のみを書き、`submission.csv` を作らない。
- `make validate-exp EXP=exp125_confidence_gate_continuity_rawtest_parity` が通る。
- `make prepare-kaggle-notebooks EXP=exp125_confidence_gate_continuity_rawtest_parity EXTRA_ARGS="--notebook train --run-on-push --strict"` が通る。
