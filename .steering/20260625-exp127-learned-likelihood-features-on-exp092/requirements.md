# 要件

## 依頼

`learned_likelihood_gate_rawtest_parity_or_ml_feature` のうち、分岐Aの raw-test parity / continuity / worst-well 診断は `exp125_confidence_gate_continuity_rawtest_parity` に統合済み。残る分岐Bとして、exp112 の target-free learned likelihood ML feature cache を exp092 系 ML に add-only confidence feature として評価する。

## 制約

- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- feature 親: `exp112_learned_pf_likelihood_weight_or_feature_followup`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- exp092 の U-projection correction plus disagreement surface、LightGBM config family、`TVT - last_known_tvt` target を固定する。
- exp112 feature cache は 155 wells subset のため、exp072/exp092 surface と exp112 feature cache の shared rows に限定して control と add-only variant を比較する。
- exp112 の `fold` は inventory にだけ使い、モデル特徴量には入れない。
- valid/test true TVT を feature source に入れない。
- この実装だけでは inference port / submission は作らない。
- Kaggle Notebook 実行を正とする。ローカル notebook 実行は明示的な smoke debug に限定する。
- 再現性: `docs/06_reproducibility.md` に従い、exp072 / exp112 gzip input は decompressed content SHA を主証拠にする。

## 受け入れ基準

- `experiments/exp127_learned_likelihood_features_on_exp092/` に config、train/inference notebook、補助スクリプト、記録ファイルがある。
- train notebook で exp072 full replay cache と exp112 ML feature cache を読み、shared-row control と learned-likelihood add-only variant を比較できる。
- metrics、by-well、bucket、projection feature summary、learned feature summary、feature importance、feature schema、OOF predictions、model manifest、summary を保存する。
- inference notebook は no-submission summary のみを書き、`submission.csv` を作らない。
- `make validate-exp EXP=exp127_learned_likelihood_features_on_exp092` が通る。
- `make prepare-kaggle-notebooks EXP=exp127_learned_likelihood_features_on_exp092 EXTRA_ARGS="--notebook train --run-on-push --strict"` が通る。
