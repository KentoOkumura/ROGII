# 設計

## アプローチ

exp111 の learned PF observation likelihood model を再学習せず、保存済み fold0 classifier / expected-error regressor を raw train/test feature frame に適用する。candidate-long の model feature は exp111 feature schema を正とし、出力は exp112 `build_ml_features` と同じ 51列 schema に戻す。

train 側は exp099 v2 wide multi-observation likelihood cache を chunk 読みして full train coverage を作る。raw test 側は exp072 の `public_notebook_replay_audit.py` を同梱し、raw competition test files から PF/Beam/likelihood-PF replay features を再生成して同じ transform をかける。debug や再検証では rawtest cache path を明示して replay を省略できる。

## 実験範囲

- 対象実験: `exp145_learned_likelihood_rawtest_feature_generator_parity`
- Route: `ml_model`
- 親実験: `exp144_learned_likelihood_hidden_stress_and_rawtest_parity`
- 直接の feature/model 親: `exp111_learned_pf_observation_likelihood_probe`, `exp112_learned_pf_likelihood_weight_or_feature_followup`
- 変更する変数: exp112 learned likelihood feature の full train / raw test generator と schema parity audit
- 固定する変数: exp111 model、candidate set、exp112 feature schema、PF/Beam/likelihood-PF source columns、exp072 replay settings

## 再現性設計

- seed policy: exp145 自体は新規 RNG なし。raw-test PF/Beam replay は exp072 の stable per-well seed policy を継承する。
- stochastic 処理の有無: 新規学習なし。上流 exp111 LightGBM と exp072 PF replay が stochastic component。
- PF/Beam / likelihood-PF / seed bagging の有無: raw-test replay で exp072 と同じ PF/Beam/likelihood-PF を使用する。
- 並列処理と乱数の関係: exp145 の generator は deterministic pandas/numpy transform。PF replay の並列・乱数は exp072 実装に従う。
- CPU/GPU runtime と deterministic flags: CPU notebook。LightGBM は保存済み booster inference のみ。
- train cache / test feature regeneration の SHA 記録方針: gzip は raw SHA と decompressed SHA を両方保存し、主証拠は decompressed SHA。
- model manifest / prediction / submission SHA 記録方針: exp111 manifest SHA と model file SHA を記録。prediction/submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後の kernel metadata と config を確認する。

## リスク

- リークリスク: exp111 OOF labels や true errors を再利用しない。candidate-long 生成から `abs_error` / `within10` を除外する。
- CV/LB 不一致リスク: この実験は scoring しない。schema/coverage parity が通っても submit 判断にはしない。
- ランタイム/メモリリスク: full train は 3.7M rows 規模のため chunk 書き出しにする。raw-test replay は PF/Beam が重い。
- 再現性リスク: exp111 は学習時 imputation medians を保存していないため、generator は batch median imputation を summary に limitation として記録する。
