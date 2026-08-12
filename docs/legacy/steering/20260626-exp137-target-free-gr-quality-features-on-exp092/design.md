# 設計

## アプローチ

exp130 の exp092 add-only LightGBM audit scaffold を再利用する。exp092 相当の U-projection correction / disagreement features を control とし、追加 feature group として target-free GR quality を足す。

GR quality は raw train horizontal well の `MD`、`GR`、`TVT_input` から作る。行ごとの生 GR 値は特徴に入れず、coverage、missingness、interpolation gap、prefix/eval mismatch の要約だけを使う。exp065 の common typewell cluster assignments と native overlap pairs は group quality context として使う。

## 実験範囲

- 対象実験: `exp137_target_free_gr_quality_features_on_exp092`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- 変更する変数: target-free GR quality add-only feature group
- 固定する変数: exp072 full replay cache rows、exp092 U-projection feature generation、target `TVT - last_known_tvt`、GroupKFold by well、LightGBM config family

## 再現性設計

- seed policy: GroupKFold seed 42、LightGBM fixed seeds、fixed thread count
- stochastic 処理の有無: LightGBM training のみ。GR quality feature generation は deterministic。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規 PF/Beam は実行しない。exp072 deterministic cache を入力として読むだけ。
- 並列処理と乱数の関係: LightGBM deterministic flags と fixed `num_threads=8` を使う。
- CPU/GPU runtime と deterministic flags: 初回は CPU deterministic mode を active にする。GPU quota に依存しない。
- train cache / test feature regeneration の SHA 記録方針: exp072 cache、exp065 assignments、exp065 native pairs、feature schema、prediction gzip decompressed content SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: train model manifest と OOF prediction SHA を記録する。submission は生成しない。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` で config / helper SHA と kernel sources を確認する。

## リスク

- リークリスク: prefix/eval GR quality は raw GR と missingness だけから作る。true TVT、oracle error、fold label は使わない。exp065 group は typewell GR artifact 由来で target-free。
- CV/LB 不一致リスク: exp092 は Public LB と by-well regression warning があるため、OOF 小改善だけでは submit しない。
- ランタイム/メモリリスク: exp072 full cache 3.78M rows で LightGBM 3 models x 5 folds を回すため重い。初回 Kaggle CPU で実行し、必要なら `audit.fast` / row cap で smoke する。
- 再現性リスク: upstream exp072 cache と Kaggle bootstrap に依存するため、input SHA と manifest を必ず記録する。
