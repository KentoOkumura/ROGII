# 設計

## アプローチ

exp127 の exp092 add-only feature audit を full-train 対応に拡張する。exp092 の base feature cache、U-projection correction、U-space disagreement、residual target は固定し、learned likelihood feature source だけを exp145 full-train cache に差し替える。

## 実験範囲

- 対象実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- feature 親: `exp145_learned_likelihood_rawtest_feature_generator_parity`
- 変更する変数: exp145 learned likelihood confidence features の add-only 投入
- 固定する変数: exp092 base surface、U-projection sources、LightGBM config family、GroupKFold 5 folds。exp092 control は再学習せず historical baseline として参照する。

## 再現性設計

- seed policy: GroupKFold seed 42。exp148 自体は新規 PF/Beam RNG を作らず、exp072/exp145 upstream cache を読む。
- stochastic 処理の有無: LightGBM GPU 学習は非 bitwise deterministic の可能性があるため deterministic anchor にはしない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp072/exp145 の保存済み生成物を利用する。
- 並列処理と乱数の関係: feature merge は RNG なし。LightGBM は `deterministic=true`、`force_col_wise=true`、固定 thread 設定。
- CPU/GPU runtime: train は GPU mode を既定、CPU deterministic mode は config に残すが active ではない。
- train cache / test feature regeneration の SHA 記録方針: gzip は decompressed content SHA を summary / session notes に記録する。
- model manifest / prediction / submission SHA 記録方針: train は model SHA、OOF prediction SHA、feature schema を保存する。inference は prediction SHA と submission SHA を保存する。
- Kaggle package bootstrap 確認方針: push 前に `prepare-kaggle-notebooks --strict` と `validate-exp` を通す。

## リスク

- リークリスク: exp145 generator は target-free だが、exp111 saved fold0 model と batch median imputation 制約を継承する。
- CV/LB 不一致リスク: exp092 Public LB 改善へ転移する保証はない。control 再学習をしないため、exp092 historical CV との差分には runtime / implementation 差が残る。
- ランタイム/メモリリスク: 3,783,989 rows で 1 variant x 3 configs x 5 folds = 15 boosters。control 再学習は行わない。
- 再現性リスク: upstream PF/Beam cache と GPU LightGBM に依存するため deterministic anchor にはしない。
