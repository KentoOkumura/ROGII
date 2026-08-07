# 設計

## アプローチ

各outer foldのtrain wellsだけからType Well群別のsupport wells/rows、GR residual sigma、
fit RMSE、`|bias@GR50|`を計算し、6列のrow featureへ変換する。outer-valid wellには
Type Well contentだけでjoinし、未seen群はglobal priorとavailability 0へ落とす。

Stage 0ではLightGBMを学習せず、feature manifestをtruth/error結合前にSHA固定する。
その後、保存済みexp148 OOFを用いてwell単位のRMSEとの順位相関、quartile lift、
fold安定性、group-label shuffleとの差だけを読む。全gate PASSと別承認時のみ、同じ6列を
exp148へadd-onlyし、3 config × 5 foldsを学習する。

## 実験範囲

- 対象実験: `exp353_typewell_group_quality_feature_preflight`
- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- 履歴参照: `exp314_label_derived_typewell_gr_quality_addonly`
- 変更する変数: Type Well群quality 6列だけ。
- 固定する変数: exp148 fold/base feature/3 configs/saved OOF、group definition、fallback、6列schema。
- Stage 0実行量: 1 preflight / 5 folds / model 0 / booster 0。
- Stage 1予約: 1 variant / 3 configs / 5 folds / 15 GPU boosters / control booster 0。

## 再現性設計

- seed policy: Stage 0はRNGなし。Stage 1はexp148のseed/fold/thread設定を継承する。
- stochastic 処理の有無: Stage 0なし。Stage 1 LightGBMは固定seedの非bitwise GPU候補。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: feature集計順を固定し、Stage 1ではconfigごとのseedを固定する。
- CPU/GPU runtime と deterministic flags: Stage 0 Kaggle CPU、GPU/internet off。
  Stage 1は別承認時のみGPU、`gpu_use_dp=true`、deterministic/force_col_wise/thread固定を確認する。
- train cache / test feature regeneration の SHA 記録方針: fold/group prior、6列schema/content、
  fallback reasonを記録する。raw-test再生成はStage 1 PASS後も別設計境界とする。
- model manifest / prediction / submission SHA 記録方針: Stage 0非該当。Stage 1では15 modelとOOF SHAを保存する。
- Kaggle package bootstrap 確認方針: package承認後にcanonical config、feature schema、
  bootstrap内configのSHA一致を確認する。

## リスク

- リークリスク: label-derived group priorへouter-valid wellが混入すると強いleakになる。
  prior fit well IDとouter-valid well IDの積集合0をhard assertする。
- CV/LB 不一致リスク: train Type Well群coverageとtest coverageが異なるため、availability/fallbackを主readoutに含める。
- ランタイム/メモリリスク: Stage 0はwell/group集計のみ。Stage 1は15 boostersなので別承認を必須とする。
- 再現性リスク: exp148 fold/runtime版の混同を避け、入力OOF/fold/content SHAをhard preflightする。
- 解釈リスク: Stage 0相関はML gainそのものではない。PASSしてもStage 1を自動実装・実行しない。
