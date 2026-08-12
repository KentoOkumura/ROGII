# 設計

## アプローチ

Public LB参照としてexp335（CV 8.146107755881022、Public 7.517）を固定する。exp378の7候補から、各候補のlastとの差7列、exp226との差7列、候補横断統計4列、support診断2列の計20列を追加する。

exp378のsaved OOFとexp335のfoldは同一性が保証されないため直接joinしない。各exp335 outer foldで、outer-train行はinner4 OOF、outer-valid行はouter-trainだけをdonor/referenceとして生成する。5 outer valid partitionと各outer内4 inner train partitionを合わせ、計25 partitionのrole manifestを保存する。

## 実験範囲

- 対象実験: `exp382_formation_physics_candidate_addonly_on_exp335`
- Route: `ml_model`
- 親実験: `exp335_dense_prefix_recent_coverage_repeat1_on_exp304`
- 物理候補源: exp378
- 変更する変数: 20 fixed physics candidate featuresのみ。
- 固定する変数: 親370特徴、3 LightGBM config、5 fold、target、学習・blend設定。
- 実行量: 1 variant×3 config×5 fold=15 GPU booster、control再学習0。
- 除外: exp379/380 HMM/PFおよびexp381 semi-Markov出力。

## 追加20列

- `d_last_frk16_*`: 7候補。
- `d_exp226_frk16_*`: 7候補。
- `cross_surface_mean/std/range/max_abs_delta`: 4列。
- `min_surface_support`, `median_effective_donors`: 2列。

合計390列。CV後の列選択、family別特徴変更、救済variantは禁止する。

## 再現性設計

- seed policy: fixed global seedと固定outer/inner fold manifest。
- stochastic 処理の有無: 物理特徴生成は乱数なし。LightGBM seed群を明示する。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。物理候補は決定論的補助特徴。
- 並列処理と乱数の関係: 特徴生成はno RNG。実装時にLightGBM `deterministic=true`, `force_col_wise=true` を使う。
- CPU/GPU runtime と deterministic flags: 特徴生成CPU、学習Kaggle T4。15 boosters。
- train cache / test feature regeneration の SHA 記録方針: 25 role manifest、20列schema、partition別decompressed content SHAを保存する。
- model manifest / prediction / submission SHA 記録方針: config/fold/model SHA、OOF prediction SHAを保存。推論・提出は本train gate外。
- Kaggle package bootstrap 確認方針: push前にoffline bootstrapとdataset SHAを検証する。

## リスク

- リークリスク: 非nested OOF流用、outer-validをdonorへ含めること、生Formation列参照が主要リスク。role-level read auditを必須にする。
- CV/LB不一致リスク: exp335はPublic良好だがworst-well guard未達。平均改善だけでなくconfig/scope/tailを必須gateにする。
- ランタイム/メモリリスク: 25 partitionの物理特徴生成と15 GPU booster。exp378合格前にコストを使わない。
- 再現性リスク: fold manifestずれ、列順ずれ、GPU reduction差。manifest/SHAとdeterministic flagsで監査する。
