# 設計

## アプローチ

修正版exp264 Stage A v4 / Stage C v6のraw-test-only nested candidate selectorを一般化し、
foldごとのexp263 candidate bundle内の既存`pf_ancc`
slotをexp271の保存済みPF ANCC pathでID厳密join後に差し替える。candidate contractは3 variantごとに固定し、Stage A feature auditと
Stage C nested compact生成を同じKaggle CPU runで行う。Stage Aはactual train/current-test headerで
`MD/X/Y/Z/GR`のavailabilityを検証し、formation raw/deltaをfail-closed拒否する。downstreamでは
clean 273列へvariant固有compact 74列をadd-onlyし、修正版exp264 Stage D v3保存済み
`matched_control__lgb_mean__pred_tvt`を比較baselineとして読む。

実行は一度に全variantを回さず、以下のstageへ分ける。

1. `nested_selector_<variant>`: 1 variant × 2 objectives × 5 outer × 4 inner = 40 CPU boosters。
2. `downstream_<variant>`: 1 variant × 3 configs × 5 outer = 15 GPU boosters。control再学習0。
3. `aggregate_compare`: 3 downstream OOFをSHA固定で集約する0-booster readout。

## 実験範囲

- 対象実験: `exp277_pf_ancc_small_seed_mean_addonly_selector_audit`
- Route: `ensemble`
- 親実験: `exp271_pf_ancc_small_seed_mean_candidate_audit`
- selector親: `exp264_exp263_candidate_confidence_dual_selector`
- candidate bank親: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- downstream親: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- 変更する変数: 既存`pf_ancc` slotのmean4/mean8への置換とdisagreement blockだけ。
- 固定する変数: 修正版raw-test-only context、exp263 core12、fold、sampling、selector objective/params、
  clean 273 allowlist、exp218 model configs、評価guard。

## Variant契約

- `mean4_only`: core12の`pf_ancc`を`pf_ancc_seed_mean_4`へ置換し、12候補を維持する。PF disagreement featureは加えない。
- `mean8_only`: core12の`pf_ancc`を`pf_ancc_seed_mean_8`へ置換し、12候補を維持する。PF disagreement featureは加えない。
- `mean4_mean8_disagreement`: core12の`pf_ancc`をmean4 + mean8へ置換して13候補にする。seed std4/8、particle std4/8、
  mean8-minus-mean4 signed/absoluteをcandidate confidence blockへ加える。
- single-candidateの既定controlはmean4。bothのgainがmean8依存でなければraw-test計算契約は4 seedへ縮約する。

## 再現性設計

- seed policy: exp264と同じseed 42、stable SHA256 sampling、deterministic LightGBMを維持する。
- stochastic処理: 新規PF生成なし。selector/TVT model学習のみで、global RNGによるfeature生成はしない。
- PF/Beam: exp271 version 2の固定pathだけを読み、raw/decompressed SHAを検証する。
- 並列処理: feature生成は決定的。LightGBMは親のCPU/GPU modeとthread設定を固定する。
- runtime: nested selectorはCPU、downstreamはT4。internet off。
- input SHA: exp271 gzip raw/decompressed/schema、exp263 manifest/catalog/partition、修正版exp264
  Stage A v4/Stage C v6 schema・manifest、Stage D v3 control OOF、clean 273 allowlist、
  exp218 source/config、hidden-like assignmentを記録する。
- output SHA: feature schema、compact manifest、model manifest、OOF prediction、metricsを記録する。
- submission SHA: inference/submission禁止のため非該当。
- bootstrap: package notebook内manifestとloose config/sourceのSHA一致をpush前に確認する。

## リスク

- リークリスク: exp271 path凍結後にだけtruthをjoinし、outer-valid targetはselector/final model fitへ使わない。
  actual current-testにないformation 6列はallowlistと生成schemaの二重guardで拒否する。
- CV/LBリスク: train-side pseudo-tail監査であり、guard通過だけでinference可とはしない。
- runtime/メモリ: 最大13 candidate-longとnested 40 modelsは重い。一stage一variant、chunked prediction、compact partition保存で制御する。
- 再現性: exp271 gzipとexp264 control OOFをKaggle kernel sourceから読むため、raw/decompressed/file SHAをfail-closedにする。
- 比較リスク: control再学習を省くためruntime差分がある。fold/row/model configのparityをassertし、
  corrected clean-273 stored controlを唯一のbaselineにする。旧380列controlと旧mean4 v1を混ぜない。

## 次

corrected `nested_selector_mean4_only`の40 CPU boostersを最初の再実行候補とする。control再学習は0、
downstream GPUはこのrunでは0であり、Kaggle push前に改めてユーザーの明示承認を得る。
