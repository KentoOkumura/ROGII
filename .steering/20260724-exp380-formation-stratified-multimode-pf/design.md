# 設計

## アプローチ

exp072 PF_ANCC系を基礎に、粒子へ `base, ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA` のmodeを付ける。初期600粒子をbase 300・各formation 50へ配分する。resampling後もbase 150・各formation 25を予約し、残り300のみposterior mass比例で再配分する。

最終予測はseed単位のPF平均、Stage 1ではseed 0〜3のmean4とする。modeごとの粒子生存率、ESS、resample回数、posterior massをwell単位で保存する。

## 実験範囲

- 対象実験: `exp380_formation_stratified_multimode_pf`
- Route: `pf_beam`
- 親実験: `exp271_feature_rich_pf_rerank_on_exp148_exp264`
- PF実装親: `exp072_pf_ancc`
- 物理候補源: exp378
- 変更する変数: particle mode、初期/最低配分、adaptive再配分。
- 固定する変数: PF観測・noise・state処理、fold、評価scope。
- Stage 0: seed 0、773 well-seed runs。
- Stage 1: seed 0〜3、計3,092 well-seed runs。Stage 0後に再承認。

## 再現性設計

- seed policy: `stable_sha256(base_seed, split, fold, well_id, mode, seed)`。
- stochastic 処理の有無: 粒子初期化、process noise、systematic resampling、adaptive配分。
- PF/Beam / likelihood-PF / seed bagging の有無: PFあり、Stage 1のみ4 seed bagging。
- 並列処理と乱数の関係: well/mode/seedごとに独立Generatorを作りglobal RNGを使わない。
- CPU/GPU runtime と deterministic flags: CPU。worker数とthread環境をmanifestへ記録する。
- train cache / test feature regeneration の SHA 記録方針: exp378候補、fold、粒子設定、well診断をcontent SHAで記録する。
- model manifest / prediction / submission SHA 記録方針: seed別・mean prediction SHAを記録し、submissionは対象外。
- Kaggle package bootstrap 確認方針: Stage 0前にoffline importとNumba cache pathを確認する。

## リスク

- リークリスク: exp378候補のfold role SHAを検証する。
- CV/LB不一致リスク: 最低粒子保証が弱いformation modeを人工的に残すため、directだけでなくnoveltyとtailをgateにする。
- ランタイム/メモリリスク: Stage 1は3,092 runs。Stage 0結果と実測時間を提示するまで開始しない。
- 再現性リスク: global RNG、worker scheduling、Python hashをseedに使わない。単独/並列/順序変更一致を必須にする。
