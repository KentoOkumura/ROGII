# 設計

## 仮説

固定mean4/mean8 pathのexp263 core-bankに対するunique headroomを測れば、PF ANCC seed bagを
低コストcandidateとして残す価値をselector学習前に判定できる。

## アプローチ

1. exp072 canonical replay cacheから評価行のID、well、target、last anchor、md_since、
   seed 0 PF ANCCを固定する。
2. raw train horizontal/typewell と exp266 exact PF ANCC kernelを使い、各wellで固定先頭8 seedを生成する。
   targetを読む前にmean4、mean8、seed std4、seed std8を凍結し、candidate path生成物へ保存する。
3. exp266 `aggregate_by_well.csv.gz` のPF ANCC mean / seed_count 4, 8とper-well RMSEを照合する。
4. exp263 Stage 0 `candidate_values/<candidate>/fold=<f>/part-000.parquet` からcore 12を読み、
   ID順・coverage・manifest SHAをfail-closedで確認する。
5. core 12、core 12 + mean4、core 12 + mean8、core 12 + mean4 + mean8を、
   row / block 128, 256, 512 / whole-wellの固定scopeでoracle比較する。
6. standalone、distance bucket、hidden-like、by-well、seed disagreement、4-vs-8差を記録する。

## 実験範囲

- 対象実験: `exp271_pf_ancc_small_seed_mean_candidate_audit`
- Route: `pf_beam`
- 親実験: `exp266_pf_ancc_pf_z_multiseed_stability_audit`
- 参照実験: `exp263_last_anchor_better_candidate_confidence_pair_cache`、
  `exp072_exp063_full_replay_feature_cache`
- 変更する変数: PF ANCC pathの集約seed数を固定 `4 / 8` とし、exp263 core bankへ追加したときの
  candidate headroomを測る。
- 固定する変数: PF ANCC kernel、particles=600、dynamics、seed順、mean集約、core 12、
  scope、tie tolerance、distance bucket、hidden-like assignments。

## 再現性設計

- seed policy: exp266 fixed seed sequenceの先頭8をexact reuseする。
- stochastic 処理の有無: PF粒子初期化、process noise、resamplingがあるが、各well/seedへ整数seedを明示する。
- PF/Beam / likelihood-PF / seed bagging の有無: PF ANCC 8-seed bagのみ。PF-Z / Beam / likelihood-PF再実行なし。
- 並列処理と乱数の関係: well-level thread並列の前に各seedをimmutable keyから生成し、
  Numba kernel内でseedを設定する。global RNGの消費順をworker schedulingへ依存させない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU/internet off、`num_workers=8`。
  1 PF dynamics × 8 seeds × 600 particles、0 booster。推定20〜40分。
- train cache / test feature regeneration の SHA 記録方針: exp072/exp266/exp263 input manifest SHA、
  candidate path raw gzip SHA / decompressed SHA / schema SHAを保存する。raw-test regenerationは対象外。
- model manifest / prediction / submission SHA 記録方針: modelとsubmissionは非該当。
  train candidate pathをprediction-like generated artifactとしてcontent SHAを記録する。
- Kaggle package bootstrap 確認方針: push前にcanonical、loose package、bootstrap ZIPの
  `config.yaml` / train source / bundled hidden-like assignment SHAを照合する。

## リスク

- リークリスク: true TVTをseed選択やgenerationへ渡すとoracle leakageになる。
  generationを先に完了・凍結し、target joinはmetrics段階だけに限定する。
- CV/LB 不一致リスク: oracle headroomはdeployable selector性能ではない。
  guard通過してもadd-only candidate/confidence feature化は別実験・別判断とする。
- ランタイム/メモリリスク: core 12 × 3.78M rowと8 seed pathを同時保持するとメモリを使う。
  candidate値はfloat32、PF seed tensorはwell単位、core cacheはcandidate単位memmapで処理する。
- 再現性リスク: exp266と異なるexperiment名で追加seedを生成するとseed系列が変わる。
  seed namespaceをexp266に固定し、seed 0 / per-well mean4/8 parityで停止する。

## 次のアクション

Kaggle実行後にmean4/mean8のoracle差、unique-best、hidden-like、worst-well、runtimeを読み、
差が小さければ4 seedへ縮約し、headroomが弱ければbranchを閉じる。
