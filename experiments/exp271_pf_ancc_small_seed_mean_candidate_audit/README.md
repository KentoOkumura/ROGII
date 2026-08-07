# exp271 PF ANCC small-seed mean candidate audit

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU train version 2完了、train-side candidate audit支持
- CV / Public LB / Private LB: 対象外 / 未提出 / 未提出
- 作成日: 2026-07-17
- 親実験: `exp266_pf_ancc_pf_z_multiseed_stability_audit`
- candidate bank: `exp263_last_anchor_better_candidate_confidence_pair_cache` Stage 0 core 12

## 仮説

exp266でPF ANCC 64-seed mean改善量の約82%を回収した固定4-seed meanが、8-seed meanと
ほぼ同じcandidate headroomを持つなら、raw-test計算量を抑えた追加candidate候補として残せる。

## 変更点

- exp266のseed namespaceと先頭8 seed、exact PF ANCC kernel、600 particlesを固定する。
- 全773 train pseudo-tail wellでmean4 / mean8 pathとrow-wise seed disagreementを保存する。
- exp263 core 12へmean4 / mean8を追加したrow、block 128/256/512、whole-well oracleを比較する。
- LightGBM、selector、PF-Z、raw-test inference、submissionは実行しない。

## 検証方針

- 評価行: exp072 canonical pseudo-tail 3,783,989 rows / 773 wells
- upstream parity: seed 0はexp072と全行exact、mean4/mean8 per-well RMSEはexp266と照合
- candidate readout: standalone、unique-best、distance bucket、hidden-like、worst-well、seed disagreement
- leakage guard: target joinは固定path生成・保存後だけ。target/error/oracleでseedやwellを選ばない

## 実行入口

- train notebook: `exp271_pf_ancc_small_seed_mean_candidate_audit_train.ipynb`
- inference notebook: disabled guardのみ
- prepared kernel: `kentookumura/exp271-pf-ancc-small-seed-mean-audit-train`
- 初回フル実行先: Kaggle CPU、GPU/internet off

## 現時点の判断

Kaggle CPU version 2を3,783,989 rows / 773 wellsで完了した。runtimeは1,386.570秒、
seed0はexp072へ全行差0、mean4/mean8 per-well RMSEはexp266へ最大7.105e-15 ft差で一致した。

- standalone RMSE: seed0 14.493051、mean4 13.126896、mean8 13.027107
- core12 row oracle: 2.986502、+mean4 2.939958、+mean8 2.936781、+both 2.921250
- core12 whole-well oracle: 4.609190、+mean4 4.580799、+mean8 4.572218、+both 4.558439
- row unique-best: mean4単独252,772、mean8単独251,635、両方追加時合計340,687 rows

単一candidateならmean4へ縮約する。保存済みpathを使う次のadd-only selector監査では、相補性を確認するため
mean4/mean8の両方とseed disagreementを残す。raw-test inferenceとsubmissionはまだ行わない。

## 所見

- 良かった点: exp266 seed順をexact reuseし、target join前のcandidate保存、3段upstream parity、
  5 scopeすべてのcore12 oracle改善を確認できた。
- 次の評価点: oracle headroomをtarget-freeなfold-safe selector gainへ変換できるか。
- リスク: oracle改善があってもtarget-free selectorが存在するとは限らない。

## 注意

oracle headroomはdeployable selector性能ではない。監査が良くてもdirect inference、hard routing、
submissionへ進まず、raw-test計算量とtarget-free選択信号を別実験で確認する。
