# exp278 formation gradient prefix stability risk readout on exp273 結果

## 状態

Kaggle CPU version 2でfull readoutを完了した。technical/parity guardは全てPASSしたが、
primary guardはfold正方向が3/5に留まりFAILした。事前契約どおり救済gridを行わず、
exp273 formation-gradient branchを閉じる。CV / LB anchorは更新しない。

## 仮説

exp273 full-prefix planeがvalidでも、last-512 / last-256 known-prefixでgradient方向、大きさ、
fit RMSE、rank、conditionが不安定なwellほど、gradient candidate familyのdirect RMSE回帰が大きい。

## 実装範囲

- 親: `exp273_two_dimensional_formation_gradient_transition`
- Route: `pf_beam`
- input: exp273固定2 shards / plane diagnostics / by-well metrics、competition raw train
- risk: 6 target-free stability component、pair最大、等重み平均
- cohort: exp273 full-gradient-valid 111 wells
- readout: stable SHA256 5 folds + pooled Spearman、fixed risk quintile
- model/HMM/booster/inference/submission: 0 / 0 / 0 / disabled / disabled

## 実行

- targeted tests: 7 passed
- repository final rerun: 153 passed、exp277の既存`run_approved`契約1件だけFAIL（exp278外）
- py_compile / Ruff / Jupytext round-trip / strict experiment validation: PASS
- canonical kernel: `kentookumura/exp278-gradient-prefix-stability-readout-train`
- Kaggle `id_no`: `127738648`
- version 1: runtime判定不備で全readout cellをskipしたtechnical no-op
- version 2: private CPU / GPU・TPU・internet off、約49.6秒でreadout summary出力
- compute: 0 variant / 0 config / 0 trained fold / 0 booster / HMM path生成0

## Technical / parity結果

- 3,783,989 rows / 773 wells / full-valid 111 wells: 期待値一致
- outer fold full-valid wells: `19 / 20 / 28 / 24 / 20`
- full-plane parity: 6項目すべてPASS
- candidate by-well RMSE parity: 3,865件すべてPASS、最大絶対差 `8.278e-13 ft`
- artifact file SHA / reproducibility manifest SHA: 全件一致

## Primary readout

| scope | Spearman rho |
| --- | ---: |
| fold 0 | 0.059649 |
| fold 1 | 0.177444 |
| fold 2 | 0.125889 |
| fold 3 | -0.123478 |
| fold 4 | -0.061654 |
| pooled | 0.074245 |

- positive folds: `3/5`（required `5/5`）→ FAIL
- pooled正方向: PASS
- lowest risk quintile mean bank delta RMSE: `-2.195778 ft`
- highest risk quintile mean bank delta RMSE: `+2.157694 ft` → q4 > q0 PASS
- candidate別pooled rho: `0.073438`〜`0.075834`
- secondary bank-max pooled rho: `0.077843`
- candidate別・secondaryともfold 3/4の負方向を救済せず、判定には使わない

## 再現性

- deterministic anchor: いいえ。成功runはversion 2の1回で、readout-only。
- seed policy: no RNG、well outer foldだけstable SHA256。
- input SHA: exp273 aggregate/shard SHAとraw horizontal manifestを照合済み。
- frozen feature logical SHA: `4d03bf82a5f5b8775661deaa7d544c97ff11dfda6ab422c68fa10efb0ba47f08`
- readout summary SHA: `7060c20e7de32e3ca5db6f2b0bd673838708f0c9f635070cd154a6f0ef352b74`
- reproducibility manifest SHA: `c99312a80f4f5c7c8f8d11451dc5eb38219a198b35b66b3ff4bcc864765d3db1`
- model / prediction / submission SHA: 対象外。

## 解釈

pooledと両端quintileには弱い方向性があるが、fold 3/4で符号が反転し、事前要求した再現性を満たさない。
また5候補はほぼ同じ相関パターンで、candidate別やbank-maxへの集約変更にも独立した救済根拠がない。
したがって、known-prefix plane instabilityをtarget-free gateへ進めるだけの安定した根拠はない。

## 次

component/window/clip/weight/threshold grid、fold-safe gate、HMM再実行、raw-test inference、submissionは行わない。
exp273 formation-gradient branchは完了negativeとして閉じ、新規救済backlogも追加しない。
