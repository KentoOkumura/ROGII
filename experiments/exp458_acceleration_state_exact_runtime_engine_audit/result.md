# exp458_acceleration_state_exact_runtime_engine_audit 結果

## 仮説

exp444と同じ3状態acceleration exact HMM posteriorを、科学仕様を変更せず、
scaled probability-space因子化/fused engineと4-well並列でruntime上限内に再現できる。

## 設定

- 科学仕様の構造参照: `exp444_acceleration_state_exact_hmm`
- Route: `pf_beam`
- 検証: 保存fixed4 exact-equivalence、2-repeat determinism、runtime/RSS/leakage
- metric: runtime speedupと数値parity。Stage 0AではRMSEを計算しない。
- シード: 42

## 結果

- Kaggle kernel:
  `kentookumura/exp458-accel-state-exact-runtime-eng-audit-train`
- 最終version / id_no: `2` / `129168013`
- 最終status: `stage0a_fail_closed`
- CV / Public LB / Private LB: -
- repeat runtime: `72.250458299 / 72.755703128 sec`
- 遅いrepeatでのexp444比speedup: `10.2583531175x`
- fixed32 / full投影: `582.045625024 / 14,114.606406832 sec`
- peak RSS: `13.033187866 GiB`
- prediction mean最大差: `1.0413506243e-4 ft`、閾値`1e-5`、FAIL
- prediction std最大差: `6.3565741206e-5 ft`、閾値`1e-5`、FAIL
- acceleration posterior最大差: `8.9772640210e-6`、閾値`1e-7`、FAIL
- rate diagnostic最大差: `1.0862973660e-6`、閾値`5e-6`、PASS
- small dense prediction最大誤差: `3.6379788071e-12 ft`
- small dense acceleration posterior最大誤差:
  `3.3306690738754696e-16`
- finite coverage: `1.0`
- posterior normalization最大誤差: `4.0856207306e-14`
- leakage read before freeze: `0`
- 専用test: `10 passed`
- 判定: runtime/repeatability/memory/leakageはPASS、親数値parity 3項目が
  FAILしたためStage 0A全体はFAIL closed

## 再現性

- deterministic anchor: false
- seed policy: RNGなし
- kernel version: `2`（runtime内`KAGGLE_KERNEL_RUN_TYPE`は`Batch`）
- scientific contract SHA:
  `f4a0bbbcc8b9cb44a55cff29e07f49ed251e11a896b3e877b4e2d6f9d08f4972`
- runtime engine contract SHA:
  `cb14a4f1dfc1d5de03a6e9329402fef476918b7ffb235552e9cd9d98d6c71451`
- prediction decompressed SHA（repeat 1/2共通）:
  `99c3fc141b39188e21dc1b8fef1c55998ef3878a6a0b0a7922f82ddeedf1aefb`
- acceleration posterior decompressed SHA（repeat 1/2共通）:
  `4299cbbdfb2aa9b54e4d372d4abba7479ecafaf06628bb5ec5ee22c42c4dffe9`
- diagnostic decompressed SHA（repeat 1/2共通）:
  `0220cd1408695e77c6d60d64e7849bb86a4e5037b8af4db8c81c47fd4c83457d`
- runtime manifest SHA:
  `6faad84d378268145735726de8272b3226a9c9456b4486cdc79b80255dafb869`
- summary SHA:
  `831bc1602ba3e04a266481b29c8436abc257ea8e9f024c39674728f0dd6de3c7`
- model SHA / manifest SHA: 非該当
- submission SHA: 非該当
- repeat result: 3種の主要SHAは同一。初回成功runだけなので
  deterministic anchorにはしない。

## 解釈

scaled probability-space engineはKaggle CPUのruntime/RSS上限へ十分入った。
ただしsmall dense trellisでは`3.64e-12 ft`だった誤差が、実fixed4の長系列では
mean `1.04e-4 ft`、acceleration posterior `8.98e-6`まで累積した。
したがって「高速だが、固定したexact-equivalence閾値ではexp444と同じposterior
とは認めない」が結論である。exp444のterminal runtime FAILも再分類しない。

## 次

exp458は再実行、gate緩和、precision/state/parameter変更による救済をせず閉じる。
Stage 0B/1、inference、submissionへ進まない。次の原因検証候補は、保存済み
exp458 v2とexp444を使い、長系列で最初に誤差が増幅するrow/stateを特定する
target-free studyとする。新しいHMM runやexp458の昇格には使わない。
