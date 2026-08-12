# 要件

## 依頼

`exp435_tvt_memoryless_u_rate_dzonly_hmm`のTVT-only状態を維持したまま、
状態遷移中心を`-ΔZ`からfold-safeな周辺井戸geometryを含む一行増分へ置き換える
`exp437_neighbor_geometry_tvt_only_transition_hmm`を設計する。

初回はアイデアバックログ、steering 3文書、実験ディレクトリ、設定・記録文書までを
作成して設計を確定した。続くユーザーの`exp437を実装してください`という明示指示に
より、compact self-contained Stage 0 train、正規notebook、専用contract test、
fail-closed inference guardまでを実装した。その後の`実行してください`という
明示指示によりKaggle private CPU Stage 0だけを実行し、mechanism gate FAILで
Stage 1、raw-test geometry再生成、推論、提出へ進まず閉鎖した。

## 仮説

exp435の`dz_only_r0`はTVT状態の遷移中心を

```text
mu_dz(t) = -delta_Z(t)
```

へ固定し、`memoryless_41rate`も毎行ゼロ中心のrate分布を周辺化するため、
符号付きの局所`Δ(TVT+Z)`を遷移平均へ持ち込めなかった。

保存済みgroup-safe exp226 geometry-only path `tvt_geop`は、outer-train周辺井戸の
K16 geometry field、XY local-linear kNN、ANCC方向、distance-regime kappaを
target wellへfold-safeに転送した経路である。この経路の一行差分を

```text
mu_geo(0) = tvt_geop(0) - last_known_TVT_input
mu_geo(t) = tvt_geop(t) - tvt_geop(t-1)
```

としてexp435と同じTVT-only transition kernelへ直接入れれば、persistent U-rate状態と
rate-to-rate hysteresisを復活させず、周辺井戸由来の非ゼロ・符号付きgeometry driftを
利用できる。

## 制約

- Routeは`pf_beam`とする。
- 親は`exp435_tvt_memoryless_u_rate_dzonly_hmm`、geometry evidence parentは
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`とする。
- 変更するのはTVT-only HMMの一行transition centerだけとする。
- persistent stateはTVT確率分布だけとし、rate state、rate responsibility、
  rate-to-rate transition、branch stateを追加しない。
- exp226保存OOFは`well_id,row_idx,suffix_offset,tvt_geop,fold`だけをallowlistで読み、
  `tvt_true,error,abs_error,gr_delta,tvt_pred`をcandidate freeze前に読まない。
- `tvt_geop`はexp226の保存済みgroup-safe 5-fold OOFをSHA固定で使い、再生成しない。
- exp435のGR/typewell emission、TVT grid、start prior、position process noise、
  five-cell kernel、forward-backward、fixed32 manifestを固定する。
- exp226 GR correction、U projection、final prediction、absolute unary、blend、
  selectorは使わない。
- exp355のjoint `(TVT, U-rate)` HMMやexp394のsoft two-branch HMMへ変更しない。
- fixed32 controlは保存済みexp435 predictionを使い、parent/control HMMを再実行しない。
- Stage 0は新candidate 1本×32 wells、Stage 1は全gate PASSと別承認時だけ
  新candidate 1本×773 wellsとする。
- 公開3 test wellsをtransition、gate、threshold、比較対象の選択に使わない。
- `docs/06_reproducibility.md`に従い、入力、allowlist、geometry schedule、
  prediction、diagnostic、metricsのlogical content SHAを記録する。

## 受け入れ基準

- steering 3文書、実験配下の`config.yaml`、`README.md`、`SESSION_NOTES.md`、
  `result.md`、`metrics.json`が同じimplementation-complete /
  execution-locked契約を持つ。
- 実装時は`KAGGLE_DIRECTION.md`の未着手バックログへ`exp437`をP2として追加し、
  Stage 0完了時に削除して判断メモへ結果を移す。
- `experiment_summary.md`へ
  `stage0_fail_closed`として登録する。
- exp355、exp394、exp436と異なる単一介入であることを明記する。
- outer 5-fold、773 wells、3,783,989 score rows、保存exp226 OOF SHA、
  exp435 fixed32 manifestとprediction SHAを固定する。
- Stage 0 fixed32 mechanism gateとStage 1 full OOF promotion gateを分け、
  fixed32をCVまたはpromotion evidenceと呼ばない。
- Stage 0はexp226 geometry pathと保存exp435 dz-onlyの両方を比較対象にする。
- Stage 1は保存exp226 final `9.427109596582213`を`0.05 ft`以上改善し、
  4/5 folds、固定scope、by-well tailのAND gateを満たすことを要求する。
- RNGなし、固定sort/reduction順、gzip decompressed content SHAを採用する。
- deterministic anchorは同一設定rerunでschedule/prediction logical SHAが一致するまで
  主張しない。
- Stage 0実行後は`execution.run_hmm=false`へ戻し、Kaggle実行を再ロックする。
- Stage 0のtechnical gateは全PASS、mechanism gateはFAILであることを記録する。
- Stage 1、同じOOFでの救済、再実行、raw-test再生成、inference、submissionを
  起動しない。

## 生成物

- compact self-contained train/inferenceのJupytext `.py`と`.ipynb`
- 実装済みの正規train/inference notebook
- exp437専用contract test
- Stage 0実行時に生成するprediction、schedule manifest、well metrics、
  input manifest、summary、metricsの固定schema

## 最終状態

Kaggle private CPU Stage 0 version 1（id_no `129056603`）を完了した。
candidateはfixed32 allでexp226 geometryより`3.751804309 ft`悪化し、
persistent 16でも`6.823650264 ft`悪化したため、
`stage0_fail_closed_without_same_oof_rescue`としてterminal closeする。
