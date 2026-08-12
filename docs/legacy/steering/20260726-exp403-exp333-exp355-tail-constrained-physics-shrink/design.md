# exp403 設計

## 仮説

exp226のK16 geometryをexp333のsegment-level補正へ、exp209 exact HMMを
exp355のK16 relative-rate prior HMMへ同時置換すると、exp263の物理候補間の
相補性を保ったまま平均RMSEを改善できる。ただしfull置換は一部wellを大きく
悪化させるため、outer-trainだけで選ぶfold単位のscalar shrinkにより
exp263へ戻せば、平均gainとwell-tail safetyを同時に満たせる可能性がある。

## 根拠

保存済みOOFをread-onlyで結合した設計前診断では、exp263固定式の再構成値
`8.238331745`に対し、両成分を固定重みのまま置換した値は`8.159425494`
（`0.078906251 ft`改善）だった。

- fold gain: `+0.289010 / +0.121206 / +0.170987 /
  -0.060223 / -0.082703 ft`
- near / mid / 1000+ gain:
  `-0.023228 / +0.088329 / +0.082785 ft`
- improved / worsened wells: `407 / 366`
- by-well delta p95: `+1.983209 ft`
- worst: `86454a6f +13.412007 ft`

これはpromotion結果ではなく、exp403を作るための探索的なread-only根拠である。
同じOOFでfull置換の重みやgateを調整して採用したとは扱わない。

## 実験範囲

- 対象実験:
  `exp403_exp333_exp355_tail_constrained_physics_shrink`
- Route: `ensemble`
- 親: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- branch:
  - `exp333_exp226_k16_segment_residual_offset_target`
  - `exp355_exp226_dip_rate_prior_on_exp209`
- 変更する変数:
  - exp263 K16 50%成分をexp333へ置換
  - exp263 exact-HMM 25%成分をexp355へ置換
  - full置換差分へfold単位scalar `lambda_fold`を適用
- 固定する変数:
  - exp263の`0.50 / 0.25 / 0.25`係数
  - exp226 / LikPF / exp209 / exp333 / exp355の保存予測
  - exp226 outer reporting fold
  - row/well/suffix identity、評価scope、promotion gate
  - 親、PF、HMM、LightGBMの再実行0

## Foldとtruth境界

1. exp263 candidate cacheはgeneration foldごとのpartitionをstreaming loadする。
2. exp333 / exp355は`well_id,row_idx`でglobal key joinする。
3. exp226由来foldを`reporting_fold`、exp263 partition foldを
   `exp263_generation_fold`として別列で保持する。
4. 両ledgerがwell内一定、support `0..4`、全773 wellsを持つことを確認する。
5. 631 / 773 wellsのfold label不一致は既知provenanceとしてcross-tabへ保存し、
   row join失敗やleakage判定に使わない。
6. suffix truth、error、hidden-like roleを読む前に、入力identity、4 source
   prediction、exp263 parity、full replacement、correction、formula contract、
   content SHAをfreezeする。
7. fold `f`のλは`reporting_fold != f`のwellだけで選び、fold `f`へ固定適用する。

## λ選択

固定候補は昇順に次の9値とする。

```text
[0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
```

各outer foldで、outer-train上のpositive λが次をすべて満たす場合だけeligibleとする。

- exp263比pooled RMSE gain `>=0.01 ft`
- near 0--250 / mid 250--1000 / 1000+ の各delta `<=+0.02 ft`
- by-well delta p95 `<=0 ft`
- worst-well delta `<=+0.25 ft`

eligibleなpositive λのうち最大を選ぶ。複数値のscore比較でbest RMSEを選ばない。
positive λがなければ`lambda_fold=0`へfail closedする。outer-validを見た再選択、
連続値最適化、fold別手動overrideは行わない。

current-testへ進める場合の`lambda_test`は、5つの`lambda_fold`の中央値とする。
full-train truthでλを再fitせず、Public LBで選び直さない。

## Scientific gate

cross-fit predictionは次をすべて満たした場合だけpromotion可能とする。

- positive λを選べたfold `>=4/5`
- `lambda_test >=0.015625`
- exp263比pooled RMSE gain `>=0.03 ft`
- 改善fold `>=4/5`
- near / mid / 1000+ / hidden-like spatial /
  hidden-like typewell-purged delta `<=+0.02 ft`
- by-well delta p95 `<=0 ft`
- worst-well delta `<=+0.25 ft`
- persistent offset episode count非増加
- 512-row recovery rate非悪化

一つでもFAILならbranchを閉じ、λ候補、係数、成分、scope、gateを同じOOFで
救済しない。

## 実行量

- scientific policy: 1
- calibration λ values: 9
- reporting folds: 5
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / HMM / Beam well-runs: `0 / 0 / 0`
- parent/control rerun: 0
- prediction generation: saved OOFの決定的再結合だけ
- runtime: Kaggle private CPU、internet off
- runtime上限: 3,600秒
- peak RSS上限: 4 GB
- 実装はexp263 partition単位のstreaming集計とし、全候補wide tableを
  pandasで同時展開しない。

## 条件付きinference

promotion PASSと別承認後だけ同じexp403内で設計する。

- exp333 current-test candidateは保存済みSHAをload-only使用する。
- exp226 / LikPF / exp209 current-test成分はexp263の保存済み再生成契約を使う。
- exp355だけを凍結済みversion 2科学式で3 test wells再生成する。
- model / booster / parent retrainは0。
- `lambda_test=median(lambda_fold)`を固定適用する。
- submission生成、submit-check、competition submitはさらに別判断とする。

## 再現性設計

- seed policy: RNGなし。
- stochastic処理: なし。
- PF/Beam/HMM: 保存済みOOF load-only。条件付きtest時のexp355 exact HMMも
  deterministicで、設計済み式を変更しない。
- 並列処理: 初回実装はsingle process streaming。並列順序依存を持たせない。
- runtime: CPU only、GPUなし、internet off。
- input evidence:
  raw/file SHAに加え、gzipはdecompressed content SHA、Parquetはmanifest /
  partition logical SHAを検証する。
- prediction evidence:
  exp263 parity、full replacement、fold λ、cross-fit prediction、scope /
  by-well / episode metrics、gate JSONのid-sorted logical content SHAを保存する。
- deterministic anchor:
  design-onlyおよび初回train-side readoutではfalse。条件付きinference後も
  test prediction SHAとsubmission SHAが固定されるまでanchorと呼ばない。
- Kaggle bootstrap:
  将来package時にloose configとbootstrap内configのbyte一致、kernel source、
  CPU/internet/run flagsを確認する。

## リスク

- leakage:
  exp333 OOFには`tvt_true`が含まれるため、prediction freeze前のloader
  allowlistから明示除外する。truthはraw trainからlate joinする。
- fold:
  exp263とexp226の独立foldを同一と誤認すると誤停止または誤ったcross-fitになる。
- tail:
  scalar shrinkでは366悪化wellを識別できず、positive λが4/5 foldsで
  feasibleにならない可能性が高い。これは本実験が反証すべき中心点である。
- CV/LB:
  exp263はCV 8.238331 / Public LB 7.800で乖離があり、LBはλ選択に使えない。
- runtime/memory:
  exp263 cacheは大きいため、全候補wide mergeは禁止しstreamingする。
- family rescue:
  FAIL後にper-well router、threshold、別λ grid、component weight fitへ
  自動拡張しない。

## 実装メモ

- exp263 generation foldの5 partitionを順に読み、全候補bankをwide展開しない。
- exp333 / exp355はtarget-free allowlistだけを読み、`well_id,row_idx`から作る
  global uint64 keyで各exp263 partitionへ対応付ける。
- source partitionをParquetへfreezeし、全partitionのschema/content/formula SHAが
  確定してからraw suffix truthとexp115 hidden-like roleを読む。
- persistent offsetはexp399と同じ`abs error > 10 ft`が128行連続、returnは
  `<=5 ft`、recovery horizonは512行とする。
- 正規Notebookは置き換えず、採用前の別名compact self-contained候補として保持する。
