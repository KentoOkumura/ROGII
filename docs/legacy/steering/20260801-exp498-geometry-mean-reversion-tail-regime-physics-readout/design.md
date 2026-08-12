# 設計

## アプローチ

exp490の平均回帰は、長く続く誤差を大きく減らした一方、一部wellの正しい長期offsetも
geometryへ戻してtailを壊した。本実験は新しい予測を作らず、平均回帰が危険になる条件を
観測可能な「GR拘束の弱さ」「geometry面との競合」「suffix開始直後の非ゼロoffset」に
分解する。

primary仮説は1つだけである。known-prefix GRが弱く、GR-corrected exp226面とgeometry面が
大きく不一致で、exp490 posteriorもsuffix開始直後からgeometryを離れるwellでは、固定強度の
geometry平均回帰が正しい非ゼロoffsetを消し、exp357比tail悪化を起こしやすい。

## 実験範囲

- 対象実験: `exp498_geometry_mean_reversion_tail_regime_physics_readout`
- Route: `pf_beam`
- 親実験: `exp490_geometry_centered_mean_reverting_offset_hmm`
- 変更する変数: 予測ではなく、保存OOFに対するtarget-free物理regime readoutを追加する。
- 固定する変数: exp490 / exp357予測、truth、fold、HMM、K16、half-life、Huber、
  state / noise / grid、persistent episode定義、全入力SHA。
- 範囲外: candidate生成、HMM/PF/Beam実行、ML、blend、selector、inference、submission。

## データフロー

```text
exp490 merge v1 prediction safe columns ----+
K16 segment contract ------------------------+--> target-free well physics table
SHA-pinned raw visible prefix + typewell ----+             |
exp490 shard decoder manifests --------------+             |
                                                           +--> contract + SHA freeze
                                                                      |
fold + saved by-well/episode outcomes ------------------------------->+--> fixed readout
```

### Phase A: target-free freeze

predictionは`well,row_idx,suffix_offset,tvt_geop,exp226_pred,md_since,dmd,
k16_segment_id,k16_segment_span,rho,geometry_mean_reverting_delta_mean,
geometry_mean_reverting_hmm_std`だけを読む。horizontalは`TVT`をusecols段階で除外する。
fold、`true_tvt_readout_only`、3 error列、by-well / episode outcomeは読まない。

well単位に固定7物理量を集約し、固定bucketと`weak_gr_geometry_conflict`を付ける。
feature schema SHA、logical content SHA、input manifest SHA、contract SHAを保存してから
Phase Bへ進む。

### Phase B: truth-late readout

freeze済みwell tableへfoldと保存済みby-well metricsを1:1 joinする。episodeはwellでjoinし、
regime / complementごとにparent SSE、candidate SSE、episode count、recoveryを集計する。
primary all-AND gateだけを判定し、secondary bucketは件数と効果量の説明表に限定する。

## bucket設計

境界はrequirements.mdの絶対値を正とする。target-free分布のquantileをrun時に再計算して
境界へ使わない。empty / low-support bucketも結合せず、そのまま記録する。

primary regimeは次の式で固定する。

```text
weak_observation = prefix_gr_sigma >= 40 OR prefix_gr_information_ratio < 1
geometry_conflict = geometry_disagreement_median >= 10
material_early_offset = early_abs_offset >= 5

weak_gr_geometry_conflict =
    weak_observation AND geometry_conflict AND material_early_offset
```

segment spanは「1 ft当たりの復元力」、suffix horizonは「観測区間長」、HMM stdは
「state識別の不確実性」の解釈に使う。ただし結果を見てprimary式へ追加しない。

## 評価設計

### Technical all-AND

- 全固定SHA一致。
- prediction 3,783,989 rows / 773 wells、segment 12,368 rows、manifest 773 wells一致。
- safe-column phaseでtruth/error/outcome read 0。
- feature freeze前のfold / outcome read 0。
- well identity 1:1、finite coverage 1.0。
- bucketは各wellちょうど1つ、primary flagはboolean。
- new prediction / HMM / model / PF / Beam / booster count 0。

### Physics-regime all-AND

requirements.mdの6条件をそのまま使う。pooledだけでなく4/5 foldsの方向再現を必須にする。
gateを満たしてもCV改善やcandidate promotionとは呼ばない。

### Secondary descriptive readout

- 各固定bucketのwell数、fold coverage、mean / median delta RMSE、harm rate。
- regime / complementのpersistent episode SSE ratio、recovery delta。
- catastrophic tail captureとfalse coverage。

secondary表から別regimeを選ばない。次仮説はprimary PASS時だけ、別expで設計する。

## planned生成物

- `exp498_geometry_mean_reversion_tail_regime_physics_readout_input_manifest.json`
- `exp498_geometry_mean_reversion_tail_regime_physics_readout_feature_contract.json`
- `exp498_geometry_mean_reversion_tail_regime_physics_readout_target_free_well_features.csv`
- `exp498_geometry_mean_reversion_tail_regime_physics_readout_by_fold.csv`
- `exp498_geometry_mean_reversion_tail_regime_physics_readout_bucket_summary.csv`
- `exp498_geometry_mean_reversion_tail_regime_physics_readout_summary.json`

予測CSV、model、submissionは生成しない。

## 再現性設計

- seed policy: no RNG、well / row / segment / bucket順をstable sortする。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理: 初期実装はouter worker 1。将来並列化しても集約順をwellで固定する。
- runtime: Kaggle private CPU、GPU / internet offを予定。実行は別承認。
- gzip: exp490 predictionはraw gzip SHAと展開後content SHAを分け、展開後を主証拠にする。
- feature: schema SHAとwell-sort済みlogical content SHAを保存する。
- model / prediction / submission SHA: 対象外であることを明示し、count 0をmanifestに記録する。
- Kaggle bootstrap: package時にlocal / packaged / bootstrap configとinput contract SHAを一致確認する。
- deterministic anchor: 予測anchorではない。readout rerunのfeature / summary SHAが一致した場合だけ
  deterministic diagnosticと呼ぶ。

## リスク

- leakage: raw horizontalの`TVT`やmerged predictionのtruth/error列を早期に読む危険。
  usecols allowlistとread ledgerでfail-closedする。
- raw SHA provenance: per-well SHAはmerge input manifestではなく4 shard decoder manifestを
  正とする。4 manifest自体のSHAとscientific contract SHAを固定し、773 wellで重複なく
  結合できない場合はfail-closedする。
- post-hoc selection: fixed bucketの中から最良regimeを選ぶ危険。primaryを1式に固定する。
- proxy誤解: `abs(exp226_pred-tvt_geop)`はgeometry真値の不確実性ではない。
  geometry disagreement proxyとだけ呼ぶ。
- sparse regime: coverage不足なら閾値を緩めずFAILとする。
- CV/LB: diagnostic OOFであり、LB予測やsubmission判断に直接使わない。
- runtime / memory: 3.78M-row gzipを読むためchunked aggregationを予定し、full frame複製を避ける。

## terminal decision

- PASS: 観測可能な不確実性で復元力を弱める単一式を、別expでdesign-onlyから開始できる。
- FAIL: bucket / interaction /閾値を救済せず、exp490 mean-reversion tail regime原因追跡を終了する。
- どちらでもexp490のterminal fail-closeは変更しない。
