# 設計

## アプローチ

### 局所表示契約

各wellのofficial evaluation tailに、exp208と同じrow-center stride 64のcenter列を置く。各centerについてexp202と同じ128 row offset `[-64, 63]` を使い、horizontal file端ではclipする。監査対象はwindow内のevaluation-tail固有rowだけとし、clipで重複したpixelをmetricsへ重複計上しない。

typewell gridはexp202と同じtarget-free flat prior

`last_known_tvt - (Z[row_center] - last_known_z)`

を中心に、±192 ftを64 binsで切る。horizontal GRは128-row view内、typewell GRは64-bin crop内で別々にmedian/IQR robust scaleし、signed mismatch `typewell_z - horizontal_z` とabsolute mismatchを作る。欠損補間、clip、smooth、threshold、morphologyはconfigで固定する。

### Segment-local barrier / corridor

raw/smoothed absolute mismatchが両方threshold以上で、MD方向持続長とTVT方向厚みを満たすセルだけをridge barrierとする。missing/flat/全面高mismatch rowはunsupported neutralとする。

supportedな隣接rowのfree TVT intervalsを、固定最大shift以内なら接続し、segment内だけのcomponent idを付ける。各truth/candidate pathはsegment内で最初に観測できたfree componentをlocal anchorとし、endpoint forbidden、連続row線分のridge crossing、anchor componentから別componentへのtransitionを記録する。history/persistenceはsegment内だけで計算し、次segmentへ持ち越さない。

### Overlap readout

同じ `id × path` が複数windowから監査されてもsignalをOR/AND/majority統合しない。view-level判定を保持し、同一rowのcoverage countの逆数をweightにしてprimary rateを計算する。これにより各unique rowの総weightを1に保ちつつ、重複window間agreement/disagreementとboundary/core差を別に評価する。

## 実験範囲

- 対象実験: `exp249_segment_local_negative_space_gr_corridor_audit`
- Route: `pf_beam`
- 親実験: `exp246_negative_space_gr_barrier_audit`
- 局所contract親: `exp202_heatmap_mdn_candidate_generator_probe`、`exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`
- candidate cache親: `exp072_exp063_full_replay_feature_cache`
- hidden-like group親: `exp115_hidden_like_spatial_holdout_from_ppt`
- 変更する変数: global full-tail surfaceから128×64 local surfaceへの変更、stride64 overlap、segment-local component/reset、view-level agreement readout。
- 固定する変数: exp072 candidate値とfamily、bad threshold 10 ft、no training、no candidate change、no inference、no submit。
- active mode / model config / fold / booster: `1 / 0 / 0 / 0`。parent/control再学習なし。

## 出力設計

- Stage 0: preview PNG、preview pixel/axis metadata CSV、preview manifest JSON。
- Stage 1: segment-path summary CSV.gz、flagged event CSV.gz、candidate metrics CSV、group metrics CSV、overlap metrics CSV、boundary metrics CSV、by-well CSV、summary JSON、metrics JSON。
- 3.78M row×全window×全pathの完全long auditは保存しない。well内でview arraysを集約し、segment/path要約とflagged eventだけをstream出力する。

## 再現性設計

- seed policy: `no_new_rng_sorted_well_segment_local_audit`。config互換seedは42だが新規RNGなし。
- stochastic処理: 新規なし。upstream exp072 PF/Beam candidate cacheのstochastic provenanceだけを継承する。
- PF/Beam / likelihood-PF / seed bagging: 再生成せず、保存済みcandidate列をread-onlyで監査する。
- 並列処理: sorted well、single process。segment centerとfallback preview wellはsort順でdeterministicに選ぶ。
- runtime: Kaggle CPU、GPU disabled、internet disabled、num_workers 1。
- input SHA: exp072 gzip raw/decompressed、exp115 assignment、raw horizontal/typewell inventory、config、schemaを記録する。
- output SHA: PNG/CSV/JSON、gzip raw/decompressedを記録する。
- model/prediction/submission SHA: 対象外。生成しない。
- deterministic anchor: false。固定inputに対する本auditはdeterministicだがprediction/submissionを作らず、upstream cacheはstochastic provenanceを持つ。
- Kaggle bootstrap: prepare後にCPU/internet off、active mode、128/64/192/64 contract、0 booster、kernel sourcesを確認する。

## Guard設計

Stage 0は自動shape/axis assertionに加え、PNGのmanual parity確認を必須にする。Stage 1のgo/no-goは次を主指標とし、初回実行前にlimitをconfigへ固定する。

- truth instantaneous false-alert rate
- good-candidate false-alert rate
- bad-candidate precision lift
- overlap view disagreement rate
- boundary-vs-core false-alert delta
- hidden-like / 1000+ / worst-well regression risk

oracle/errorはreadout labelにだけ使い、segment選択、barrier生成、threshold選択、window統合には使わない。union oracle before/afterは計算しない。

## リスク

- リークリスク: local prior/crop/segment/thresholdをtail truth/errorで選ぶとtarget proxyになる。すべてconfig固定し、targetはscoring cellまで読まない。
- 表示parityリスク: 元表示のcolor normalizationが異なると赤ridgeの意味が変わる。Stage 0 manual gateを設け、full auditを初期無効化する。
- overlap擬似改善リスク: stride64でrowを二重計上するとmetricが偏る。inverse-coverage weightingとagreementを必須にする。
- boundaryリスク: local normalizationとcomponent resetがwindow端にsignalを作る可能性がある。boundary/coreと同一rowのoverlap disagreementを保存する。
- CV/LBリスク: train-side risk濃縮がhidden testへ移らない可能性がある。hidden-like subgroupを必須とし、Stage 1 positiveでもraw-test inferenceへ直行しない。
- ランタイム/メモリリスク: 128×64は小さいがwindow重複でview数が増える。well単位処理、state64固定、row-long全保存禁止で抑える。
- 再現性リスク: upstream candidate cacheはstochastic provenanceを持つ。decompressed SHA固定で入力差を検出する。

## 禁止事項

`valid_after_history`のsegment外伝播、window OR集約、candidate prune/平均、oracle/errorによるwindow選択、segment/stride/threshold同時grid、HMM/PF/Beam edge cut、raw-test inference、submissionを禁止する。
