# 設計

## アプローチ

raw horizontal の known prefix 終端直後を official evaluation start とし、MD 256 ft / stride 128 ft の segment startを target-free に固定する。末尾は right-align した start を追加して coverage を保証し、短い tail は実在範囲だけを使う。

horizontal/typewell GR は各 well 全体で独立に robust-z 化する。horizontal は 4 ft bin 内の有限値 median、typewell は重複 TVT の raw GR 平均後に 129-state gridへ線形補間する。shuffled control は typewell robust-z 列だけを stable SHA circular shiftし、realとwindow/grid/support/source/sink/candidateを共有する。

## DAG と corridor

supported node の cost は `abs(horizontal_z - typewell_z)`。右向き `dx=1, dy=-1/0/+1` と、単独 unsupported columnだけを跨ぐ `dx=2, dy=-2..+2` で DAG を作る。DP state は `(bottleneck, cumulative_cost)` の辞書順最小とし、stable predecessor orderで再現可能にする。

`tau_star` path の node 上下1 stateをmaskした second DPを1回実行する。`tau_star + 0.25` 以下の node に限定して sourceからforward、sinkからbackward reachabilityを計算し、積を near-optimal corridor とする。first segmentのanchor graphと各segmentのunanchored spanning graphを別々に保持し、primary readoutはfirstだけanchor、以後spanningを使う。

## Candidate / truth readout

row-orderを保った nearest-MD raw rowを各horizontal binの代表rowにし、固定 candidate値を変更せず nearest typewell stateへ写す。endpoint support、corridor inside/outside、corridor距離、node cost、excess、directed reachability、crossing、gap exposure、unsupported、real-vs-shuffled差を保存する。truthはgraph完成後に同じ関数へ通すだけとする。

Stage 1 の primary は family別・pooled `corridor_outside_fraction` AUC、q90 bad-rate lift、good false-alert。overlap pairは path TVT差、corridor Jaccard、candidate risk Pearson/Spearman、event一致だけを保存する。

## 実験範囲

- 対象実験: `exp250_segment_local_negative_space_gr_corridor_audit`
- Route: `pf_beam`
- 親実験: `exp246_negative_space_gr_barrier_audit`
- 参照: exp179/202 の heatmap表現と stable SHA policy、exp072 fixed cache、exp115 hidden-like assignment。
- 変更する変数: exp246 の full-tail global hard-history を MD-local directed corridorへ置き換える。
- 固定する変数: candidate family/value、target、raw input、bad threshold 10 ft、no training/no inference/no submission。

## 実行段階

- `stage0_preview`: 12 raw-only selected wellsを対象に contract/synthetic parity、plot、pixel/plot manifestを作る。full科学判定はしない。
- `stage1_full_audit`: `audit.stage0.manual_parity_confirmed=true` と `audit.stage1.enabled_after_stage0_confirmation=true` の両方がなければ停止する。773 wells をsorted single-processでstreaming処理する。

## 保存設計

- Stage 0: plot PNG、stage0 plot manifest CSV、preview manifest JSON。
- Stage 1: segment metrics CSV.gz、candidate-segment metrics CSV.gz、candidate metrics CSV、group metrics CSV、overlap metrics CSV、by-well CSV、summary JSON、metrics JSON。
- 3.78M-rowの完全long auditは保存しない。segment/candidate単位を正とする。

## 再現性設計

- seed policy: seed 42。新規乱数なし。shuffled controlのみ `SHA256(experiment_name, well, seed)` からshiftを決める。
- stochastic 処理: audit内はなし。upstream exp072 candidate cacheのstochastic provenanceだけを継承する。
- PF/Beam / likelihood-PF / seed bagging: 再生成なし。
- 並列処理: sorted well/segment/candidate/state、single process。global RNGとPython `hash()`を使わない。
- runtime: Kaggle CPU、GPU/internet disabled、worker 1。
- SHA: raw inventory、candidate cache raw/decompressed、hidden-like assignment、config、全生成物。gzip raw/decompressedを分離し、decompressedを主証拠とする。
- model/prediction/submission SHA: 対象外。生成しない。
- deterministic anchor: false。固定入力auditのdeterminismだけを主張する。
- Kaggle bootstrap: prepare後にactive mode、2 surfaces、256/128/4/±256/0.25、0 booster、CPU/internet off、kernel sourcesを確認する。

## リスクと対策

- リークリスク: truth/error/hidden-likeをsignal固定前に読まない。segment inventoryとStage 0 selectionもraw inputだけで作る。
- topologyリスク: left edge/cycle、gap over-bridge、anchor carry-overをsynthetic contractで検出する。
- supportリスク: unsupported column/stateをlow-cost nodeにせず、短segmentとno-pathをcoverageに残してprimaryから除外する。
- overlap擬似改善リスク: row riskへ統合せずpaired agreementだけを評価する。
- runtime/メモリリスク: well単位streaming、129-state DP、segment/candidate要約だけを保存する。
- 再現性リスク: stable SHA shift、deterministic gzip、sorted order、input SHA固定で検出する。
