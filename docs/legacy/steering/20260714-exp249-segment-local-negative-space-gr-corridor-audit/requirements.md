# 要件

## 目的・仮説

`exp246_negative_space_gr_barrier_audit` で失敗した full-tail global hard-history barrier と、着想時の局所 heatmap を分離する。horizontal×typewell GR mismatch を exp202 系の局所表示契約で作り直し、赤い ridge を越えて隣接する低 mismatch corridor へ移る candidate event が、segment 内に限れば bad candidate risk を高 precision に濃縮できるかを監査する。

## 依頼

`segment_local_negative_space_gr_corridor_audit` backlog を実験化する。Stage 0 では少数 well の局所 mismatch image、axis、normalization、barrier overlay を保存し、表示契約を確認できるようにする。Stage 1 では固定した exp072 candidate cache を変更せず、773 train wells の official evaluation tail を overlapping local segment で監査する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic provenance、Kaggle bootstrap、input/output SHA を記録する。
- 親は `exp246_negative_space_gr_barrier_audit`、局所 window contract は `exp202_heatmap_mdn_candidate_generator_probe` と `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe` を正とする。
- 初期契約は horizontal 128 rows、typewell 64 bins、target-free prior center ±192 ft、row-center stride 64 とする。segment 長・stride・threshold の同時 grid は行わない。
- horizontal GR は各 128-row segment、typewell GR は各 64-bin crop の median/IQR で個別に robust scale する。well 全体の mismatch surface や full-tail color normalizationを作らない。
- barrier、corridor、segment 選択に evaluation-tail true TVT、candidate error、oracle、hidden-like role を使わない。true TVT/error は signal 固定後の scoring にだけ使う。
- segment 開始ごとに component/corridor/history を resetする。`exp246.valid_after_history` を segment 外へ持ち越さない。
- 重複 window の判定は OR/AND/majority で統合しない。各 view を保持し、overlap agreement と inverse-coverage weighted readoutだけを作る。
- candidate prune、candidate平均、candidate値変更、HMM/PF/Beam edge cut、selector学習、raw-test inference、submissionは禁止する。
- active mode は初期状態では `stage0_preview` 1本。Stage 1も実装するが、表示契約の確認前はfull auditを起動しない。
- LightGBM config 0、fold training 0、booster 0、parent/control再学習なし、GPUなし。

## 受け入れ基準

- `docs/legacy/steering/`、`config.yaml`、self-contained Jupytext train/inference notebook、`SESSION_NOTES.md`、`result.md`、`metrics.json` が揃う。
- Stage 0 は deterministic に選んだ少数wellについて、signed mismatch、absolute mismatch、barrier、candidate/truth overlay、pixel/axis metadataをPNG/CSV/JSONへ保存する。
- Stage 1 は773 wellsをwell単位streaming処理でき、segment/path summary、candidate risk readout、truth survival、overlap agreement、boundary sensitivity、distance/hidden-like group、by-well worstを保存する。
- candidateごとに instantaneous endpoint、ridge crossing、local component transition、within-segment persistence、candidate-relative barrier exposureを保存する。
- primary readout はbad-candidate precision lift、good-candidate false-alert、truth survival、overlap disagreement、boundary deltaであり、oracle improvementを採否指標にしない。
- synthetic contractで、segment外へ履歴が伝播しないこと、ridge crossing、component transition、overlap inverse-coverage weight、target-free segment constructionを確認する。
- inference notebookはdiagnostic-only guardで停止し、submissionを生成しない。
- deterministic prediction/submission anchorとは扱わない。固定inputに対するaudit determinismと、upstream exp072 candidate cacheのstochastic provenanceを分けて記録する。
- gzip生成物はraw gzip SHAとdecompressed content SHAを分け、後者を主証拠にする。

## Stage 0確認後の次アクション

Stage 0 PNGで着想元とのpixel/axis/normalization parityを確認後、同じconfigの`audit.active_mode`だけを`stage1_full_audit`へ変更してKaggle CPU full auditを行う。確認前にthreshold調整、Stage 1、inference、submissionへ進めない。
