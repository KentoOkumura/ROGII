# 要件

## 依頼

`exact_hmm_posterior_mode_candidate_audit` を exp270 として実装する。exp209 の raw exact HMM を科学的な親とし、同じ posterior から posterior mean、周辺 MAP、global Viterbi、joint-state top-K path を生成して、平均化で失われている mode 候補の train-side headroom を監査する。

ユーザー確認済みの契約は次のとおり。

- 親は exp223 ではなく exp209 とする。
- joint-state top-K は 5 とする。
- top-K は TVT grid-index 列が同一なら rate-state 列が異なっても重複として除く。
- oracle block は 128 / 256 / 512 行とする。

## 制約

- Route: `pf_beam`
- exp209 の grid、transition、Gaussian GR emission、既知 prefix calibration、初期位置・rate prior、missing-GR 処理を固定する。
- exp223 の self-GR emission は導入しない。exp236 の ML unary、exp243/252 の PF seed medoid も候補生成には使わない。
- 未知 suffix の真の TVT は全候補を凍結した後の readout にだけ使い、decoder、重複排除、候補順位、path 診断には使わない。
- row / block / well oracle は診断専用であり、oracle prediction、selector、blend、推論、submission は作らない。
- top-K=5 は joint `(TVT grid, rate)` path の厳密順位である。TVT path 重複排除後は 5 本未満を許し、足りない本数を別の近似探索で埋めない。
- full posterior tensor は保存せず、行単位の必要な周辺統計と候補 path のみ保存する。
- 再現性は `docs/06_reproducibility.md` に従う。HMM と decoder は乱数を使わず、gzip は decompressed content SHA を主証拠にする。
- Kaggle Notebook 実行を正とする。version 3 の単一run time limitを受け、ユーザー承認済みの2 deterministic well shardsをCPUで実行し、両shard完了後に同じexp270のaggregate notebookを実行する。
- well shardはwell idだけからstable SHA256で決め、候補値、truth、error、runtimeを分割に使わない。
- 2 shardsの合計HMM well-runsは773のまま維持し、各wellをちょうど1回だけ生成する。
- shard候補はwell単位で逐次gzipへ保存し、全wellのDataFrameをmemoryへ保持しない。
- exp209 parityはcandidate/controlの同一順序をchunk単位で照合し、全IDのobject sortや全control読込を行わない。
- gzip raw/decompressed SHAとprediction array SHAはmemory-boundedに計算し、stageごとのelapsed / current RSS / peak RSSをlogへ残す。

## 受け入れ基準

- exp209 の raw exact-HMM posterior mean と保存済み exp209 control が id 単位で一致し、最大絶対差が設定 tolerance 以下である。
- posterior mean は exp209 cache の保存契約どおり float32 へ正規化してから照合し、新規 mode path 候補は float64 readout を維持する。
- posterior mean、marginal MAP、global Viterbi、重複排除済み top-K path が target-free に生成される。
- top-K decoder は小さい trellis の全列挙と joint path score / path が一致する単体テストを持つ。
- TVT grid-index sequence による重複排除と、重複により unique path が 5 未満になる契約を単体テストする。
- posterior mass、mode gap、path score gap、candidate 間距離、grid edge、switch、curvature を記録する。
- direct candidate metrics、row oracle、128/256/512 block oracle、well oracle、by-well、distance bucket、hidden-like、focus well `11d0f5ac` を記録する。
- oracle headroom は direct metric と明確に分離し、oracle prediction を artifact に保存しない。
- 実行する HMM variant 数、LightGBM config 数、fold 数、booster 数が config と `SESSION_NOTES.md` に明記される。
- feature schema/content SHA、prediction array SHA、入力 manifest SHA、Kaggle kernel version が記録される。gzip は raw SHA と decompressed content SHA の両方を記録する。
- Jupytext percent `.py` を正とし、self-contained `.ipynb` へ変換でき、構文、F821、実験 validator、単体テストが通る。
- shard 0/1は候補断片、path/pairwise診断、well/input manifest、schema、raw/decompressed SHA、prediction SHAをそれぞれ保存する。
- aggregateは固定された2 shard SHAを検証し、3,783,989 rows / 773 wellsを重複・欠落なく結合してからexp209 parityと全readoutを実行する。
- streaming write前後で候補列順、well/row_idx順、finite、coverageをfail-closedで検証し、従来のin-memory array-bundle SHAと同じ値になることを単体テストする。
