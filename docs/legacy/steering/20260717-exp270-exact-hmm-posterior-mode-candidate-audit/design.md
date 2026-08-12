# 設計

## アプローチ

exp209 の joint-state exact HMM を 1 回実行し、forward-backward の位置周辺 posterior と同じ emission / transition score を decoder に渡す。

1. `posterior_mean`: exp209 と同じ `sum_p p(p_t | y) * grid[p]`。
2. `marginal_map`: 各行で位置周辺 posterior が最大の grid 点。
3. `global_viterbi`: joint `(position, rate)` path の最大スコア列。
4. `topk_path_1..5`: 各 state に上位 5 部分 path を保持する exact dynamic programming で global joint-state top-5 を復元する。rank 1 は global Viterbi と同一である。
5. 復元後、TVT grid-index sequence の SHA と完全一致で重複排除する。rate path だけが異なる path は候補を増やさない。重複排除後の候補が 5 本未満でも backfill はしない。

global path score は exp209 の初期位置/rate prior、位置遷移、rate 遷移、Gaussian GR emission の総和である。`path_log_posterior = path_score - forward_log_likelihood` として path posterior mass の対数も記録する。

候補が凍結した後で真の TVT を結合し、direct RMSE 等と oracle headroom を読む。block oracle は well ごとの未知 suffix row order を 128/256/512 行に固定分割し、各 block 内 SSE が最小の候補を診断用に選ぶ。oracle path 自体は保存しない。

version 3は772 / 773 wells完了後、最後のwell開始時にKaggle time limitとなり、全loop後の保存へ到達しなかった。回復実装では候補生成を`sha256("exp270::well_shard::<well>") mod 2`の2 CPU shardsへ分ける。shard 0は363 wells / 1,792,363 rows、shard 1は410 wells / 1,991,626 rowsで、合計773 wells / 3,783,989 rowsである。

2 shard version 2では12時間指定が正しく反映されたが、全well DataFrameを保持した後の一括`concat`、object ID全件整列、gzip保存、SHA計算がmemory/time boundedでなく、両shardとも生成物を残せなかった。version 3 recoveryでは2 shard割当を維持し、各wellの候補を決定的gzip streamへ順次書き出す。prediction SHA用のrow index / float32 candidate matrixだけを一時binary streamへ退避し、最終row数を確定後に既存array-bundle契約と同じSHAをchunk計算する。exp209 parityはcandidate/controlを固定row chunkで順に比較し、`np.setdiff1d`による数百万object IDのsortを行わない。各stageはelapsed / current RSS / peak RSSを出力する。正規`train` notebookも線形parity、低圧縮chunk保存、chunk SHAを使い、両shardをSHA固定入力としてcoverage、direct/oracle readout、最終生成物保存だけを行う。

version 3で両shardのHMM / stream writeは完了したが、exp209 cacheが保存直前にnumeric列をfloat32化する一方、exp270は再計算posterior meanをfloat64でCSV保存したため、最大約0.000973 ftの保存量子化差で`1e-5 ft` parityをfail-closeした。version 4はposterior meanだけを親の保存dtypeへ正規化してからparityと保存を行い、marginal MAP / global Viterbi / top-K等の新規mode候補はfloat64を維持する。許容値、HMM、decoder、shard割当は変更しない。

## 実験範囲

- 対象実験: `exp270_exact_hmm_posterior_mode_candidate_audit`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 参照実験: exp223（別 emission）、exp236（posterior readout）、exp243/252（PF path candidate）、exp266（worst-well readout）
- 変更する変数: posterior readout / exact global decoder のみ
- 固定する変数: exp209 の grid、rate states、transition、emission、calibration、初期 prior、raw train input、score row 定義
- 対象外: self-GR emission、ML unary、PF seed、candidate blend/mean、selector、inference、submission

## top-K 実装契約

- DP state: `(time, position_index, rate_index, rank)`
- predecessor: exp209 transition grammarに従う最大 15 state と、それぞれの rank 0..4
- 保存: score 2 面と、全時刻の predecessor position/rate/rank backpointer
- terminal: 全 position/rate/rank を score 降順に並べた global top-5
- tie break: score、terminal flat state、rank、predecessor flat state の決定的順序
- exactness: 小規模 trellisを全列挙し score と joint path を比較する
- TVT dedup: 復元した position-index 列の完全一致。top-5 joint path 内だけで実施し、不足分の探索はしない

## 生成物

- 行候補 gzip: id/well/row_idx、readout-only truth、posterior mean/MAP、unique top-K TVT path、posterior mode mass/gap
- candidate metrics: overall / distance bucket / hidden-like
- by-well metrics: candidate と oracle headroom、worst-well 順位
- path summary: score、log posterior、top1 gap、dedup、edge/switch/curvature、TVT path SHA
- pairwise path distance: well 単位 candidate 間 RMSE / mean absolute distance
- oracle scope metrics: row / block128 / block256 / block512 / well
- focus well `11d0f5ac` readout
- input manifest / summary JSON / metrics.json

## 再現性設計

- seed policy: `no_rng_exact_hmm_decoder`
- stochastic 処理の有無: なし
- PF/Beam / likelihood-PF / seed bagging の有無: なし。exact HMM のみ
- 並列処理と乱数の関係: 乱数なし。well 単位 outer parallel は使わず各shardで`outer_workers=1`、Numba thread数を固定する。shard割当はwell idのstable SHA256だけで決まる
- CPU/GPU runtime と deterministic flags: CPU only、GPUなし。Numba の並列 reduction による微小差を考慮し deterministic anchor には昇格しない
- train cache / test feature regeneration の SHA 記録方針: raw input file SHA と manifest SHA、行候補 gzip raw/decompressed SHA、schema SHA、array bundle SHA を保存する
- model manifest / prediction / submission SHA 記録方針:学習 model と submission は存在しない。decoder config manifest SHA と prediction array SHA を保存する
- Kaggle package bootstrap 確認方針: notebook 冒頭で Python/NumPy/Pandas/Numba と Kaggle runtime、config 内容を表示し、internet=false の metadata を準備時 validator で確認する
- shard reproducibility: 各shardの候補gzip raw/decompressed SHA、schema SHA、path/pairwise/well/input manifest SHA、prediction array SHAをsummaryへ保存し、aggregate前にconfigへ固定する。gzipはmtime 0 / compresslevel 1で決定的に書く

## リスク

- リークリスク: truth を候補生成前に同じ frame に載せると oracle leakage を起こしやすい。generator の引数から truth を外し、候補凍結後に id で結合する。
- CV/LB 不一致リスク: 本実験は train 全体の診断であり、oracle は deployable score ではない。直接候補も submission candidate と解釈しない。
- ランタイム/メモリリスク: top-K backpointer は `T*P*R*K`。各shardでwellを1本ずつ処理し、posterior tensorとper-well DataFrameを累積保持せず、メモリ推定guardで上限超過を停止する。candidate gzipとSHA inputは逐次書き、parityは固定chunkで線形比較する。単一run・全ID sort・全well frame concatへの回帰は禁止し、stage/RSS logを必須とする。
- 再現性リスク: Numba parallel floating arithmetic の微小差。候補 sequence SHA と tolerance parity の両方を残す。
- 候補数不足: rate-state だけ異なる joint path が top-5 を占めると unique TVT path が 5 未満になる。これは監査結果として記録し、近似 backfill しない。
