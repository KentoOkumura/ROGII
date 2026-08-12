# 設計

## アプローチ

exp444のlog-space exact forward-backwardを、float64 scaled probability-spaceの
因子化engineへ置換する。各rowでjoint `(position, rate, acceleration)` の密な
遷移tensorを作らず、exp444と同じ順序で次を適用する。

1. acceleration 3x3 transition
2. destination accelerationごとのfull-support OU 41x41 rate transition
3. exp444と同じ5-offset position transition
4. exp444と同じGR emission
5. rowごとのscale正規化とlog-scale記録

backwardはforwardで保存したscaleを用いて同じ因子順を逆向きに適用し、
posterior mean/std、rate diagnostic、acceleration posteriorを同じfloat64状態から
読み出す。OU kernelは同一well内でfloat64 `delta_MD`のbit patternが完全一致する
場合だけ再利用し、丸めや量子化はしない。

fixed4は4 wellsを独立processで同時decodeし、各workerのNumba、OpenMP、MKL、
OpenBLAS threadを1へ固定する。well間に共有reductionを置かず、completion orderを
捨てて`well_id,row_idx`でstable sortする。候補を同じKaggle run内で2回実行し、
数値SHAは完全一致、runtimeは遅い方を採用する。

## 実験範囲

- 対象実験: `exp458_acceleration_state_exact_runtime_engine_audit`
- Route: `pf_beam`
- 科学仕様の構造参照: `exp444_acceleration_state_exact_hmm`
- root参照: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 独立仮説: exp444の科学結果ではなく、同一posteriorのruntime実装可能性を問う。
- 変更する変数:
  - log-spaceのedge単位logsumexpからscaled probability-space演算へ変更
  - acceleration/rate/position遷移のfactorized matrix/stencil fusion
  - exact-bit `delta_MD` OU kernel cache
  - well outer workers `1 -> 4`
- 固定する変数:
  - exp444 fixed4 well identity、入力row順、scientific contract
  - acceleration値`[-0.0005,0,+0.0005]`、transition`0.08/0.84/0.08`、
    initial zero
  - full-support OU 41 states、rate span、momentum、sig_r
  - TVT grid step/band、position kernel、sig_p
  - GR emission、sigma、start prior、rate prior、posterior mean/std readout
  - state support、float64、runtime/RSS/leakage gate

## 段階

- Stage 0A: 保存exp444 fixed4をload-only baselineにし、candidate fixed4を
  2 repeats実行するexact-equivalence/runtime audit。親rerunは0。
- Stage 0B: Stage 0A全gate PASSかつ別承認時だけ、exp444と同じfixed32
  mechanism gateをcandidate engineで評価する。Stage 0A採用repeatの4 wellsを
  再利用し、追加28 wells、親rerunは0。
- Stage 1: Stage 0A/0B全gate PASSかつ別承認時だけ、exp444と同じ773-well
  full OOF gateを評価する。
- inference/submission: Stage 1 promotion PASS後も別承認対象。
- 現在の範囲: compact self-contained候補と専用testの実装、正規train
  Notebook採用、Kaggle package、Stage 0A run。Stage 0B/1以降は行わない。

## 再現性設計

- seed policy: RNGなし。fixed4 well ledger、well/row/state走査順を固定する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed baggingの有無: なし。deterministic exact HMMのみ。
- 並列処理と乱数の関係: well-local stateをprocessごとに分離し、共有reductionを
  禁止する。future completion orderは出力へ反映しない。
- CPU/GPU runtime: Kaggle private CPU、internet off、outer workers 4、
  Numba/BLAS threads per worker 1、GPUなし、float64のみ。
- input/cache SHA: fixed4 ledger、raw rows、exp444保存3生成物、scientific contract、
  source、unique exact-bit `delta_MD` key ledgerを記録する。
- output SHA: 各repeatのprediction、posterior、diagnostic、runtime manifestを
  stable sort後に記録し、gzipはdecompressed content SHAを主証拠にする。
- model/submission SHA: model、booster、submissionを生成しないため非該当と明記する。
- Kaggle bootstrap: packageが承認された場合、embedded configとloose configの
  authorization flags、workers/threads、stage、scientific contract、source一致を
  push前に監査する。

## リスク

- リークリスク: Stage 0Aはtruth/role/fold/episode/causeをfreeze前に読まない。
  exp444保存predictionはparity比較専用でcandidateの状態や分岐へ渡さない。
- 数値リスク: log-spaceからscaled probability-spaceへ演算順が変わるためbitwise
  parent parityは要求できない。small dense referenceと固定数値許容差でfail closedする。
- runtimeリスク: exp399では同値fusionが`6.168148x`へ到達した一方、同一codeの
  Kaggle CPU varianceで`3.106782x`まで低下した履歴がある。2 repeatsの遅い方で
  `4.75x` gateを判定し、rerunだけで有利なversionを選ばない。
- メモリリスク: exp444 fixed4 peak RSS `2.282776 GB`の単純4倍は約`9.13 GB`だが、
  process copyとBLAS workspaceを含む実測RSSを`25 GB`でfail closedする。
- oversubscriptionリスク: outer 4 x inner multithreadを禁止し、effective worker/thread
  ledgerが期待値と異なればruntimeが良くてもFAILとする。
- 科学解釈リスク: Stage 0A PASSは高速化の証拠だけであり、exp444のmechanism、
  CV、promotionのpositive evidenceにしない。
