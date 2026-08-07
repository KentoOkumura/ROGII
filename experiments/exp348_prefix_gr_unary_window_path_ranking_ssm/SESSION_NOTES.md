# exp348_prefix_gr_unary_window_path_ranking_ssm セッションノート

## 目的

exp332の固定window内path-level learningを保ちながら、exact partition計算を固定positive/negative path rankingへ置換する高リスク案をimplementation-onlyで実装する。

## 現在の状態

- Route: `ensemble`
- 状態: Kaggle T4 Stage 0 version 2完了、learning/runtime gate FAILでbranch閉鎖
- 親: `exp332_prefix_gr_unary_fixed_window_structured_ssm`（Stage 0 runtime FAILでterminal close済み）
- 先行順位: exp347のterminal decision後、またはユーザーの明示override時だけ検討
- CV / LB: なし / なし
- 正規Notebook: trainはcompact self-contained版を採用、inferenceはfail-closed候補のまま未採用
- compact source / 別名Notebook候補 / tests: 実装済み
- Kaggle package: version 2完了。Stage 0必要成果物だけを監査済み。Stage A model / prediction / submissionはなし

## 2026-07-22 設計確定

- ユーザー依頼により、先に提示した案4「経路ランキングloss」をexp348として採番した。
- exp332は再開せず、window schedule、training-only teacher boundary、architecture、local CE、fixed exp209 grammar、full-well exact decodeとcontrolsを継承する。
- structured NLLだけを、`softplus(0.05 + score(negative) - score(positive))`の全negative平均へ置換する。path scoreはvalid rowあたりのneural unary + fixed transition/boundary log potential平均。
- positiveはGaussian sigma`0.35 ft`のlabel-conditioned Viterbi 1本。negative最大16本はposition offset 6、rate offset 4、midpoint rate pulse 4、保存済みexp209 1、geometry-only 1に固定する。
- out-of-grid pathはclipせず除外し、positive/negative重複をdedupする。unique negative 12未満はfail-closed。model score、outer-valid error、oracleでnegativeを追加・選択しない。
- batchはexp332と同じ1 window + accumulation 4。exp347のbatched exact DPを混ぜない。
- Stage 0は1 benchmark variant / 固定16 windows / temporary neural model 1。persisted model、trained fold、LightGBM config、booster、PF/Beam、parent/control再学習は`0/0/0/0/0/0`。
- technical checksに加え、early-holdout positive top-1`>=0.80`、positive-max-negative margin`>=0.02`、T4保守的fold外挿`<=8.5 h`、peak`<=14 GB`を要求する。
- Stage Aではranking accuracyだけを根拠にせず、full-well exact decodeがshuffle/geometry/保存済みexp209に勝ち、p95/worst safetyも通ることを必須とする。
- FAIL後のnegative family/count、margin、loss、window、architecture、decoder、epoch救済は禁止する。

## 実行量ガード

- implementation-only現在: active variant / model / fold / LightGBM config / booster / PF-Beam / control再学習 = `0 / 0 / 0 / 0 / 0 / 0 / 0`。
- 将来Stage 0承認時: `1 / temporary 1 / 0 / 0 / 0 / 0 / 0`。
- 将来Stage A承認時: active architecture 1 / fold 0 / seed 42 / neural model 1。control再学習0。
- exp347と同時にGPU実行しない。

## コマンドログ

```text
make new-steering EXP=exp348_prefix_gr_unary_window_path_ranking_ssm
make new-exp EXP=exp348_prefix_gr_unary_window_path_ranking_ssm
```

## 2026-07-24 implementation-only

- ユーザーの`exp348を実装してください`をimplementation approvalとして記録した。exp347は2026-07-23にterminal close済みで、設計上の先行条件を満たす。
- exp332 compact self-contained trainの13章、3,045行を構成参照にし、exp348 train候補は13章、path bank/ranking/Stage 0/Stage Aを含むself-contained構成へ展開した。
- positiveはsigma`0.35 ft` label-conditioned joint Viterbi 1本。固定16 negative templateはposition offset 6、constant rate offset 4、midpoint pulse 4、saved exp209 1、geometry-only 1。
- templateはgrid外をclipせず除外し、fixed exp209 grammarのjoint Viterbiでlegal pathへ決定論的に投影する。position sigma`0.35 ft`、rate-index sigma`0.25`。positive/既出negativeとのstate-sequence重複を除外し、unique 12未満は例外でfail-closedする。
- 各pathのboundary/transition log potentialはbank freeze時に事前計算する。fit loopはunary gather、事前計算済みfixed potential、全negative平均`softplus(0.05 + score_neg - score_pos)`、local CE`0.25`だけを使い、training partition sweepは0。
- window/path単位のposition/rate SHA、fixed potential、template SHA、dedup/exclusion reason、window bank SHAをmanifestへ保存する。schedule、teacher boundary、exp209 inputを確定後、model構築前に全path bankをfreezeする。
- Stage 0固定16 windowsは12 optimizer benchmark + suffix quartileごと1件の合計4 early holdoutに固定する。path bank生成、ranking forward/backward、early-holdout forward、full-well real/shuffle/geometry exact decodeを計測し、path bank全量生成コストもfold外挿へ加える。
- Stage 0 gateはtechnical、early-holdout top-1`>=0.80`、mean margin`>=0.02`、T4保守的`<=8.5 h`、peak`<=14 GB`のAND。Stage Aは従来どおりfull-well exact decodeの科学gateが必須。
- inference候補はtraining path bankとranking candidateのcurrent-test利用を明示禁止するfail-closed実装。
- 正規train/inference Notebookは上書きしていない。Kaggle package/push/run、Stage A、推論、提出も未実施。

### 実装時コマンド

```text
.venv/bin/python -m py_compile <exp348 compact train/inference> <exp348 test>
.venv/bin/ruff check <exp348 compact train/inference> <exp348 test>
.venv/bin/pytest -q tests/test_exp348_prefix_gr_unary_window_path_ranking_ssm.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <exp348 compact train/inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <exp348 compact train/inference.py>
make validate-exp EXP=exp348_prefix_gr_unary_window_path_ranking_ssm
make validate-template
make test
```

最終確認はJupytext train/inference変換と`--test`、py_compile、Ruff、strict experiment validation、template validation、exp348専用pytestをPASS。exp348専用pytestは`16 passed, 1 skipped`。skipはローカル環境にPyTorchがないためのexact decoder / gather-ranking gradient数値testであり、Kaggle T4 Stage 0ではtechnical gateとして実行時確認する。

全体pytestは784件中`776 passed, 6 skipped, 2 failed`。2 failureは未変更の`exp296_exp223_self_gr_known_tvt_support_gate`で、完了後status `completed_train_side_guard_failed_closed`に対し古いtestが`kaggle_cpu_*`を要求する不一致と、同configの`execution.run_variant=false`によりapproval testが先にfail-closeする既存不一致。exp348 testを含む残りはPASSした。

親compact比較はexp332 train `3,045`行 / 13章、exp348 train `3,880`行 / 13章。exp348はpath-bank helperとranking Stage 0を追加し、親のsetup、mask-first loading、neural unary、full-well exact decode、freeze-first readout、Stage A orchestrationを欠落なく維持した。train/inference sourceに`__file__`はない。

## 再現性メモ

- seed policy: seed 42 + stable SHA256 window/boundary/path family/order。
- path bank生成はdeterministic。将来の学習はCUDA convolution、AMP、AdamW、dropout、dataloader orderを含む。
- deterministic anchor: false。
- 将来記録: input/fold/window/boundary/positive/negative/dedup、path bank content、model、posterior、prediction、package/kernel SHA。gzipはdecompressed content SHAを主証拠とする。

## 次のアクション

1. 正規Notebook採用とKaggle T4 Stage 0は、実行量`1 variant / temporary model 1 / persisted model・fold・LightGBM・booster・PF/Beam・control再学習各0`を再提示し、別の明示承認後だけ行う。
2. Stage 0全gate PASS後もStage A fold 0の1 neural modelは別承認を必要とする。

## 2026-07-24 Kaggle T4 Stage 0 実行承認

- ユーザーの`実行してください`を、compact train候補の正規Notebook採用、Kaggle package/push/run、固定16-window Stage 0の明示承認として記録した。
- 実行量を再確認した。active variant 1、固定16 windows、temporary neural model 1、persisted model 0、trained fold 0、LightGBM config 0、booster 0、PF/Beam 0、親/control再学習0。
- Stage A/B/C、推論、提出は今回の承認範囲外。
- canonical kernel id/titleは`kentookumura/exp348-prefix-gr-window-path-ranking-ssm-stage0` / `exp348 prefix gr window path ranking ssm stage0`とし、title由来slugを一致させる。
- 正規train Notebook SHA256は`fc5caac53441e0682e64b7b9e602eeaaf6cd14c9ed88b560169a3c2f910fa5e6`。support bootstrap注入後のKaggle Notebook SHA256は`6bbe4dd9aa600d5f8f38379928b3bf90d3afe92690b50311b1eb7cc111cf5f00`。
- package内外の`config.yaml` SHA256は`94a5d798de5d943e135a71ebc0dd54f545981cbd99c7fbc3eb112c3e01d784e8`で一致した。embedded support manifestも同じconfig SHAを保持し、`selected_stage=stage0_microbenchmark`、承認scope、canonical train採用を確認した。
- `kernel-metadata.json` SHA256は`acc8f84b385825b3c77cdd86808ec85e04d84463006f93add27ff66248c98ffb`。private、internet無効、GPU有効、`machine_shape=NvidiaTeslaT4`、`run_on_push=true`、exp209/exp115 kernel sourceを確認した。
- push直前のstrict experiment validationと専用pytestを再実行し、`16 passed, 1 skipped`。skipはローカルPyTorch不在によるもので、Kaggle Stage 0 technical gateで実行時検証する。

## 2026-07-24 Kaggle T4 push試行

```text
kaggle kernels push \
  -p experiments/exp348_prefix_gr_unary_window_path_ranking_ssm/kaggle/train \
  --accelerator NvidiaTeslaT4
```

- Kaggle API応答は`Kernel push error: Maximum weekly GPU quota of 45.00 hours reached.`。notebook uploadやkernel version作成より前に拒否された。
- 最新kernel一覧に`kentookumura/exp348-prefix-gr-window-path-ranking-ssm-stage0`が存在しないことを確認した。search結果の先頭にはKaggle API由来の空ref placeholderが返ったが、実在refを列挙した最新20件にはexp348がなく、push自体も成功応答を返していない。
- このブロックは実装、technical gate、learning gate、runtime gate、memory gateのFAILではない。Stage 0結果、report SHA、artifact、model、CV/LBは存在しない。
- local configは`stage0_blocked_kaggle_gpu_quota`へ更新したため、retry時はpackageを再生成し、埋め込みconfig SHAとkernel metadataを再確認する。
- 科学契約と実行量は変更しない。Kaggle quota回復後の同一T4 retry、またはユーザー合意後の別T4環境移行まで、Stage A/B/C、推論、提出へ進めない。

## 2026-07-25 quota回復後のStage 0 retry承認

- ユーザーの`quota回復しました。実行してください。`を、同じcanonical kernel idとT4 Stage 0科学契約によるretry承認として記録した。
- 実行量を再確認した。active variant 1、固定16 windows、temporary neural model 1、persisted model 0、trained fold 0、LightGBM config 0、booster 0、PF/Beam 0、親/control再学習0。
- OAuth credentialによるKaggle CLI利用可能性を再確認した。credential実値は記録していない。
- Stage A/B/C、推論、提出は今回も承認範囲外。
- retry packageの正規train Notebook SHA256は`fc5caac53441e0682e64b7b9e602eeaaf6cd14c9ed88b560169a3c2f910fa5e6`、bootstrap注入後Kaggle Notebookは`91f05422c148d3797a49f06eb6ee8202f1183be45dfd75a70979375995b50e1f`。
- package内外の`config.yaml` SHA256は`bce20c98ca906a9dd04059189ceaa217a8c48548fa733b96c515cec05cb3ff5d`で一致し、embedded support manifestも同値。`stage0_microbenchmark`、retry approval scope、canonical train採用を確認した。
- `kernel-metadata.json` SHA256は`acc8f84b385825b3c77cdd86808ec85e04d84463006f93add27ff66248c98ffb`。canonical id/title、private、internet無効、T4、run-on-push、exp209/exp115 inputを再確認した。
- push前検証はstrict experiment/template validation、Ruff、`__file__`不在、専用pytest `16 passed, 1 skipped`をPASSした。

## 2026-07-25 Kaggle T4 Stage 0 version 1

- quota回復後の最初のretryは`Kernel push error: Notebook not found`。canonical IDのstatusは404、metadata pullは500、list/searchは空ref placeholderのみで、version/session/outputのない空kernel shellが残った状態と確認した。
- ユーザー承認のもと、空shell `kentookumura/exp348-prefix-gr-window-path-ranking-ssm-stage0`だけを削除した。科学結果、version、session、outputの削除はない。
- slug/titleとpackageを変更せず同じcanonical IDへ再pushし、Kaggle version 1を作成した。push時刻は`2026-07-25 00:05:27 UTC`、id_noは`128524049`。
- Kaggle側metadata pullをPASS。private、internet無効、`machine_shape=NvidiaTeslaT4`、competition input、exp209/exp115 kernel sourceを確認した。
- 実行packageのconfig SHA256は`bce20c98ca906a9dd04059189ceaa217a8c48548fa733b96c515cec05cb3ff5d`、notebook SHA256は`91f05422c148d3797a49f06eb6ee8202f1183be45dfd75a70979375995b50e1f`。
- 現在はRUNNINGとして扱う。CLI logsが空でも再pushせず、同じkernelの完了またはERRORを監視する。

### version 1 ERROR診断

- version 1は約258.697秒でERROR。bootstrap、CUDA/approval/scientific contract表示、raw input読込までは通過し、最初のfrozen path bank生成時に停止した。
- 例外は`ValueError: Usecols do not match columns, columns expected but not found: ['id']`。`align_exp209_window_template`がraw horizontal CSVに存在しない`id`列を読もうとしていた。
- ROGII raw schemaは`MD/X/Y/Z/.../TVT/GR/TVT_input`で`id`列を持たない。exp209 cacheは正規実装どおり`f"{well}_{row_index}"`をIDとしており、保存cache自体やinput SHAの問題ではない。
- model構築、optimizer、ranking学習、Stage 0 report生成より前の停止で、temporary model学習は0、Stage 0 gateは未評価。科学的FAILには数えない。

### version 2修正

- `align_exp209_window_template`をraw `id`列読込から、suffix row indexのbounds/unique検証後に`f"{well}_{row_index}"`を生成してexp209 cacheをreindexする実装へ変更した。
- baselineはofficial hidden suffix rowだけを持つため、full raw rowへのreindexではなく要求されたsuffix rowだけを順序保持して整列する。
- production同様にraw CSVが`id`列を持たず、baselineが逆順でもsuffix指定順へ整列する回帰testを追加した。専用pytestは`17 passed, 1 skipped`。
- path family/count、loss、margin、window、architecture、decoder、seed、input、実行量は変更していない。technical implementation fixだけとして、同じcanonical kernelのversion 2へ進む。
- version 2正規train Notebook SHA256は`940bdd261a6999030d08383d1e6e2e7a50866bec8f7489c6c9c9c1c8014c15d8`、bootstrap注入後Notebookは`14deb5dfda9c27367cbf4444e85b9c80b4e602d702076397437e55a438e20cc3`。
- package内外config SHA256は`d4deb4632eb1c9a2dd7b64097d5cb3e47f1223118ebdaeb175ed0cc1fe026a8b`で一致した。strict validationと専用pytestを再度PASSし、package notebookに旧raw `id`列読込がないことを確認した。
- 同じcanonical kernelへversion 2を`2026-07-25 00:13:51 UTC`にpushした。id_noは引き続き`128524049`、T4、internet無効、実行量と科学契約はversion 1から不変。
- version 2はversion 1の停止点約259秒を越えてRUNNINGを維持し、行ID整列修正がKaggle実環境で通過したことを確認した。
- ユーザーの`監視は止めていいです。完了したら連絡します。`に従い監視を停止した。最後の確認は`2026-07-25 00:28:33 UTC`でRUNNING。kernel自体は停止せず、再pushも行っていない。
- 完了連絡後は同じversion 2のlogsを取得し、Stage 0 reportとSHA確認が必要な生成物だけを取得してgate判定・記録を再開する。

## 2026-07-25 Kaggle T4 Stage 0 version 2完了

- ユーザーの完了連絡後、同じcanonical kernelのstatusとlogsを再確認した。version 2は`COMPLETE`、report生成時刻は`2026-07-25T00:39:46.104748+00:00`、notebook elapsedは約`1566.692 sec`。
- 固定16 windowsはoptimizer 12 / early holdout 4、path bank 16、positive path 16、negative path 256。全windowでunique negativeは16となり、最低12のtechnical gateを通過した。
- path bankはmodel構築前にfreeze済み。training exact partition sweep、outer-valid truth access、trained Stage A modelはいずれも0。Technical gateはPASSした。
- peak GPU memoryは`1.1935901641845703 GB <= 14 GB`でMemory gateをPASSした。
- early-holdout positive top-1は`0.0 < 0.80`、positive − max-negative mean marginは`-0.388485386967659 < 0.02`。Learning gateはFAILした。
- path-bank workloadは14,816 windows、保守的path-bank外挿は`256636.74303372978 sec`。p50 / 保守的fold runtime外挿は`74.22868112803303 / 75.35670035238391 h`で、固定上限`8.5 h`をFAILした。
- Technical / Learning / Runtime / Memoryの固定AND gateはFAIL。decisionを`close_without_negative_bank_margin_or_science_rescue`とし、Stage A/B/C、推論、提出へ進まない。

### 成果物・SHA監査

- 大きなoutput archiveは取得せず、判定に必要なreport、fixed16 window/boundary manifest、measurements、path-bank manifest、path-bank summaryの6生成物だけを`/tmp`へ取得した。
- Stage 0 report SHA256: `2ba5d21934ca1ce49b2e384dd1ea7414f618e4926b4167ac0991d2787fe34c9b`
- Kaggle log SHA256: `5e22fc21cfb1d8be8d5e6687fd0b4f2e974ff62c90555996d800c7aaf18caa5a`
- selection/window manifest SHA256: `f4cbf9d4085d0571da0b96d2a43ec0f8c4d211756e441b64d73136e3557a6f97`
- teacher boundary manifest SHA256: `c8cd31589989285d1757eeb7bbf40823d6b3eb3d8a65641926df10bb88cff078`
- measurements SHA256: `4f32553d3e8de0c36fb9e74a36d02dfc7d86010ea212310201d273af6be73519`
- path-bank manifest decompressed SHA256: `93a49579cae7b01b79df4349a874ade334d89a4e20e708cead9ca608ee5c3985`
- path-bank summary SHA256: `3a2db9e79a3a0ffeb141fdd0245c6eefefc6c5ac1494827905b04ee4d7268bbb`
- exp209 baseline decompressed SHA256は固定値`8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`と一致した。

## 最終判断

exp348は`stage0_failed_branch_closed`で完了した。negative family/count、margin、loss、window、architecture、decoder、epochの救済は行わない。同じpath-bank方式は再開せず、構造学習を再検討する場合はper-window Viterbi bankを持たない局所transition-consistency surrogateを独立仮説・独立Stage 0として事前固定する。
