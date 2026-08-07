# exp452_scale5_likpf_direct_public_lb_audit セッションノート

## 目的

tail guardで不採用となった固定scale-5 LikPFを単体Public LBで記述評価する。

## 現在の状態

- Route: `pf_beam`
- 状態: inference v1・submit-check・ユーザー外部提出scoring完了
- train-side OOF: `10.914522073`
- Public LB: `8.797`（ref `55149125`、ユーザー確認済み）
- 実装 / Kaggle run / submission file / competition submission: `1 / 1 / 1 / 1`
- competition submission by Codex: `0`
- evaluation Notebook契約: 1物理モデルにつき1本

## コマンドログ

- 2026-07-30:
  `make new-steering EXP=exp452_scale5_likpf_direct_public_lb_audit`
- 2026-07-30:
  `make new-exp EXP=exp452_scale5_likpf_direct_public_lb_audit`
- 2026-07-30:
  exp417/404/413、`docs/06_reproducibility.md`を読み、候補、seed、parity、
  hidden cardinality、LB解釈、禁止事項を設計として固定した。
- 2026-07-31:
  ユーザーの`exp452を実装してください`を実装と正規Notebook採用の承認として記録した。
- 2026-07-31:
  exp413 v4 source/config SHA、実際に使われたexp073 PF source SHA `4af212...`、
  公開参照gzip/content/candidate SHAを照合した。
- 2026-07-31:
  `EXP452_RUN_FULL_PARITY=1 ... pytest -k full_public_scale5 -s`でNotebookではなく
  PF関数だけを公開3 wellsへ適用し、`1 passed`（100.09秒）を確認した。
- 2026-07-31:
  親実装と同じNumba warm-upをwell並列化前に追加し、最終sourceで完全parity testを
  再実行して`1 passed`（80.77秒）を確認した。
- 2026-07-31:
  ユーザーの`実行してください。提出は絶対にしないでください。`を、Kaggle package、
  push、CPU inference、output取得、submit-checkの承認として記録した。
  competition submission、code submission、`kaggle competitions submit`は明示禁止。
- 2026-07-31:
  Jupytext変換/test、`py_compile`、Ruff F821、専用test `6 passed`、
  `make validate-exp EXP=exp452_scale5_likpf_direct_public_lb_audit`を実行した。
  `task` executableは環境になかったためMakefile fallbackを使った。

### 実行承認済みの作業

- Kaggle package作成・CPU実行・output取得
- submit-check

### 禁止作業

- competition submission、code submission、提出監視
- `kaggle competitions submit`

## 変更点

- Kaggle version 1でprediction/outputを生成し、`/tmp`取得後にsubmit-checkした。
- exp417候補を直接LBへ露出する別実験として採番した。
- exp413 v4のscale-5公開surfaceを将来のparity基準に固定した。
- 凍結提出順を3候補中1番目にした。
- compact self-contained inferenceはscale-5だけをmaterializeし、arithmetic mean、
  scale 3/8/12、ML、HMM、Beam、fallbackを生成しない。
- sample submission由来のwell/IDだけを動的に解決し、公開3-well固定assertはない。

## 実行量契約

- scientific variants: 1
- PF: 500 particles × 128 seeds × dynamic test wells
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- HMM / Beam / parent-control rerun: `0 / 0 / 0 / 0`
- train-side PF再実行: 0

## 再現性メモ

- seed policy: stable SHA256 per-well / feature family / seed index
- stochastic components: particle初期化、transition noise、resampling roughening
- CPU/GPU runtime: CPU-only / GPUなし / internet off / 上限30,600秒
- Kaggle kernel id:
  `kentookumura/exp452-scale5-likpf-public-audit-inference`
- run config SHA:
  `9ecd61cf980cc8d4574e0acd55e382b4de07fec520ef660445e3d3246f7b1f12`
- input manifest SHA:
  `9051daf975f1bb6f38f28dcc3272cff126968f3b6369b5eeae6398188f3765f6`
- generation manifest SHA:
  `0f57cea5096efe3919b28d77970beff01b1755a7247c506324f755a300f96e51`
- public parity target:
  `b713ade7adb5b185dacc941edf19aec324bcd7e075a8e903d33a23f59eb809f3`
- model manifest / model SHA: 非該当、model count 0を記録
- prediction logical / gzip / decompressed SHA:
  `b713ade7...` / `e87ad2e8...` / `085567c7...`
- submission SHA: `4ace1476ac4777fd7fc17742f3d3786ae05f1c436e35e37ae3f4348350f51217`
- rerun: 未実行。deterministic anchorはまだ主張しない。

## 実装検証

- source: 1,116行、11章。親exp413 current-test inferenceは1,563行、7章。
  親の12候補/ML orchestrationを持ち込まず、runtime/input、PF kernel、scale-5限定生成、
  parity、submission、metrics/SHAをNotebook上で追える構成にした。
- source SHA256: `f8777d583c9de2a5706e5c5981b418c2b51bc771d554b51b4c4ffd62122b3cf7`
- public function parity: rows `14,151`、wells `3`、float32 max abs `0.0 ft`
- candidate logical content SHA:
  `b713ade7adb5b185dacc941edf19aec324bcd7e075a8e903d33a23f59eb809f3`
- stable seed bases:
  `000d7d20=805188988`、`00bbac68=829597097`、`00e12e8b=1365511604`
- lightweight contract tests: `6 passed`
- full public parity test: `1 passed, 6 deselected`
- experiment validation: strict PASS
- Kaggle package / run / submission.csv / competition submission: `1 / 1 / 1 / 1`
- competition submission by Codex: `0`

## 初回push 400とcanonical slug短縮

- 初回push候補
  `exp452-scale5-likpf-direct-public-lb-audit-inference`は52文字で、Kaggle
  `SaveKernel 400 Bad Request`により実行開始前に拒否された。
- idとtitle由来slugは一致していた。科学条件、入力、実行量、Notebook、
  `competition_submission_approved: false`は変更していない。
- Kaggleの50文字上限に収め、意味のあるsuffixを維持するため、canonical ID / titleを
  `kentookumura/exp452-scale5-likpf-public-audit-inference` /
  `exp452 scale5 likpf public audit inference`（42文字）へ短縮して再packageした。

## Kaggle inference version 1

- kernel: `kentookumura/exp452-scale5-likpf-public-audit-inference`
- version / id_no: `1 / 129271895`
- Kaggle metadata: private、CPU、TPU/GPUなし、internet off、machine shape `None`
- status: `COMPLETE`
- Notebook runtime / artifact log timestamp: `58.784 / 74.298 sec`
- rows / wells: `14,151 / 3`
- particles / seeds / well-seed runs / trajectories:
  `500 / 128 / 384 / 192,000`
- fallback rows / wells: `0 / 0`
- exp413 v4公開surface parity: float32 max abs `0.0 ft`
- downloaded `submission.csv`: sampleとheader、行数、ID順序が完全一致。
  duplicate / missing / nonfiniteは0。
- prediction artifactとsubmissionのID順序は一致し、TVT max absは`0.0 ft`。
- submit-check: `PASS`、FAIL `0`、WARN `0`
- output保存先:
  `/tmp/kaggle-output/exp452_scale5_likpf_direct_public_lb_audit/inference_v1`
- competition submission: ユーザー外部提出ref `55149125`、status `COMPLETE`、
  Public LB `8.797`。Codex submitは0。

## Competition submission ref 55149125

- 2026-08-01、ユーザーからscoring完了の連絡を受けた。
- Kaggle APIでref `55149125`を確認:
  - submitted at: `2026-08-01T00:00:36.783Z`
  - submitted by: `kentookumura`
  - team: `SORA2`
  - status: `COMPLETE`
  - Public LB: `8.797`
  - Private LB: 未表示
- 同時点の別提出ref `55160891` / `8.248`はsubmitter `uplus26e7`であり、
  exp452へ誤帰属しない。
- Kaggle APIはsubmissionからkernel IDを返さないためユーザーへ確認し、ref
  `55149125`がexp452であるとの明示回答を得た。
- Codexはcompetition submissionを実行していない。2026-07-31の提出禁止指示を遵守し、
  外部提出済み結果の読み取りと記録だけを行った。
- scoring完了後に確認を開始したため、scoring elapsed timeは記録不能。
- 同一SHA256 seed familyのexp434 v10 arithmetic LikPF controlはPublic LB `9.807`
  （ref `55133074`）。scale-5 candidate `8.797`は`1.010`改善し、OOF改善
  `0.680375810 ft`と方向一致した。
- exp417のby-well p95 `+2.941688483 ft` / worst `+25.311274575 ft` FAILは
  維持する。Public LBからの自動昇格、再提出、パラメータ変更、救済は行わない。

## 次のアクション

1. この実験では追加run、rerun、再提出を行わない。
2. exp417のtail FAILを維持し、Public LB `8.797`は記述censusとして閉じる。
3. temperature、seed、particle、weight、候補式をLB後に変更しない。
