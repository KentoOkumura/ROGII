# exp491_exp226_final_tvt_rate_direct_hmm セッションノート

## 目的

fold-safeなexp226最終`tvt_pred`から毎行計算したrateを、persistent rate状態を
介さずTVT-only exact HMMのtransition centerへ直接入れる仮説を固定し、
Stage 0機構確認を実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0 version 2完了・`stage0_fail_closed`
- CV / LB: なし
- 実装承認: 2026-07-30 ユーザー指示「exp491を実装してください」
- Kaggle実行承認: 2026-07-31 ユーザー指示「実行してください」
- inference / submission承認: なし

## 2026-07-30 設計記録

ユーザー指示:

```text
まずはHMM版だけを実装し、その結果を見てPFにすすむこととしてください。
バックログ、実験ディレクトリ、steeringを作成して設計を確定させてください。
実装はまだです。
```

確定内容:

- 最新番号exp490の次として`exp491`を採番した。
- 親をexp437、予測源をexp226、rate-lag evidenceをexp408とした。
- exp437比で変更するのはtransition scheduleの入力列だけとする。
- `tvt_geop`ではなくexp226 final`tvt_pred`の一行差分を、そのままTVT transition
  centerとして使う。
- 監査上のTVT-rateは`Δtvt_pred/ΔMD`、U-rateは
  `(Δtvt_pred+ΔZ)/ΔMD`とし、HMM位置遷移では
  `U-rate×ΔMD-ΔZ=Δtvt_pred`の恒等式を要求する。
- rate state、rate smoothing、clip、scale、K16集約、momentum、
  residual offset/rate stateを追加しない。
- exp437のGaussian typewell-GR emission、TVT grid、position noise、
  start prior、forward-backwardを固定する。
- exp226 finalとHMMがsuffix GRを二重利用する点を、leakageではなく
  CV-to-hidden一般化リスクとして明記した。
- Stage 0はfixed32、Stage 1は全773 wellsとし、それぞれ別承認を必要とする。
- PF版はexp491結果を確認した後に別実験として設計する。今回の範囲に含めない。

## 実行量契約

実行承認された科学的契約は1 candidate / 32 HMM well-runsである。
version 1は32/32 wellsのHMMを完了後、科学的gate前のartifact直列化で失敗した。
version 2は同じ32 wellsを再実行するため、完了時の累計attempted HMM well-runsは64となる。

| 段階 | scientific variant | HMM well-runs | control再実行 | ML config | trained fold | booster | PF | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| version 1実績（技術失敗） | 1 | 32 | 0 | 0 | 0 | 0 | 0 | 0 |
| version 2 retry予定 | 1 | 32 | 0 | 0 | 0 | 0 | 0 | 0 |
| Stage 0予定 | 1 | 32 | 0 | 0 | 0 | 0 | 0 | 0 |
| Stage 1最大 | 1 | 773 | 0 | 0 | 0 | 0 | 0 | 0 |

Stage 0はCVではない。Stage 1はStage 0全gate PASS後の別承認が必要である。

## コマンドログ

設計scaffold作成:

```bash
make new-steering EXP=exp491_exp226_final_tvt_rate_direct_hmm
make new-exp EXP=exp491_exp226_final_tvt_rate_direct_hmm
```

## 2026-07-30 実装記録

- `exp491_exp226_final_tvt_rate_direct_hmm_compact_selfcontained_train.py`
  をJupytext percent形式で作成した。
- exp437のabsolute-TVT grid、Gaussian typewell-GR emission、start prior、
  5-cell position kernel、forward-backward、posterior meanを移植した。
- schedule sourceだけをstrict allowlistで読むexp226 final`tvt_pred`へ交換した。
- 最初の差分、隣接差分、正の`delta_MD`、TVT/U-rate恒等式を実装した。
- truth、role、fold outcome、persistent episode境界は32 wellすべての
  schedule / prediction / diagnostic freeze後にだけ読む。
- 保存exp226 finalとのall32 / control / persistent / fold / episode SSE /
  by-well tail gateを固定した。raw-GR observed/missingはStage 0 report-onlyで
  出力し、Stage 1 promotion gate用の証拠再利用監査につなぐ。
- 同一exp helper import、rate/offset/branch state、smoothing、clipping、
  scaling、segment集約、blend、selector、PFを追加していない。
- 正規train/inference scaffoldは上書きしていない。

検証:

```bash
.venv/bin/python -m py_compile \
  experiments/exp491_exp226_final_tvt_rate_direct_hmm/exp491_exp226_final_tvt_rate_direct_hmm_compact_selfcontained_train.py \
  experiments/exp491_exp226_final_tvt_rate_direct_hmm/tests/test_exp491_contract.py
.venv/bin/ruff check \
  experiments/exp491_exp226_final_tvt_rate_direct_hmm/exp491_exp226_final_tvt_rate_direct_hmm_compact_selfcontained_train.py \
  experiments/exp491_exp226_final_tvt_rate_direct_hmm/tests/test_exp491_contract.py \
  --select F821,E9
.venv/bin/pytest -q \
  experiments/exp491_exp226_final_tvt_rate_direct_hmm/tests/test_exp491_contract.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp491_exp226_final_tvt_rate_direct_hmm/exp491_exp226_final_tvt_rate_direct_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp491_exp226_final_tvt_rate_direct_hmm/exp491_exp226_final_tvt_rate_direct_hmm_compact_selfcontained_train.py
make validate-exp EXP=exp491_exp226_final_tvt_rate_direct_hmm
make update-summary
```

結果:

- `py_compile`: PASS
- Ruff `F821,E9`: PASS
- contract tests: 8件PASS
- execution-lock preview: PASS
- Jupytext conversion / round-trip: PASS
- `make validate-exp`: strict PASS
- `make update-summary`: PASS（460 experiments）
- 保存済みexp226 OOF入力preflight:
  decompressed SHA一致、10列中allowlist 5列の存在を確認。
  `tvt_true` / `tvt_geop` / `gr_delta` / `error` / `abs_error`はsourceに
  存在するため、実装どおり`read_csv(usecols=...)`時点で除外する必要がある。
- fixed32 / persistent episode asset:
  SHA一致、156,088 suffix rows、persistent 16/16 wellsのepisode coverageを確認。
- 親compact比較: exp437 9章 / 1714行、exp491 9章 / 1948行
- notebook sourceの`__file__`参照: 0
- train source SHA:
  `ad31ebf51ad8b6480bfb5e96e1d1e50615134692f86451ba92240894626ac6a4`

train/inference notebook実行、ローカルHMM、Kaggle package、push、runは実行していない。
`task validate-exp`は環境に`task`実行ファイルがないため起動できず、リポジトリ指定の
同等コマンド`make validate-exp`で検証した。

## 2026-07-31 Stage 0実行承認・package記録

ユーザー指示「実行してください」を、compact Stage 0実装の正規train notebook採用と
canonical Kaggle private CPU Stage 0を1回実行する明示承認として記録した。

実行量のpush前再確認:

| scientific variant | HMM well-runs | control再実行 | ML config | trained fold | booster | PF | Beam | GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Stage 1、inference、PF後続、submissionは未承認のままである。

認証とCLI:

- Kaggle CLI: `2.2.3`
- OAuth credentials: 利用可能
- legacy CLI credentials: 利用可能
- headless API token: 未設定だが、今回のKaggle CLI操作にはOAuth/legacyを使う

package生成:

```bash
make prepare-kaggle-notebooks \
  EXP=exp491_exp226_final_tvt_rate_direct_hmm \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp491-exp226-final-tvt-rate-direct-hmm-train \
  --title 'exp491 exp226 final tvt rate direct hmm train' \
  --run-on-push --strict"
```

package検証:

- kernel id / title slug: canonical一致
- private: true
- CPU / internet: GPU false / internet false
- run_on_push: true
- competition source: `rogii-wellbore-geology-prediction`
- kernel source: `kentookumura/exp226-k16-kappa-repro-train`
- bootstrap: 32 files
- fixed32 manifest / persistent episode asset: bootstrap内SHA一致
- bootstrap内config:
  1 variant / 32 HMM well-runs / control再実行0 / booster 0 / PF 0
- canonical notebook cell source SHA:
  `23a5df3e2d185184c5a49febca6a93b8d373dcd3c5ae8165a771157cda74a0f9`
- package notebook SHA:
  `ea4afe9d7253cb75b321c8b35df6d3d29e5532c1658302fc7854f46c6c25a5cc`
- kernel metadata SHA:
  `7f4b09c5e8f3c7f26452b751ce9e9d50a297d4b82f7a555f73c8f8d51587f022`
- train source SHA:
  `ad31ebf51ad8b6480bfb5e96e1d1e50615134692f86451ba92240894626ac6a4`
- contract tests: 8件PASS
- Jupytext round-trip / strict experiment validation: PASS

## 2026-07-31 Kaggle version 1失敗とversion 2修正

canonical kernel version 1をpushした。

```bash
kaggle kernels push \
  -p experiments/exp491_exp226_final_tvt_rate_direct_hmm/kaggle/train
kaggle kernels status \
  kentookumura/exp491-exp226-final-tvt-rate-direct-hmm-train
kaggle kernels logs \
  kentookumura/exp491-exp226-final-tvt-rate-direct-hmm-train
```

Kaggle記録:

- kernel:
  `kentookumura/exp491-exp226-final-tvt-rate-direct-hmm-train`
- version: `1`
- id_no: `129213586`
- status: `ERROR`
- Docker image:
  `gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`
- push前の同slug照合: pullは403、listはNot foundで既存kernelなし

version 1のログではbootstrap 32 files、実行量
`1 variant / 32 HMM wells / control 0 / booster 0 / PF 0 / GPU 0`を確認し、
32/32 wellsのcandidate HMM計算は完了した。最後のwell時点で
elapsed `17.740945331 s`、peak RSS `0.427150726 GiB`だった。

その後、truth / role / episode readoutより前のcombined prediction gzip保存で
次の技術エラーになった。

```text
EOFError: Compressed file ended before the end-of-stream marker was reached
```

原因は`TextIOWrapper.close()`だけでは外側の`gzip.GzipFile`がreadback前に
確実にcloseされず、gzip終端マーカーが未書き込みだったことである。
科学的gateは1件も評価されておらず、negative resultやgate不合格ではない。

修正:

- raw file、`gzip.GzipFile`、`TextIOWrapper`を入れ子context managerでcloseする。
- gzip artifactを直後にreadbackし、logical SHA一致まで確認する回帰テストを追加する。
- 科学的変数、入力、fixed32、HMM、gateは変更しない。

修正後検証:

- contract tests: 9件PASS
- `py_compile`: PASS
- Ruff: PASS
- Jupytext round-trip: PASS
- train source SHA:
  `ce5fc269ec18f81494e52926438418872216a7861a5ae0f36d6a2578ec4f3a7b`
- canonical notebook cell source SHA:
  `4cdf83b06d152a81cadd5fc55529247c7d2f761883be2915c8287aaea2219ded`
- version 2 package notebook SHA:
  `0f9d2b0a344eaffe1419479a216c8a0e03c8e0f7cf9280c91bba7c52fa7c0c67`
- version 2 package config SHA:
  `add79aaf768de617b77181efb3c1092e86e6b78d4726295cfcd16685a24b03b0`

version 2のpush前実行量は
`1 variant / 32 HMM wells / control 0 / ML config 0 / trained fold 0 /
booster 0 / PF 0 / Beam 0 / GPU 0`である。version 1と合わせた累計attempted
HMM well-runsは64だが、科学的variantは同一の1本である。

同じcanonical kernel idへのversion 2 pushは成功し、status `RUNNING`を確認した。
id_no、private / CPU / internet off、kernel / competition source、Docker imageは
version 1と同じである。

## 2026-07-31 Kaggle version 2完了・Stage 0判定

version 2はstatus `COMPLETE`で終了した。実行契約は
`1 variant / 32 HMM wells / control 0 / ML config 0 / trained fold 0 /
booster 0 / PF 0 / Beam 0 / GPU 0`で一致した。

主要値:

| scope | exp226 final | candidate | candidate - exp226 |
| --- | ---: | ---: | ---: |
| all32 | 7.976056519 | 12.290250882 | +4.314194362 ft |
| matched control | 7.081195340 | 6.101789221 | -0.979406119 ft |
| persistent | 8.757067232 | 16.169236485 | +7.412169253 ft |
| raw GR observed | 8.277209950 | 12.363989490 | +4.086779541 ft |
| raw GR missing | 7.197283117 | 12.110384813 | +4.913101696 ft |

- improving folds: `3 / 5`
- persistent episode SSE reduction: `-3.142299927`
- paired by-well delta p95: `+22.805438506 ft`
- worst well delta: `+24.277444237 ft`
- elapsed: `35.917593300 s`
- candidate HMM: `10.073263762 s`
- full-runtime projection: `243.332277751 s`
- peak RSS: `0.449199677 GiB`

technical gateは全件PASSした。mechanism gateはmatched-control safetyだけPASSし、
all32 gain、persistent gain、4/5 folds、episode SSE、by-well p95、
worst wellの6件をFAILした。`stage0_all_gates_pass=false`、
`stage1_eligible_for_separate_approval=false`である。

固定停止条件に従って`stage0_fail_closed`とし、run lockをfalseへ戻した。
Stage 1、rate/emission/grid/blend/selectorのsame-OOF救済、PF後続、
inference、submissionは行わない。ログにsummaryと全artifact SHAが含まれたため、
Kaggle output archiveは取得していない。

最終検証:

- contract tests: 9件PASS
- `py_compile`: PASS
- Ruff: PASS
- `metrics.json` parse: PASS
- `make validate-exp`: strict PASS
- `make update-summary`: PASS（461 experiments）
- canonical `config.yaml`: `run_hmm=false` / `run_approved=false`
- `kaggle/train/`は、実際にpushしたversion 2 packageのSHA再現用として
  run lock有効時の内容を保持し、fail-closed後のcanonical configでは上書きしていない

## 再現性メモ

- seed policy: HMM本体は乱数なし。fold / well / row / state /
  reduction順を固定する設計。
- stochastic components: なし。
- CPU/GPU runtime: Kaggle private CPU、GPU 0で完了。
- exp226 OOF decompressed SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`。
- fixed32 manifest SHA:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`。
- input decompressed SHAとallowlist schema SHAの照合は実装済み。
- scientific contract SHA:
  `b89a89e14cc0ff628aaa8e49814f4fa66f68eab7d5e0a732b7d1565e4fb841c4`。
- schedule manifest content SHA:
  `0dc5b8eb152ad70bda55c33917530eb737f4f3dfac83ddd3e5e6b165e4ba68d0`。
- prediction decompressed / logical / readback SHA:
  `8b137edc5ee9cf578f0c6f17d6ab7fd3c34f5b86d3c3a6cbc29f432a7d15ee56`。
- diagnostic manifest SHA:
  `b6ff40e23788d0213fc52eaf0b064d8fc64341980d998c0e9a3ce7503110b019`。
- summary artifact SHA:
  `1a434e951a62e80aaaaa4a3a2b4e9b2cb7731c0de67c26f6829f4b891b0d57d6`。
- model SHA: 学習モデルなし。
- submission SHA: 対象外。
- deterministic anchor: false。初回runだけでは主張しない。
- Kaggle bootstrap: package生成済み。正規config、bootstrap config、
  source SHA、外部asset SHAを照合済み。

## 次のアクション

exp491をStage 0で完了・fail-closedとする。Stage 1、PF後続、inference、
submissionへ進まない。
