# exp450_dzdmd_conditioned_tvt_rate_likelihood_pf セッションノート

## 目的

exp446で悪化したnaive persistent TVT-rateを、visible prefixだけで学習する
`dZ/dMD`条件付きtransitionへ置き換える設計を確定する。

## 現在の状態

- Route: `pf_beam`
- Status: `implemented_not_run`
- Priority: P3
- 科学variant: learned affine residual-AR 1本
- compact候補・専用test: 2026-07-30のユーザー依頼で実装承認・完了
- 正規Notebook採用・package・実行・Stage 1・inference・submission:
  未承認

## ユーザー確認

設計には次の2解釈があった。

1. visible prefixから`beta/intercept`を学習する条件付き遷移。
2. `beta=-1, intercept=0`による親U-rate PFの厳密座標変換。

2026-07-30のユーザー回答「1で確定してください」により1を科学候補とした。
2は実装正当性を確認するStage 0A parity sentinelだけに限定する。

## 固定差分

- valid prefix step:
  positive `delta_MD`かつ`delta_TVT_input/delta_Z/delta_MD` finite。
- valid 10 steps以上:
  `q=beta*g+intercept`のunweighted OLS。
- 不足またはnonfinite:
  `beta=-1, intercept=0`。
- transition:
  `q_t=mu_t+0.998*(q_prev-mu_prev)+0.002*noise`。
- position:
  `TVT_t=TVT_prev+q_t*delta_MD+0.005*noise`。
- 初期rate:
  親tail30 U-rateから最後のprefix gを引く。
- PF_Zのrate likelihood、zsig noise、smoothed-GR mixtureは使わない。
- exp404の500 particles、128 seeds、GR x1.0、temperature 5、
  resampling、roughening、initial spreadを固定する。

## 実行契約

- Stage 0A: parent 12 + exact transform 12 = 24 PF well-runs、
  3,072 seed-well、1,536,000 particle starts。
- Stage 0B: candidate 32 PF well-runs、4,096 seed-well、
  2,048,000 particle starts。
- Stage 1: 別承認時だけcandidate 773 PF well-runs、98,944 seed-well、
  49,472,000 particle starts、4 CPU shards。
- 保存exp404 control rerun 0。
- LightGBM config / trained fold / booster / fitted model:
  `0 / 0 / 0 / 0`。
- HMM / Beam / GPU: `0 / 0 / 0`。
- 現在の実行数はすべて0。

## 再現性

- exp404のper-well/seed stable SHA seedを継承する。
- global RNGをthread間で共有しない。
- Stage 0Aは親とexact transformで同じdraw順を使う。
- prefix-fit、config、prediction、diagnostic、truth-late ledger、
  decompressed content、kernel versionのSHAを記録する。
- unknown suffix truth/error/role/episode/fold scoreはprefix fitと予測の
  freeze後だけattachする。
- 初回成功runをdeterministic anchorにしない。

## コマンドログ

```bash
make new-steering EXP=exp450_dzdmd_conditioned_tvt_rate_likelihood_pf
make new-exp EXP=exp450_dzdmd_conditioned_tvt_rate_likelihood_pf
```

- steeringを先に作成し、その後templateからdesign-only scaffoldを生成した。
- 親コードやNotebookはコピーしていない。
- 正規train/inference Notebookはtemplate placeholderのまま。

```bash
make validate-template
make validate-config
make validate-exp EXP=exp450_dzdmd_conditioned_tvt_rate_likelihood_pf
make update-summary
```

- project template / strict config / strict experiment validation:
  `PASS / PASS / PASS`。
- config / metrics / route / status / authorization / 実行量の
  design consistency assertion: PASS。
- `backlog/KAGGLE_DIRECTION.md`未着手backlogと`experiment_summary.md`へ反映済み。

## 2026-07-30 compact self-contained実装

ユーザー依頼:

```text
exp450を実装してください
```

承認範囲は、凍結済みsteeringの「別承認後にcompact self-contained
Jupytext候補と専用testを実装する」までと解釈した。既存正規
`exp450_dzdmd_conditioned_tvt_rate_likelihood_pf_train.ipynb` /
`..._inference.ipynb`は上書きしていない。Kaggle package、push、run、
Stage 1、raw-test inference、submissionも行っていない。

作成:

- `exp450_dzdmd_conditioned_tvt_rate_likelihood_pf_compact_selfcontained_train.py`
- `exp450_dzdmd_conditioned_tvt_rate_likelihood_pf_compact_selfcontained_train.ipynb`
- `exp450_dzdmd_conditioned_tvt_rate_likelihood_pf_compact_selfcontained_inference.py`
- `exp450_dzdmd_conditioned_tvt_rate_likelihood_pf_compact_selfcontained_inference.ipynb`
- `experiments/exp450_dzdmd_conditioned_tvt_rate_likelihood_pf/tests/test_exp450_dzdmd_conditioned_tvt_rate_likelihood_pf.py`

実装内容:

- 同一wellの連続visible-prefix stepだけから
  `q=dTVT_input/dMD`、`g=dZ/dMD`を作り、positive `delta_MD`かつ
  finiteな10 step以上でunweighted OLSを1回fitする。
- support不足またはnonfinite係数は`beta=-1, intercept=0`へfallbackし、
  clip、shrink、regularization、gridは実装していない。
- prefix末尾20 valid stepsをholdoutするtarget-free backtestを実装し、
  coefficient/well選択には使わない。
- exp404のinitial U-rateから最後のprefix `g`を引いてinitial qを作り、
  first suffixはlast visible MD/Zから`delta_MD`と`g_t`を作る。
- scientific kernelは
  `q_t=mu_t+0.998*(q_prev-mu_prev)+0.002*epsilon`、
  `TVT_t=TVT_prev+q_t*delta_MD+0.005*eta`を実装した。
- particles 500、seeds 128、initial spread `4.5/0.01`、roughening、
  ESS/systematic resampling、GR x1.0、temperature 5をconfig fail-fastで固定した。
- Stage 0Aは親U-rateとexact `(TVT,q)`を単一paired kernel内で同じ初期draw、
  process draw、resampling uniform、roughening drawから生成し、seed prediction、
  particle weight、log likelihood、position/rate座標、clip/resampling、
  temperature-5 predictionを`1e-10` gateで比較する。
- Stage 0Aの12 wells / 24 PF well-runsを全PASSした場合だけ、Stage 0Bの
  fixed32 / scientific 1 variant / 32 PF well-runsを開始する。
- prefix-fit ledger、candidate prediction、PF diagnostic、selected raw inputを
  deterministic CSV/SHAでfreezeし、その後に保存exp404 control、role/fold、
  suffix truth、episode/causeをattachする。
- Stage 0Bはprefix backtest、zero-directed under-response、forward cause、
  persistent episode、persistent well/fold、matched control、runtime/RSSを
  事前登録gateでAND判定する。fixed32をCV/promotion evidenceとして扱わない。
- inference候補は明示的に例外を送出するfail-closed guardだけである。

構成比較:

- 直接のcompact parent exp404は11章・2,174行。
- exp450 train候補は13章・2,408行で、runtime/config、truth-free asset、
  prefix OLS、parent/exact parity、scientific PF、freeze、truth-late readout、
  gate、guarded orchestrationをNotebook上に展開した。
- 同一exp helper import、`__file__`、`Path(__file__)`は0。

実行契約:

- Stage 0A: technical exact variant 1、parent 12 + exact 12 =
  24 PF well-runs、3,072 seed-well、1,536,000 particle starts。
- Stage 0B: scientific variant 1、candidate 32 PF well-runs、
  4,096 seed-well、2,048,000 particle starts。
- 保存exp404 control rerun 0。
- LightGBM config / trained fold / booster / fitted model:
  `0 / 0 / 0 / 0`。
- HMM / Beam / GPU: `0 / 0 / 0`。
- `execution.selected_stage=null`、package/push/train approvalはfalse。

検証:

```bash
PYTHONPYCACHEPREFIX=/tmp/exp450-pycache \
NUMBA_CACHE_DIR=/tmp/exp450-numba-cache \
.venv/bin/pytest -q \
  experiments/exp450_dzdmd_conditioned_tvt_rate_likelihood_pf/tests/test_exp450_dzdmd_conditioned_tvt_rate_likelihood_pf.py

JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp450_dzdmd_conditioned_tvt_rate_likelihood_pf/\
exp450_dzdmd_conditioned_tvt_rate_likelihood_pf_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp450_dzdmd_conditioned_tvt_rate_likelihood_pf/\
exp450_dzdmd_conditioned_tvt_rate_likelihood_pf_compact_selfcontained_train.py

.venv/bin/python -m py_compile <compact train/inference sources>
.venv/bin/ruff check <compact train/inference sources and dedicated test> \
  --select F821,F401,E9
```

- dedicated testsはprefix OLS/fallback/backtest、first-row、residual-AR式、
  exp404 parent path、paired exact-coordinate parity、stable SHA seed、
  truth-late、fixed asset identity、実行量、inference guard、Notebook構造を
  対象にする。
- 合成parent pathのexp404との差は最大約`5.5e-12`で、固定`1e-10`以内。
- Kaggle PF、raw-test inference、submissionは実行していない。

最終結果:

- dedicated tests: `16 passed`。
- train/inference Jupytext変換と`--test`: PASS。
- `py_compile`: PASS。
- Ruff `F821,F401,E9`: PASS。
- Ruff format check: PASS。
- template / strict config / strict exp validation: PASS / PASS / PASS。
- `make update-summary`: PASS、454 experimentsを再生成。

全体回帰:

```bash
PYTHONPYCACHEPREFIX=/tmp/exp450-full-pycache \
NUMBA_CACHE_DIR=/tmp/exp450-full-numba-cache make test
```

- 1,619 testsを収集する前段で、exp450と無関係な既存5 module
  (`exp297`, `exp301`, `exp333`, `exp336`, `exp349`) が各自のconfig contract
  mismatch / missing keyでcollection errorとなり停止した。
- exp450専用16 testsはその前後で独立にPASSしており、上記5実験には変更を
  加えていない。

## 次のアクション

正規Notebook採用、Kaggle package作成、Stage 0A/0B push/runは別承認まで
停止する。Stage 1はStage 0A/0B全PASS後も別承認とする。

## 2026-07-30 実行承認

ユーザーの「実行してください」を、実装済みcompact trainの正規train
Notebookへの採用、Kaggle CPU package作成・push、Stage 0A全PASS時だけ
同一run内でStage 0Bを実行する承認として受領した。Stage 1、raw-test
inference、submissionは承認範囲外として無効のまま維持する。

push前の実行量再確認:

- scientific variant: 1
- Stage 0A: parent 12 + exact transform 12 = 24 PF well-runs、
  3,072 seed-well、1,536,000 particle starts
- Stage 0B: candidate 32 PF well-runs、4,096 seed-well、
  2,048,000 particle starts
- 保存exp404 control再実行: 0
- LightGBM config / fold / booster / fitted model: `0 / 0 / 0 / 0`
- HMM / Beam / GPU: `0 / 0 / 0`

正規train Notebookをcompact self-contained Jupytext sourceから生成した。
正規inference Notebookはguard scaffoldのままである。Kaggleの50文字slug
制限に合わせ、kernel IDは
`kentookumura/exp450-dzdmd-tvt-rate-likpf-train`とする。

初回pushはKaggle APIの`400 Bad Request`で受付前に失敗し、kernel/runが
作成されていないことを一覧で確認した。入力datasetとcompetition参照、
slug/title一致には問題がなかった。self-contained trainはrepository
`src/`をimportしないため、不要な`src/` bootstrapを除外してAPI payloadを
縮小してから再packageする。

## 2026-07-30 Kaggle version 1

- kernel: `kentookumura/exp450-dzdmd-tvt-rate-likpf-train`
- version / id_no / state: `1 / 129167787 / COMPLETE`
- Kaggle側metadata: private、CPU、internet disabled、
  dataset `kentookumura/exp404-v1-frozen-predictions`
- 初回400後のpackageは`--no-src`でAPI textを約1.15 MBから約0.79 MBへ
  縮小し、再pushに成功した
- log終端: 約299.639秒

Stage 0A結果:

- 24 PF well-runs、3,072 seed-well、1,536,000 particle startsを実行
- well count、実行量、finite、clip decision parityはPASS
- exact-coordinate parityとartifact readbackはFAIL
- 11 wellsはresampling decision mismatch 0。ただし最大seed prediction差
  `3.296009e-09 ft`、log-likelihood差`2.288914e-08`、
  temperature-5差`4.836693e-09 ft`で固定`1e-10`を超えた
- `5f4d2a52`だけ57 resampling decision mismatch。最大seed prediction差
  `21.176790850 ft`、particle weight差`0.039011922`、
  log-likelihood差`179.415276`、position差`36.715908902 ft`、
  rate差`0.163260907`
- Stage 0A `passed=false`、status
  `stage0a_exact_coordinate_parity_failed_closed`
- Stage 0Bは開始せず、scientific candidate PF well-runsは0

SHA:

- scientific contract:
  `136852c5ecc3cc6b97e928c32d42f88f6455103113c044811a770162296de355`
- parity report:
  `babe45f774f0657bc3b53e9e2685145e98430ed3ab29250ef81a1baa98c35261`
- paired prediction raw:
  `f49d827f576b80da78e65587f471a4472b5bffeb06ad339e05dc8b63b7581686`
- paired prediction decompressed:
  `a4b3f251dc931fda3fff4799c000caa103fae4fb5faa827dc57a7a38050fd33b`

判定:

数学的に等価なU-rateとTVT-rateの別演算順による小さい浮動小数差が、
長い実suffixでESS/resampling境界を跨いで1 wellの粒子系列を分岐させた。
科学candidateの性能は未評価であり、positive/negative scientific evidenceとは
扱わない。現行exp450はtechnical fail-closedとしてrun flagを再び無効化する。
threshold緩和、resampling同期、同じsentinel上の修正選択、rerun、Stage 0B/1、
inference、submissionは行わない。

## 2026-07-30 Version 2再開承認

ユーザーの「微小な丸め誤差なら次に進んでください」を受け、version 1の
temperature-5集約予測差最大`4.836692824e-09 ft`を実用上同一と扱う。
version 2のStage 0A hard gateは`<=1e-6 ft`（約`0.305 micrometer`）へ
変更し、seed prediction、particle weight、log-likelihood、position/rate、
resampling差は`diagnostic_checks_not_used_for_gate`として全値を保存する。
finite、clip decision、well/run/seed/particle count、truth-free、
artifact readbackはhard gateのまま維持する。

CSV readbackは`%.17g`で保存し、`float_precision="round_trip"`で読み戻す。
科学variant、OLS、PF設定、Stage 0B mechanism gateは禁止事項を含めて変更しない。

version 2実行予定:

- Stage 0A: parent 12 + exact transform 12 = 24 PF well-runs、
  3,072 seed-well、1,536,000 particle starts
- Stage 0B: scientific candidate 1 ×32 = 32 PF well-runs、
  4,096 seed-well、2,048,000 particle starts
- version 2最大合計: 56 PF well-runs、7,168 seed-well、
  3,584,000 particle starts
- 保存exp404 scientific control rerun: 0
- model / LightGBM config / fold / booster / HMM / Beam / GPU: 全て0
- Stage 1 / inference / submission: 無効

## 2026-07-30 Kaggle version 2

- version / id_no / state: `2 / 129167787 / ERROR`
- elapsed to error: 約`818.786 sec`
- 改訂Stage 0AはPASSし、Stage 0B candidate 32 PF well-runsの生成とfreezeまで進行
- 保存exp404 controlのraw/decompressed SHAはPASS
- source logical SHA照合で
  `ValueError: saved exp404 source logical SHA changed`
- 原因は、exp404が列名・dtype・numeric raw bytesで計算したtyped logical SHAを、
  exp450が17桁CSV文字列SHAで照合していた実装不一致
- 科学candidate、OLS、PF設定、mechanism gateの問題ではない

version 3ではexp404と同じtyped logical SHA関数とdtype正規化を移植する。
保存controlのexpected SHAは変更せず、科学設定・gateも変更しない。
version 2の実行量はStage 0A 24 + Stage 0B candidate 32 = 56 PF well-runs、
7,168 seed-well、3,584,000 particle starts。保存control PF rerunは0。

## 2026-07-30 Kaggle version 3

- kernel / id_no / state:
  `kentookumura/exp450-dzdmd-tvt-rate-likpf-train / 129167787 / COMPLETE`
- Kaggle側metadata: private、CPU、internet disabled、
  dataset `kentookumura/exp404-v1-frozen-predictions`
- runtime: `779.670823 sec`
- Python / NumPy / pandas: `3.12.13 / 2.0.2 / 2.3.3`
- peak RSS: `1.977406 GB`
- 実行量: Stage 0A 24 + Stage 0B 32 = 56 PF well-runs、
  7,168 seed-well、3,584,000 particle starts
- 保存exp404 control PF、model、booster、HMM、Beam、GPU rerun: 全て0

Stage 0A:

- 最終temperature-5集約予測をprimaryとする改訂gateをPASS
- 最大差`4.836692824e-09 ft <= 1e-6 ft`
- artifact readback、finite、clip decision、well/run countをPASS
- 57 resampling mismatches、最大seed prediction差`21.176790850 ft`、
  最大particle weight差`0.039011922`、最大log-likelihood差`179.415276`は
  事前改訂どおりdiagnosticとして保存
- parity report SHA:
  `84856a32a220719a1c4038841f00dfba02d149ba72339c6e8272c3d59e4303a1`
- paired prediction raw / decompressed SHA:
  `619d08043538cb269d64763fde630b211fae2de4b76983709fdab5ba3b21b895` /
  `cb1a7cd685c255a3ee0eb13a66c47beda8f30716e2bd5b51fab96a53c6a5af31`

Stage 0B:

- status: `stage0b_mechanism_failed_closed`
- 全16 gate中10 PASS、6 FAIL
- prefix tail20 backtest SSE ratio`0.241989`、非悪化`5/5 folds`でPASS
- persistent scope pooled RMSEは`12.785573 -> 12.462589 ft`、
  改善well`10/16`
- zero-directed under-response share削減`-0.004650`、
  forward-cause episode SSE削減`-0.057969`、
  persistent episode SSE削減`+0.013603`で3 gateをFAIL
- persistent改善fold`2/5`でFAIL
- matched control pooled / by-well p95 delta
  `+0.292528 / +1.678265 ft`で安全性2 gateをFAIL
- full runtime投影`11,906.624 sec`、finite、OLS/fallback、role/fold、
  truth-late、SHAはPASS
- fixed32全体`9.616741 -> 9.468335 ft`は記述値であり、CVではない

主要SHA:

- scientific contract:
  `e8a9f5abf42f654a925c002b3e7940f19c407c8fa2a379bbd1af5518605442fe`
- candidate logical / decompressed:
  `327c190c17ebc23b8568076e5b4a56b9d26b49538ed24821d7eb370c0d72ab03` /
  `97db5a745934ae8d98924676d91d8d8ca35f020762553fe68c92014caeb3ecad`
- prefix-fit logical:
  `ed47678864d149a4016617a48fd75346024d5f631a35c343f54b059f45e57881`
- saved exp404 source logical:
  `5f4b6e715081b598b0a34607ad0c81339d0ecd5882ea3a45dd79f33123959a00`

判定:

visible-prefixのaffine centerはprefix backtestでは有効だったが、unknown suffixの
under-response原因へ一貫して効かず、matched controlを大きく悪化させた。
事前固定の全AND条件どおりexp450をnegative resultとして終了する。Stage 1、
再実行、beta/intercept・window・support・PF設定の探索、well/row gate、
blend/selector、raw-test inference、submissionは行わない。run後のconfigは
`selected_stage=null`かつ全run/package/push/train approvalをfalseへ戻した。
