# exp381_formation_contact_order_semimarkov_hmm セッションノート

## 目的

地層接触の位置、contact-TVT、順序がsemi-Markov priorとして成立するかを、
HMMを1回も動かさずに先行監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU Stage 0完了・固定AND gate FAIL・branch closed
- CV / LB: Stage 0診断のみ。CV / LB / submissionなし
- Notebook: compact trainを正規trainへ採用。fail-closed inference候補と正規inference scaffoldは未採用
- Kaggle package / push / Stage 0 run: private CPU version 2で完了
- inference / submission: 未承認・未実施
- Stage 1: Stage 0不合格により未実装・0 runでfail closed
- package preflight: private CPU / internet off / run-on-push / exp226 input、
  bootstrap 22 files、path safety、bytes / SHA全一致を確認
- 同一kernel slug: Kaggle kernel listで`Not found`を確認

## コマンドログ

2026-07-24:

```bash
.venv/bin/python -m py_compile experiments/exp381_formation_contact_order_semimarkov_hmm/exp381_formation_contact_order_semimarkov_hmm_compact_selfcontained_train.py experiments/exp381_formation_contact_order_semimarkov_hmm/exp381_formation_contact_order_semimarkov_hmm_compact_selfcontained_inference.py
.venv/bin/ruff check experiments/exp381_formation_contact_order_semimarkov_hmm/exp381_formation_contact_order_semimarkov_hmm_compact_selfcontained_train.py experiments/exp381_formation_contact_order_semimarkov_hmm/exp381_formation_contact_order_semimarkov_hmm_compact_selfcontained_inference.py tests/test_exp381_formation_contact_order_semimarkov_hmm.py
.venv/bin/pytest -q tests/test_exp381_formation_contact_order_semimarkov_hmm.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp381_formation_contact_order_semimarkov_hmm/exp381_formation_contact_order_semimarkov_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp381_formation_contact_order_semimarkov_hmm/exp381_formation_contact_order_semimarkov_hmm_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp381_formation_contact_order_semimarkov_hmm/exp381_formation_contact_order_semimarkov_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp381_formation_contact_order_semimarkov_hmm/exp381_formation_contact_order_semimarkov_hmm_compact_selfcontained_inference.py
make validate-exp EXP=exp381_formation_contact_order_semimarkov_hmm
```

`task`は環境に存在しなかったため、スキル記載の同等手順`make validate-exp`を使用した。
初期実装時点ではローカルnotebook実行、Stage 0の実データfull run、
Kaggle package生成を行っていなかった。その後の実データfull runはKaggleのみで実施した。

同日、ユーザーの「実行してください」により、compact trainの正規trainへの採用と
Kaggle private CPU Stage 0 version 1を1回実行することを承認済みとして記録した。
科学的実行量は1 diagnostic / 6 reporting surfaces / 5 reporting folds、
model / HMM / PF / Beam / LightGBM booster各0、parent control再実行0、GPU 0。
Stage 1実装・実行、正規inference採用、inference、submissionは承認対象外である。

push直前preflightのtrain source SHA256は
`99a3cda6846a7ba657bc5ba7d76b316ef45c052b80f4e84e86bea3c638c0b56d`。
push対象package notebook SHA256は
`294020a8cc8b55c0683aeab285c04e6ffebfd187d9ba484ef17612536a04f5cf`。
1回実行承認は`2026-07-24T10:04:32Z`にconsumedとして記録した。
Kaggle private CPU version 1は`2026-07-24T10:06:18Z`にpush済み。
kernel id_noは`128461656`で、remote metadataでもGPU / internet無効と
exp226 kernel sourceを確認した。

version 1は約`43.463 sec`でouter-train source読込中にERRORとなり、
surface fit、target-free crossing、truth late join、固定gateには未到達だった。
原因は6 Formation列を全坑井で有限と誤って要求したこと。実データはANCCが
7/773 wells、EGFDLが1/773 wellsで列全体欠損、部分欠損は0だった。
修正版はMD/X/Y/Z/TVTの有限契約を維持し、Formationごとに有限なouter-train
wellだけから固定k=10 donorを選ぶ。k、surface式、crossing、offset、gate、
実行量は変更していない。欠損donor専用testを追加し、専用testは10件となった。
version 2のtrain source SHA256は
`09b8632baadf316dd4e298c323d47c084af45568691be915628777de8c4b53e7`、
push package notebook SHA256は
`a12e578e96369cec68c5ea61672a3bb9ea1f81f0d29d8b0d6263eb8f6b06eaca`。
version 2は`2026-07-24T10:13:54Z`にpushし、初回statusは`RUNNING`。
`2026-07-24T10:24:41.970116Z`にStage 0を完了し、notebook log時刻は
`653.714 sec`。Kaggle statusは`COMPLETE`、scientific decisionは
`stage0_failed_close_without_semimarkov_hmm`だった。

## Stage 0結果

| 指標 | 値 | gate |
| --- | ---: | --- |
| eligible wells | 349 / 773 (`0.451488`) | `>=0.25` PASS |
| contact events | 1,291 | `>=1000` PASS |
| crossing MD MAE | `35.994405 ft` | `<=128` PASS |
| crossing MD p90 | `61.799226 ft` | `<=512` PASS |
| contact-TVT RMSE | `44.770101 ft` | `<=15` **FAIL** |
| correct order率 | `0.997135` | `>=0.95` PASS |
| positive folds | 5 / 5 | `>=4` PASS |
| constant surface比gain | `687.676085 ft` | `>=0.10` PASS |

全AND gateはcontact-TVT RMSEだけが不合格。位置と順序は移送できたが、
outer-train contact centerをtarget known-prefix単一offsetで校正する固定式は
TVT制約として必要な精度へ届かない。surface k、formation除外、offset、gateを
実行後に救済せず、semi-Markov HMM Stage 1を実装・実行しない。

fold別contact-TVT RMSEは`50.3553 / 43.0637 / 40.2907 / 50.7324 / 38.4834 ft`で、
全foldが15 ftを超えた。formation別はANCC `36.4549`、ASTNU `55.1462`、
ASTNL `38.2471`、EGFDU `40.2478`、EGFDL `44.5037`、BUDA `225.3564 ft`
（BUDAは2 events）だった。

## 実装した固定契約

- 保存済みexp226 OOFの`well_id/fold/row_idx/suffix_offset`だけをpre-freezeで読む。
- outer-train well-median参照の`FormationPlaneKNN(k=10)`を主surfaceとする。
- outer-train formation中央値のconstant surfaceをpaired controlとする。
- 各formationはfull-wellのMD昇順first crossingだけを線形補間する。
- outer-train真接触のformation別contact-TVT中央値を作る。
- targetは`TVT_input + Z - F_hat - contact_center`の有限中央値1個だけで校正する。
- target-safe surface/crossing/contact prediction/resourceをSHA freezeしてから、
  outer-validの`TVT`と6 Formationをlate joinする。
- plane / constant / truthのtriple-matched formationが2個以上の坑井だけをeligibleとする。
- eligible率、event数、MD MAE/p90、contact-TVT RMSE、order率、fold改善、
  constant比gainを固定AND gateで判定する。
- Stage 0がPASSしてもStage 1を自動実装・実行しない。
- canonical train slugはKaggle 50文字制限内の
  `exp381-formation-contact-order-semimarkov-train`（47文字）へ固定する。

## 実行量

- scientific diagnostic: 1
- reporting surfaces: 6
- outer reporting folds: 5
- fitted model / model config / trained fold: `0 / 0 / 0`
- HMM / PF / Beam / LightGBM booster: `0 / 0 / 0 / 0`
- parent/control再実行: 0
- GPU: 0。Kaggle private CPU
- resource readout: SHA256順で事前固定した16坑井、report-only

## 再現性メモ

- seed policy: RNGなし、fold / well / formation / MD順を固定
- stochastic components: なし
- CPU/GPU runtime: CPU、GPUなし
- input / fold / surface reference / contact center / crossing / resource SHA:
  Stage 0実行時に保存
- gzip: raw gzip SHAに加え、decompressed content SHAとlogical content SHAを保存
- model manifest: fitted modelなし。Stage 0 solver/contact契約をfreeze manifestに保存
- prediction SHA: target-free crossingとlate-joined contact readoutを保存
- submission SHA: inference/submission禁止のため非該当
- deterministic anchor: 初回runでは主張しない
- target-free bundle logical SHA:
  `69787c31c0857128939e8b1dc23034892767f4d700d99cc73b3eacf82d4b6955`
- truth manifest logical SHA:
  `b27a1c42d177eff0ad5228168eef619f024323800aca8d2f6144e9b848e0cf2c`
- contact event logical SHA:
  `c967ee577830e9f3377c9469cb93736749a87480682f1eb5552c6fb212712c11`
- 期待15 artifactが存在。SHA manifest 14行のraw / decompressed SHAは全一致。
- pre-freeze outer-valid 773/773 wellsは`MD/X/Y/Z/TVT_input`のみ、
  truth / Formation hit 0。late truth joinは773/773 wellsでfreeze後。
- resource 16 wells: 773-well投影`172.987 sec`、max RSS `0.591427 GB`、
  report-only。

## 検証

- 専用contract test: `10 passed`
- Ruff: PASS
- py_compile: train / inference PASS
- Jupytext compact train / inference生成: PASS
- Jupytext train / inference round-trip: PASS
- `make validate-exp`: strict PASS
- trainは10章、1,927行で、surface、crossing、freeze、truth late join、gateを
  Notebookセル内で追える。`__file__`や同一exp helper importに依存しない。
- 構成参照元exp377 compact trainは9章、2,535行。exp381はK16/HMMを持ち込まず、
  Stage 0 contact readoutに必要な役割だけを10章へ展開した。
- 全repository testは`876 passed / 6 skipped / 3 failed`。初期exp381専用9件と
  notebook/scaffold系は全PASS。FAILは未変更の既存状態であるexp296の
  status/run flag期待2件と、exp384のexecution authorization期待1件のみ。
- version 1修正後のexp381専用10件、Ruff、py_compile、Jupytext round-trip、
  strict experiment validationは再度PASS。

## 次のアクション

1. exp381 branchはStage 0不合格として閉じる。
2. 同一surface / prefix-offset / gateの救済、Stage 1、inference、submissionは行わない。
3. 位置精度と順序精度は独立した知見として保持するが、contact-TVT priorへの昇格根拠にはしない。
