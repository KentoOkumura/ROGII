# exp439_continuous_kinematic_joint_transition_exact_hmm セッションノート

## 目的

exp209のrate履歴を維持しながら、TVT/rateを台形積分したcorrelated joint edgeで
結合し、position latticeの確率、一次・二次moment、`Cov(delta_TVT, delta_r)`を
保存する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_stage0_technical_failed_closed_moment_projection_infeasible`
- 実装承認: 2026-07-29のユーザー依頼「exp439を実装してください」
- 正規Notebook採用: 2026-07-29のユーザー依頼「実行してください」で承認
- Kaggle package / push / Stage 0 run: version 1完了、technical fail-close
- Stage 1 / inference / submission: 対象外、再実行も無効
- CV / LB: なし

## 2026-07-29 実装セッション

### 実装内容

- Jupytext percent形式のcompact self-contained train候補を実装した。
- exp209のinput preparation、rate marginal、state、prior、emission、grid、
  noise、readout、rate/position境界semanticsを固定した。
- 各`source_rate -> destination_rate` edgeを
  `0.5*(r_source+r_destination)*delta_MD-delta_Z`で作った。
- 5/7/9セルを固定順で探索する非負maximum-entropy moment projectionを実装した。
- 全supportでtarget mean/varianceがlattice convex hull外ならfail-closeする。
- joint edge tableを1回precomputeし、forward/backwardの両方へ同じ配列を渡した。
- rate marginal、edge weight sum、mean、variance、source-row covariance、
  exp209 grid biasをtruth-free auditへ保存する。
- exhaustive small-path referenceとjoint forward/backward parityを実装した。
- fixed32の全prediction/diagnostic/SHAをfreezeするまでtruth、role/fold、
  persistent episode/causeを読まないguardを維持した。
- inferenceはfail-closed候補だけを実装した。
- 既存の正規Notebook placeholderは上書きせず、compact候補Notebookを別名で生成した。

### 予定実行量

- scientific variant: 1
- Stage 0 candidate HMM well-runs: 32
- Stage 1 candidate HMM well-runs: 最大773、未承認
- parent exp209 HMM rerun: 0
- fitted ML model / LightGBM config / trained ML fold / booster:
  `0 / 0 / 0 / 0`
- PF / Beam / GPU: `0 / 0 / 0`

Kaggle push前に同じ数を再確認する。control再学習は含まれない。

### 検証コマンド

```text
.venv/bin/pytest -q experiments/exp439_continuous_kinematic_joint_transition_exact_hmm/tests/test_exp439_continuous_kinematic_joint_transition_exact_hmm.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact source>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact source>
.venv/bin/python -m py_compile <train> <inference> <test>
.venv/bin/ruff check <train> <inference> <test>
make validate-exp EXP=exp439_continuous_kinematic_joint_transition_exact_hmm
```

結果:

- contract test: `12 passed`
- py_compile: PASS
- Ruff: PASS
- Jupytext train/inference変換と`--test`: PASS
- strict experiment validation: PASS
- `task validate-exp`は環境に`task`がなく実行不能だったため、規定どおり
  `make validate-exp`へ切り替えた。

### 親compactとの比較

- exp438 compact train: 2,780行、9章
- exp439 compact train: 3,000行超、9章
- exp439にはinput/SHA/truth-late/orchestrationに加え、moment solver、
  correlated joint-edge table、共有forward/backward、edge audit、
  exhaustive path referenceをNotebook上へ展開した。
- 同じ実験配下のhelper import、`__file__`、薄い`main()`呼び出しは使っていない。

## 再現性

- RNGなし。well、row、source rate、destination rate、position offset順を固定。
- moment solverはfloat64、support順、初期値、反復上限、tol、dampingをconfig固定。
- HMM messageは親どおりfloat32、posterior/auditはfloat64。
- fixed32 manifest、episode ledger、保存exp209 SHAをconfig固定。
- joint-edge table、moment audit、prediction、rate readout、metricsにlogical SHAを保存する。
- first runはdeterministic anchorとせず、rerun SHA一致後だけ再判定する。

## 注意

0.35 ft latticeと固定`effective sigma=0.1225 ft`では、平均が半セル付近のedgeは
指定varianceがlattice上の最小分散を下回り得る。この場合は設計どおり5/7/9の
すべてを不可能と判定し、noise/grid/support規則を変更せずfail-closeする。
これは実装上のfallbackではなく、事前登録したtechnical contractである。

## 実行前に必要だった承認

Stage 0実行前は、正規Notebook採用とKaggle package/push/runを別途承認対象とした。
承認後もfixed32の32 candidate HMM well-runsだけを先に実行し、全AND gateを
満たさなければStage 1へ進まない契約とした。

## 2026-07-29 Stage 0 実行承認

ユーザー依頼「実行してください」を、正規Notebook採用、canonical Kaggle CPU
package、push、Stage 0 fixed32 runの承認として記録した。

実行量を再確認:

- scientific variant: 1
- candidate HMM well-runs: 32
- parent control HMM rerun: 0（保存exp209 predictionを使用）
- reporting folds: 5
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- fitted ML model / PF / Beam / GPU: `0 / 0 / 0 / 0`

Stage 1、inference、submissionは未承認のまま維持する。

### Kaggle package / first push

- canonical train / inference notebookをcompact self-contained版へ採用した。
- strict packageは`private / CPU / internet off / run_on_push / --no-src`。
- 初回の60文字slug
  `exp439-continuous-kinematic-joint-transition-exact-hmm-train`は
  Kaggle SaveKernel APIが`400 Bad Request`で拒否し、runは開始しなかった。
- API応答に詳細理由はなかったが、このリポジトリの稼働済みmetadataが50文字以下へ
  短縮されていることから、50文字制約への抵触と判断した。
- 仮説・実行内容を変えず、43文字の
  `exp439-continuous-kinematic-joint-hmm-train`へ短縮して再packageする。
- 短縮slugのKaggle kernel version 1はpush成功し、Stage 0を開始した。

## 2026-07-29 Stage 0 terminal result

- kernel: `kentookumura/exp439-continuous-kinematic-joint-hmm-train`
- version / id_no: `1 / 129058811`
- terminal: `KernelWorkerStatus.ERROR`
- failure検出: 約`33.181 sec`
- private / CPU / internet off

最初のtarget-free scope wellはwell昇順の`060ab2b8`。そのrow 0、
`source_rate=0`、`destination_rate=0`、
`mean_shift=-0.11000000000021828 ft`で
`moment projection infeasible`を発生させ、事前登録どおりfail-closeした。

固定lattice step `0.35 ft`、effective sigma `0.1225 ft`なので、
target varianceは`0.015006249999999999 ft^2`。平均を挟むlattice点
`-0.35 / 0.0 ft`で実現可能な最小分散は
`0.026400000000028373 ft^2`で、targetは
`0.011393750000028374 ft^2`不足する。supportを5/7/9へ広げても近傍の
lattice間隔は変わらず、非負確率でmean/varianceを同時保存できない。

判定:

- technical gate: FAIL
- failed contract: `nonnegative_lattice_moment_feasibility`
- scientific / mechanism gate: 未評価
- candidate HMM well-runs: 予定32、完了0
- attempted wells: 1
- prediction / joint-edge artifact / truth-late metrics: 未生成
- truth / role / fold / episode pre-freeze reads: 0
- deterministic anchor: なし
- Stage 1 eligible: false
- inference / submission: なし

これはsolver収束、bootstrap、入力、Kaggle packageの不具合ではなく、実装前に
明記したrepresentation contractの実データ上の不成立である。support、moment、
noise、grid、rate、emission、prior、gateをこのexpで救済せず、再実行、
Stage 1、inference、submissionなしでbranchを閉じる。
