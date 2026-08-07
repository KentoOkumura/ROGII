# exp432_symmetric_datum_defensive_particle_reinjection

## 状態

- ルート: `pf_beam`
- 状態: Kaggle Stage 0完了、`stage0_fail_closed`
- CV / LB / Submit ID: なし
- 作成日: 2026-07-28
- PF 親: exp404
- 原因/trigger根拠: exp410 / exp412

## 仮説

最初の persistent HMM rate-gap eventで方向を使わず、base 80%、`-datum` 10%、`+datum` 10%のimportance-corrected proposalを一度だけ使えば、元PF targetを変えずに有限粒子supportを回復できる。

## 固定した設計

- triggerはexp412と同じだが、beta方向は完全に捨てる。
- eventは各well最初のinactive→active row、一回だけ。
- `datum=max(filtered HMM position std, 0.35 ft)`。
- `q=0.8p0+0.1p-+0.1p+`、weightへ`p0/q`を反映、clipなし。
- event後はparent PFへ戻り、branch labelはancestry監査だけに使う。
- PFはx1.0、500 particles、128 seeds、Gaussian evidence T=5。

数式、RNG、gateは [steering design](../../.steering/20260728-exp432-symmetric-datum-defensive-particle-reinjection/design.md) を正とする。

## Stage 0実行量

fixed32でunchanged HMM 32 well-runs、baseline PF 32、treatment PF 32、合計PF 64、8,192 seed-well trajectories、4,096,000 particle starts。親PF control再実行を含むため、実装後のKaggle pushにも別途明示承認が必要。

## full実行量

Stage 0全PASS後のみ、HMM trigger cache 773 well-runsとtreatment PF 773 well-runs、98,944 seed-well trajectories、49,472,000 particle startsを別artifact/4 PF shardで実行する。保存済み親PFの独立full rerunは0。

## 検証方針

fixed32 Stage 0でproposal density、`p0/q`、RNG分離、no-event parityを先に検証する。全artifact freeze後にtruthをjoinし、trigger後512 rowsのparticle support外率、SSE、fold、matched control安全性をAND gateで判定する。Stage 0はCVではない。

## 実装

- Jupytext source:
  `exp432_symmetric_datum_defensive_particle_reinjection_compact_selfcontained_train.py`
- compact Notebook（正規trainへ採用済み）:
  `exp432_symmetric_datum_defensive_particle_reinjection_compact_selfcontained_train.ipynb`
- 専用contract test:
  `tests/test_exp432_symmetric_datum_defensive_particle_reinjection.py`
- exp209 first-pass HMM、exp412 event、exp404 baseline/treatment PF、truth-late
  readout、technical/mechanism AND gateをnotebook内に実装した。
- `p0/q`は通常の比へ実体化せず、finiteな`log p0 - log q`をlog-weightへ加える。
  `datum >= 0.35 ft`とposition noise `0.005 ft`の組合せで起きるfloat64
  underflowを回避し、clipは行わない。

## 実行入口

2026-07-29の実行承認によりcompact候補を正規train Notebookへ採用し、Stage 0を
canonical Kaggle kernel
`kentookumura/exp432-symmetric-defensive-reinjection-train`で実行した。
full、inference、submissionは実施せず、Stage 0再実行もロックした。

## 結果

Kaggle private CPU version 1（id_no `128974856`）を完了した。triggered-window
SSEは`12.0871%`改善したが、support外率は`0.4278 points`悪化し、nonworse foldは
`3/5`、worst controlは`+0.583073 ft`だった。mechanism AND gateを満たさず
`stage0_fail_closed`。これはfixed32 mechanism preflightでありCVではない。

## Artifact

Kaggle outputとしてtarget-free prediction / trigger schedule / truth-late rows /
well metrics / SHA ledgerを生成した。主要SHAは`metrics.json`と
`SESSION_NOTES.md`に記録した。実ファイル確認が不要なためoutput archiveは
ダウンロードしていない。

## 所見

方向推定の再利用ではなく、event時刻だけを使う対称proposalである。importance correctionを外すと元targetを変えるため、実装時の最優先contractとする。

## 次

現行branchを閉じる。fixed32上の救済、full、inference、submissionは行わない。
