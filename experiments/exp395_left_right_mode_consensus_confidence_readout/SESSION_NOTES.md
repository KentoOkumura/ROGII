# exp395_left_right_mode_consensus_confidence_readout セッションノート

## 目的

同一のstable HMM mode lineageをheel側 / toe側のdisjoint GRで独立採点し、
左右mode posterior overlapがpersistent offset / large errorのconfidenceになるかを
train-sideで監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: exp391 Stage A1 FAILにより未実装のまま閉鎖
- CV: まだなし
- LB: まだなし
- implementation: 承認済み・前提gate待ちのため未実装
- Kaggle package / push / Stage 0 run: 承認済み・前提gate待ちのため未実行
- inference / submission: 無効

## コマンドログ

### 2026-07-25 実装承認・前提確認

- ユーザーの`exp395を実装してください`をimplementation-only承認として記録した。
- `kaggle kernels status kentookumura/exp391-prefix-anchored-mode-persistence-hmm-train`
  で、前提となるexp391 Stage A1が`RUNNING`であることを確認した。
- Kaggle kernelの`lastRunTime`は`2026-07-25 01:00:09.747000 UTC`。
- exp395の固定契約どおり、exp391 Stage A1全technical / mechanism gate PASS前には
  Notebook source、test、正規Notebookを実装しない。
- 今回の承認はKaggle package / push / Stage 0 run、full OOF、inference、
  submissionの承認には拡張しない。

### 2026-07-25 Stage 0実行承認・待機

- ユーザーの`実行してください`を、exp395実装後のKaggle private CPU
  package / push / Stage 0 run承認として記録した。
- 承認対象の実行量は1 diagnostic variant / 16 exact-HMM well runs /
  reporting 5 folds / LightGBM config 0 / trained fold 0 / booster 0 /
  PF 0 / Beam 0 / GPU 0 / parent-control rerun 0。
- 前提のexp391 Stage A1は開始から約1時間42分以降も
  `KernelWorkerStatus.RUNNING`。45秒間隔で8回確認しても実行中だった。
- exp391の同一canonical kernelを重複pushせず、exp395のNotebook実装、
  package、push、Stage 0 runも開始していない。
- full OOF 773 exact-HMM wells、inference、submissionは未承認のまま。

### 2026-07-25 design-only

- `task new-steering EXP=exp395_left_right_mode_consensus_confidence_readout`
  - `task` CLIが存在せず未実行。
- `make new-steering EXP=exp395_left_right_mode_consensus_confidence_readout`
  - steering scaffoldを作成。
- `make new-exp EXP=exp395_left_right_mode_consensus_confidence_readout`
  - template experiment scaffoldを作成。
- AGENTS、`kaggle-review-exp`、`kaggle-strategy`、
  `docs/agent-playbooks.md`、`docs/06_reproducibility.md`、
  exp386/387/391と現行backlogを確認。
- requirements / design / tasklist、config、README、SESSION_NOTES、
  result、metrics、backlogをdesign-only状態へ更新。
- `make validate-exp EXP=exp395_left_right_mode_consensus_confidence_readout`
  - strict validation PASS。
- `make update-summary`
  - `experiment_summary.md`へ親子edgeとdesign-only行を追加。

未実行:

- Notebook / helper / test実装
- Jupytext変換
- ローカルNotebook実行
- Kaggle package / push / train
- inference / submission
- 生成物作成

## 固定した設計

- 親: exp391 stable transition-overlap mode lineage
- decoder: exp209 exact HMM
- primary event: exp391 frozen 1,234 decoder-separation events
- left/right: 512 rows、checkpoint両側64-row gap
- primary confidence: mode posterior overlap
- primary risk: `1 - overlap`
- negative control: right GRの固定2048-row circular shift
- truth: confidence / null / SHA freeze後だけlate join
- prediction変更: 0

## 実行量

設計上のStage 0:

- diagnostic variant: 1
- fixed wells: 16
- exact-HMM well runs: 16
- LightGBM config / trained fold / booster: 0 / 0 / 0
- PF / Beam / GPU: 0 / 0 / 0
- parent/control rerun: 0

設計上のfull OOF:

- diagnostic variant: 1
- exact-HMM well runs: 773
- reporting folds: 5
- LightGBM config / trained fold / booster: 0 / 0 / 0
- PF / Beam / GPU: 0 / 0 / 0
- parent/control rerun: 0

どちらも未実行。historicalなStage 0実装・実行承認は、exp391 Stage A1 FAILにより
使用せず閉鎖した。full OOFも実行しない。

## 再現性メモ

- seed policy: RNGなし、fold / well / row / checkpoint / mode / directionをstable sort。
- stochastic components: なし。
- CPU/GPU runtime: CPU想定、GPUなし。実測なし。
- input / feature schema SHA: 未生成。
- mode ledger / confidence / null content SHA: 未生成。
- decoder/scientific contract SHA: 未生成。
- prediction SHA: predictionを生成しないため対象外。
- submission SHA: submission無効。
- deterministic anchor: false。rerun一致証拠なし。

## Design-only検証

- YAML / JSON parse: PASS
- strict experiment validation: PASS
- `experiment_summary.md` registration: PASS
- implementation / package / Stage 0 run approval: true。実行flagはfalse
- full OOF / inference / submission approval: false
- train / inference Notebook: templateの6-cell scaffoldのまま
- 新規Jupytext source / helper / test / Kaggle package: 0

## 次のアクション

1. 実装、Stage 0、full OOF、inference、submissionへ進まない。
2. exp391のthresholdやmode carrierを救済せず、未実装の閉鎖状態を維持する。

### 2026-07-25 exp391 Stage A1依存判定

exp391 Stage A1 version 3は`fail_closed`。HMM-supportedは1/19 events・1/5 foldsで、
必要な60%以上・4/5 foldsを満たさず、parity、normalization、projected runtimeも
FAILした。exp395の固定先行条件が不成立となったため、historicalな実装 / Stage 0
承認を使用せず、Notebook/test/package/Kaggle runを0のまま閉じた。
