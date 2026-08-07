# 要件

## 初回設計依頼

exp408で確認したtranslation-gauge lockを直接扱う新実験として、exp209 exact HMMに
rateとは独立したabsolute datum branchを追加する設計を確定する。

初回の依頼範囲は次に限定した。

- `exp425_symmetric_datum_reanchor_exact_hmm`を新規実験として登録する。
- backlog、steering、実験ディレクトリ、設計config、実験記録を作成する。
- 仮説、変更点、固定条件、Stage 0、成功条件、停止条件、再現性契約を事前登録する。
- HMMコード、Jupytext source、Notebook実装、test、Kaggle package / push / runは作成しない。

## 2026-07-28 実装依頼

ユーザーの
`exp425_symmetric_datum_reanchor_exact_hmm です`
という対象確定を受け、次を追加で承認済みとする。

- compact self-contained Jupytext train / inference sourceを実装する。
- unchanged exp209 first pass、最初のpersistent event、対称3 datum枝のexact
  sum-product marginalization、truth-late readout、Stage 0 gateを実装する。
- parent parity、parent-only branch parity、single-event、soft branch、
  truth-late freeze、inference fail-closeの専用testを実装する。
- Jupytext sourceから正規train / inference Notebookを採用する。

次は今回も未承認のままとする。

- Kaggle package / push / Stage 0 run
- Stage 1実装・実行
- current-test inference / submission

## 科学要件

- 親は`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`とする。
- exp412で確認したsmoothed-versus-filtered rate disagreementは、修正の
  `activation`にだけ使う。
- exp412のrate修正符号をdatum shift符号へ直接転用しない。
- 最初のpersistent activation eventで、次の3枝を同時に生成する。
  - parent datumを維持する枝
  - absolute datumを正方向へreanchorする枝
  - absolute datumを負方向へreanchorする枝
- 3枝は将来GRを含むbackward尤度でsoftに選択し、truth / error / oracleで枝を選ばない。
- rate grid、rate transition、position noise、GR emission、initial prior、
  state support、posterior-mean readoutは親から変更しない。
- 同一OOF上でtrigger、shift scale、branch prior、gateを探索しない。

## 制約

- Route: `pf_beam`
- Phase / priority: late / P3 high-risk mechanism experiment
- 再現性: `docs/06_reproducibility.md`に従い、first-pass message、
  activation event、datum shift、branch posterior、prediction、metricsのSHAを記録する。
- RNGは使わず、well、row、position、rate、branchの走査順を固定する。
- Stage 0 fixed32はerror-selected mechanism sampleであり、CV、full OOF、
  promotion evidenceとは呼ばない。
- Stage 0実装と正規Notebook採用は2026-07-28のユーザー依頼で承認済み。
- Kaggle実行、Stage 1、inference、submissionはそれぞれ別の明示承認なしに行わない。
- exp412はnegative resultのまま維持し、本実験で再分類しない。

## 受け入れ基準

- steering 3文書に仮説、exact branch semantics、固定パラメータ、
  Stage 0 gate、runtime fail-close、禁止事項が記載されている。
- 初回scaffoldの`config.yaml`にroute、lineage、design-only状態、planned
  well-runs / branch state数 / model・booster数が記載されている。
- 初回の`SESSION_NOTES.md`、`result.md`、`metrics.json`が未実装・未実行状態と
  一致していた。
- `KAGGLE_DIRECTION.md`の未着手backlogと`experiment_summary.md`へ登録されている。
- deterministic submission anchorとは扱わない。
- gzip生成物を将来比較する場合はraw gzip SHAではなくdecompressed content SHAを
  主証拠として記録する。

## 実装受け入れ基準

- first passのzero position-shiftが独立exp209実装と固定tolerance内で一致する。
- eventはpersistent activationの最初のfalse→true rowだけで、rate-gap符号を
  datum方向へ使わない。
- `negative / parent / positive` conditional exact-HMMを固定priorとfull-sequence
  evidenceでsoft周辺化し、hard branch selectionを行わない。
- eventがない場合はparent predictionとbranch mass `[0, 1, 0]`を厳密に返す。
- 全32 wellsのevent、shift、branch posterior、prediction SHAがfreezeするまで
  truth / cause episodeを読めない。
- train / inference sourceはnotebook-safeかつself-containedで、`__file__`や
  同一実験helper importに依存しない。
- configは実装済み・未実行、Stage 0 run false、model / booster / PF / Beam / GPU
  すべて0の状態と一致する。
