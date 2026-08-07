# exp364_signed_curvature_exact_hmm セッションノート

## 目的

exp209 exact HMMへpersistent signed-curvature stateだけを追加する前に、GRが固定3軌道の
符号を識別できるかと、状態数3倍のresource上限をStage 0で判定する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_stage0_gate_failed_closed`
- CV / LB: なし
- compact self-contained train / fail-closed inference候補: 実装済み
- 正規train Notebook: compact self-contained候補を採用済み
- Stage 0 private Kaggle CPU: version 1完了、科学gate FAIL
- Stage 1 / inference / submission: 未実装・未実行、branch閉鎖

## コマンドログ

### 2026-07-23 設計

```bash
make new-steering EXP=exp364_signed_curvature_exact_hmm
make new-exp EXP=exp364_signed_curvature_exact_hmm
```

### 2026-07-25 Stage 0実装

- ユーザーの `exp364を実装してください` をStage 0実装承認として記録した。
- Jupytext percent形式でcompact self-contained train候補を実装し、compact `.ipynb`へ変換した。
- fail-closed inference候補を実装した。
- `config.yaml`の実行フラグはすべてfalseのまま。Kaggle実行は行っていない。

### 2026-07-25 Kaggle Stage 0実行承認

- ユーザーの「実行してください」を、compact self-contained train候補の正規Notebook採用と、
  private Kaggle CPU Stage 0 package / push / run承認として
  `2026-07-25 09:57:27 JST`に記録した。
- 実行対象はdiagnostic variant 1、fixed signed path 3、reporting fold 5、
  resource projection well 16。
- exact-HMM well-run 0、LightGBM config 0、trained fold 0、booster 0、
  parent control rerun 0。
- GPU / internetは無効。Stage 1 exact HMM、inference、blend、submissionは
  引き続き未実装・未承認である。
- canonical kernelは
  `kentookumura/exp364-signed-curvature-exact-hmm-train`
  (`exp364 signed curvature exact hmm train`) とする。
- push前のcanonical照会は`Not found`。同IDの既存kernelはない。
- 正規train Notebookはcompact候補と同一SHA
  `109f54a611db2fa04f147e09bae7d3ca2a15d85df4db114e51d6dac13d2bf698`
  で採用した。
- packageはcanonical 22 cellsへsupport bootstrap 1 cellを加えた23 cells。
  bootstrap 24 filesを監査し、loose / package / bootstrapの`config.yaml` SHAは
  `84cf014d2b0de5aac26c4f42ec00595e61c4f9494f081785ad21024defca04d5`
  で一致した。
- compact train source SHAは
  `1ebbbba3374f4dd62157126c365ba2c36c24a1f9416220a05b4031cbcbf6423d`、
  package Notebook SHAは
  `78d746fa985091bc26af93cb920a61207c117a593256e9a2cbc1eb0a79b880b3`、
  `kernel-metadata.json` SHAは
  `1f42b303839c8699373488b103351c5d512283211143806f323d17aa3595cac4`。
- metadataはprivate CPU、GPU / TPU / internet off、run-on-push true、
  competition source 1、exp115 hidden-like kernel source 1である。

### Kaggle version 1

- canonical kernel version 1、id_no `128529795`をpushし、status `COMPLETE`を確認した。
- Stage 0本体は`224.73707962036133 sec`、773 wells発見、772 wells /
  3,783,582 candidate rows / 13,631 complete blocksを評価した。
- freeze前truth row / hidden-like role rowは`0 / 0`。candidate / block score /
  resource projectionのSHA readback、unique key、complete-block、16-well extrema、
  exact-HMM run 0、Stage 1 disabledを含むtechnical gateは`12 / 12 PASS`。
- overall top1 `0.550143`、zero-first比MRR gain `0.252574`はPASSした。
  1000+ / hidden-like spatial / hidden-like typewell-purgedのselected-path RMSE gainも
  `+2.027732 / +1.466085 / +1.459314 ft`で全て正方向だった。
- real-minus-circular top1は`0.003081 < 0.03`、passing foldは`3 / 5 < 4 / 5`でFAIL。
  fold 1と2のreal-minus-circular top1は負だった。
- 16-well projectionはpeak RSS `4.880433 GB <= 25 GB`をPASSしたが、
  固定runtime `33857.604 sec > 30600 sec`でFAILした。
- technical PASS / scientific FAIL / Stage 0 FAIL。
  判定を`STAGE0_FAIL_CLOSE_WITHOUT_RESCUE`としてStage 1へ進まない。
- Stage 1 exact-HMM well-run、LightGBM config、trained fold、booster、parent control
  rerun、inference、submissionはいずれも0。

### output監査

- Kaggle output archiveは丸ごと取得せず、Stage 0監査成果物10件だけを
  `/tmp/exp364_stage0_output_v1`へ取得した。
- freeze manifestのraw SHAと、gzip 3件の展開後content SHAをローカルで再計算し、
  summary記録値と一致した。
- candidate path decompressed content SHA:
  `64a8c744ec99730905032f950875f76c61c23df7504a0731592abc6101c63cbb`
- block GR score decompressed content SHA:
  `40bdad70b46670b6acd01b26baf8b319bbcfdb302899a6e7787145ce20955048`
- postfreeze readout decompressed content SHA:
  `4d1de039fce56ba4c0cff63dfbda84f92b74c467c9ea55c3a4b337e9dca32618`
- resource projection / freeze manifest / gate report / summary SHA:
  `cef21a2f44137df5a4b98520d80f6273045dd6aff2bea534194171ee393af4a9` /
  `9c0d27a304ceff3778a4e5b69ae9d56224c1932ecbd64050c9327a7cc9ad2feb` /
  `52e1c3d4e4e46b67ba6e1ddbf1a39d22e421c76b72121b6e0ca2ca1ded2fb741` /
  `75f32f0fef65f1af9ae8f8e58ccd1ae05a471f518acecf5653e0dd312df73541`。

## 実装した固定契約

- 変更変数は`c=-1/0/+1`だけ。driftは`c * 0.005 / 512`/row、遷移と初期確率は
  design-frozen値。
- exp209のposition step `0.35`、41 rate cells、momentum `0.998`、Gaussian GR emission、
  prefix sigma、terminal prefix rate、band pad `100 ft`を固定した。
- complete 512-row blockのみ、stride 256、score tieは`[0,-1,+1]`。
- negative controlはwithin-well GR blockを1 block circular shiftし、single-block wellは
  256 rows shiftする。
- foldはstable well orderのGroupKFold 5分割。fold passはtop1 gain、MRR gain、
  real-minus-circular top1がすべて正。
- 1000+ / hidden-like spatial / hidden-like typewell-purgedは、GR top1 pathのzero path比
  pooled RMSE gainが正であることを要求する。
- input identity、candidate path、GR score、16-well resource projectionを保存・SHA再読込後に
  suffix truthとhidden-like roleを読む。

## Resource projection

- 16 wellsはparent state-cell workloadのstable quantileから選び、最小・最大workloadを必ず含む。
- runtimeは採用済みexp209 v5 HMM wall time `11285.868 sec`を3 curvature statesで
  `3.0x`する固定projection。設定上のprojected runtimeは`33857.604 sec`で、
  上限`30600 sec`より大きい。これはStage 0実行時のhard scientific gateであり、
  値を見てcurvature/transition/runtime式を調整しない。
- peak RSSは選択wellの実position/rate/suffix shapeからalpha/posterior/emission/workspaceを
  算出し、1 GB fixed overheadと1.25 safety factorを加えた結果`4.880433 GB`だった。
- Stage 0実行量: diagnostic 1 / fixed paths 3 / reporting folds 5 /
  resource projection wells 16 / exact HMM well-runs 0 / LightGBM config 0 /
  trained fold 0 / booster 0 / parent control rerun 0。
- 条件付きStage 1予約: 1 variant / 773 exact-HMM well-runs / booster 0 /
  parent control rerun 0。未実装・未承認。

## Notebook構成比較

- 親exp209にはcompact self-contained版がないため直接比較は非該当。
- 同じsigned-path Stage 0を持つexp367 compact trainを構成参照にした。
- exp364 train候補は10章、約1,700行で、exp367の候補生成・freeze・late join・metricsに加え、
  exp209固定観測契約と16-well resource projectionをNotebook上へ展開した。
- 同一exp helper importと`__file__`は使用していない。

## 静的検証

- `py_compile`: train / inferenceともpass。
- `ruff --select F821`: train / inferenceともpass。
- `pytest -q tests/test_exp364_signed_curvature_exact_hmm.py`: 9 passed。
- `jupytext --to ipynb --test`: train / inferenceともpass。
- `make validate-exp EXP=exp364_signed_curvature_exact_hmm`: strict pass。
- `make validate-template`: pass。
- `make test`: exp364を含む`980 passed / 6 skipped`、既存exp296契約test 2件だけFAIL。
  exp296の実行後status / run flagと旧test期待値の不一致であり、exp364変更対象外。

### 実装SHA

- compact train source:
  `1ebbbba3374f4dd62157126c365ba2c36c24a1f9416220a05b4031cbcbf6423d`
- compact train Notebook:
  `109f54a611db2fa04f147e09bae7d3ca2a15d85df4db114e51d6dac13d2bf698`
- fail-closed inference source:
  `d1e2280c5e50c64b0726eb55ee0f852f4fb959823c7f48c84d4be6e3e53bfdac`
- fail-closed inference Notebook:
  `5c7ff35c7e534992a888d803302f89f9bc4ac04cf69d9d0a66eaea700c45056f`
- dedicated test:
  `d12050826ea9ea6175bf74a54164e4c9b6a86d64664446439c6711561c75863a`
- config:
  `156cc9060c796329ce3571191c665ced05dc102bb67587b397b69523186182de`

## 再現性メモ

- seed policy: RNGなし、well / row / stateのstable order。
- stochastic components: なし。
- CPU/GPU: Kaggle CPU single worker、GPU off。
- gzipはdecompressed content SHAを主証拠とする。
- kernel version、candidate / score / resource / gate / summary SHAを記録済み。
- prediction / model / submissionは未生成。
- `deterministic_anchor`はtrue。

## 次のアクション

1. exp364はbranchを閉じ、Stage 1 exact HMMを実装・実行しない。
2. curvature magnitude/persistence、emission、sigma、adaptive noise、parallelism、
   blendで救済しない。
3. 将来のpath-ranking preflightでは、exp364再昇格と切り離し、複数の十分離れた
   circular nullがnegative-controlとして機能するかだけを独立0-HMMで監査する。
