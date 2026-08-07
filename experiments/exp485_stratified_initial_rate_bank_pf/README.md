# exp485_stratified_initial_rate_bank_pf

## 状態

- Route: `pf_beam`
- 状態: `stage1_gate_failed_terminal_close`
- CV: `11.092618091` / LB・Submit: なし
- 親: exp417、実装参照・保存control: exp404

## 仮説

tail30/32/64/128/256の複数initial-rate modeを各seed内で維持すると、
単一rate centerによる初期mode lossを減らせる。

## 変更点

500 particlesを5中心へ100粒子ずつinterleaveする。各中心のspreadは元の
`0.01`。5本のPFを作るのではなく、単一のequal-strata mixture priorとする。

## 検証方針

fixed32でallocation、rate bank、component ancestry、seed、truth-late、SHAを
確認する。runtime以外の全PASS、runtime例外、別承認後に773 wellsを
保存exp404 controlと比較する。
Stage 0はsuffix truth、fold、hidden-like role、保存controlを一切読まない。

## 所見

### 良い点

exp268ではw128が`0.042706 ft`改善し、5-bank oracleは`0.097314 ft`だった。

### リスク

423/773 wellsでrate spreadが0であり、bankが実質退化するwellが多い。

## Stage 0結果

- Kaggle private CPU version 1: COMPLETE
- fixed32: 32 wells / 156,088 rows
- candidate runtime: `1,278.942 sec`
- full runtime projection: `30,894.444 sec`
- fixed limit: `30,600 sec`
- gate: 13/14 PASS、runtime projectionのみFAIL
- multiple unique centers: 25/32 wells、single center: 7/32 wells
- fallback center: 0
- component extinction seed fraction max: `0.921875`

元の固定runtime gateは`294.444 sec`（`0.962%`）超過でFAILのまま保持する。
ユーザーがこの実行時間を許容し、Stage 1を明示承認したため例外実行する。

## 成果物

- compact self-contained Jupytext train sourceと正規train Notebook
- 5中心initial-rate bank、component ancestry、ESS/extinction診断
- fail-closed Stage 0 gateとSHA freeze
- 専用契約test 11件（PASS）
- run-on-pushで実行済みのKaggle train package version 1

Stage 0生成物はKaggle Notebook outputに保存した。Stage 1では全773 wellsの
truth-late CVを実行し、inference/submissionは行わなかった。

## Stage 1結果

- Kaggle private CPU version 3: COMPLETE
- candidate / saved exp404 control:
  `11.092618091 / 10.914522073`
- improvement: `-0.178096018 ft`
- positive folds: `1/5`
- by-well p95 / worst regression:
  `+0.422388632 / +33.053515117 ft`
- fixed HMM+PF 50:50 delta: `+0.032681136 ft`
- technical gate: PASS
- primary scientific gate / fixed-blend guard: FAIL / FAIL

version 2のtarget-free成果物をSHA固定してversion 3でtruth-late評価だけを
再開し、candidate PF rerunは0だった。事前登録どおりbranchを閉じ、
parameter/gate/blend/selector救済、inference、submissionは行わない。
