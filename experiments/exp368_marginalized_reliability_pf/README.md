# exp368_marginalized_reliability_pf

## 状態

- ルート: `pf_beam`
- 状態: Stage 0完了・technical PASS / scientific FAIL・fail-close
- CV / LB / Submit: なし
- 作成日: 2026-07-23
- 親実験: `exp072_exp063_full_replay_feature_cache`

## 仮説

GRが一時的に信用できない区間をstickyな`normal / weak`状態として粒子ごとに
厳密周辺化すれば、追加粒子を消費せずexp072の観測過信を抑えられる。

## 親実験と変更点

親exp072の粒子数・seed数・dynamicsは変更せず、今回はStage 0だけを実装した。

- 最終連続known-TVT 192行を128 history / 64 held-outに分け、exp072同一GR補間と
  full-prefix sigmaで逐次予測NLLを評価する。
- 保存済みexp072 `likpf_mean` path上で512行 / stride 256のweak posteriorを作る。
- prefix NLLとsuffix weak posteriorをSHA freezeしてからsuffix truthをjoinする。
- bad10 AUC、circular control、5 folds、hidden-like 2面、weak massをAND評価する。
- Stage 1 PF、raw-test inference、submissionは実装していない。

## 検証方針

- known-prefix pooled predictive NLL gainを1%以上にする。
- saved exp072 pathのbad10 AUCを0.60以上、circular差を0.02以上にする。
- real bad10 AUCが0.50を超えるfoldを4/5以上にする。
- hidden-like spatial / typewell-purged AUCを両方0.55以上にする。
- row-weighted weak massを`[0.02, 0.50]`に収める。
- 1条件でもFAILした場合はStage 1へ進まない。

## 固定したreliability

- q遷移: `[[511/512, 1/512], [1/128, 127/128]]`
- 初期確率: `[0.8, 0.2]`
- normal: Type Well GR平均、exp072 sigma
- weak: 同じ平均、sigma 4倍
- update: 正規化済みGaussianによるexact 2-state forward recursion

## 実行量

- Stage 0: 1 diagnostic / 5 reporting folds
- PF seed-well runs / control replay: `0 / 0`
- model / LightGBM / trained fold / booster: `0 / 0 / 0 / 0`
- 条件付きStage 1: 1 treatment / 500 particles / 128 seeds / 773 wells /
  98,944 seed-well runs

## Notebook

- `exp368_marginalized_reliability_pf_train.ipynb`:
  compact self-contained Stage 0
- `exp368_marginalized_reliability_pf_inference.ipynb`:
  submissionを生成しないfail-closed Notebook

## 結果

Kaggle private CPU version 1を`630.531264 sec`で完了した。

- technical gate: PASS
- pooled bad10 AUC: `0.636675`
- real - circular AUC: `+0.058264`
- AUC > 0.50 folds: `5/5`
- hidden-like spatial / typewell-purged AUC: `0.641795 / 0.636115`
- known-prefix NLL gain: `0.037356% < 1%`でFAIL
- weak mass: `0.009689 < 0.02`でFAIL
- decision: `stage_0_failed_close_without_rescue`
- Stage 1 / inference / submission: 不適格・未実施

## 所見

### 良かった点

- Stage 0だけでreliabilityの予測尤度とsuffix error識別力を判定できた。
- 親exp072 PF controlを再実行せず、GPU・booster・PF runを消費しない。
- bad-block識別はpooled、5 folds、hidden-like 2面で安定していた。

### 悪かった点

- known-prefix NLL改善は事前下限の約3.7%に留まった。
- weak stateはsuffix row massの約0.97%にしか発火せず、PFに組み込む根拠がない。

### リスク / 注意

- exp363は同じq契約の0-HMM readoutでhidden-like spatial AUCとweak massをFAILした。
- exp232/233/241のrobust-likelihood系も強いnegative referenceである。
- Stage 0失敗後のsigma / transition / block / gate / blend rescueは禁止する。

## 次

branchを閉じる。Stage 1 PF、推論、提出は実装・実行しない。同じqの調整による
救済は行わず、再訪にはknown-prefixとsuffixのactivation乖離を説明する独立した
truth-free仮説を要求する。
