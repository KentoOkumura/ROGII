# 要件

## 依頼

「TVT-rateの遷移を`dZ/dMD`で条件づける」学習型likelihood-PFを、
`exp450_dzdmd_conditioned_tvt_rate_likelihood_pf`として設計確定する。
backlog、steering、実験ディレクトリを作るが、実装・実行は行わない。

## 仮説

exp446は持続`TVT-rate`から既知`Z`勾配の駆動を外したため、fixed32で
under-responseとRMSEを悪化させた。visible prefixだけで

```text
q = dTVT/dMD
g = dZ/dMD
mu(g) = beta * g + intercept
```

をwell別に推定し、`q-mu(g)`だけを持続させれば、TVT-rate状態の解釈可能性を
保ちながら既知`Z` forcingをtransitionへ戻せる。

## 制約

- Routeは`pf_beam`。
- 親endpointはexp417で監査した保存済みexp404
  temperature-5 / GR scale x1.0 likelihood-PFとする。
- 科学variantは学習型1本だけ。`beta=-1, intercept=0`は技術parity専用で、
  科学候補やselector候補にしない。
- `beta`とinterceptはwell自身のvisible prefixだけから、既存`PF_Z`と同じ
  最低10 valid stepsのOLSで推定する。unknown suffix TVTは読まない。
- 変更はrate transition centerだけとする。500 particles、128 seeds、
  momentum、process noise、position noise、initial spread、roughening、
  resampling、GR likelihood、temperature-5 seed aggregationはexp404から固定する。
- PF_Zの追加rate likelihood、smoothed-GR mixture、rate noise推定は移植しない。
- Stage 0Aはexp410 sentinel12で親座標とのpaired parity、Stage 0Bは
  exp411/exp446 fixed32でmechanismとcontrol安全性を確認する。
- Stage 1の全773 wellsはStage 0A/0Bの全AND gate PASSかつ別ユーザー承認時だけ。
- 保存exp404 controlを再実行しない。ML、HMM、Beam、booster、GPUは0。
- `docs/06_reproducibility.md`に従い、per-well/seed stable SHA seed、
  thread非依存RNG、truth-late freeze、decompressed content SHAを設計する。
- 実装、test、Jupytext source、Kaggle package、push、run、inference、
  submissionは未承認。

## 受け入れ基準

- OLS入力、minimum support、fallback、初期rate、`g_t`、`mu_t`、residual-AR
  transition、position updateが一意に定義されている。
- `beta=-1, intercept=0`が同一乱数で親U-rate PFと一致するparity gateがある。
- prefix-only回帰をunknown suffix、truth/error、fold/role読取前にfreezeする。
- Stage 0A/0B/1のvariant数、well-run、seed-well、particle starts、
  control rerun、model/booster/GPU数が記載されている。
- primary RMSE、fold、固定scope、well-tail、fixed HMM/PF blendの
  全AND gateが事前登録されている。
- beta/interceptのclip・shrink・grid、rate likelihood、well/row gate、
  blend/selector、same-OOF救済が禁止されている。
- 初回成功runをdeterministic anchorとせず、anchor化には独立rerun、
  input/config/prefix-fit/prediction SHAとKaggle kernel versionを要求する。
- gzip生成物はraw archive SHAだけでなくdecompressed content SHAを主証拠にする。

## 2026-07-30 Stage 0A後のユーザー承認

Kaggle version 1では、数学的に等価な二つの演算順の丸め差が、1 wellで
ESS/resampling境界を跨いで内部粒子系列を分岐させた。一方、実際の親出力である
temperature-5集約予測の最大差はsentinel12全体で
`4.836692824e-09 ft`だった。

ユーザーの「微小な丸め誤差なら次に進んでください」により、Stage 0Aの
受け入れ基準を次へ改訂する。

- primary parityはtemperature-5集約予測の最大差`<=1e-6 ft`とする。
- seed別予測、particle weight、log-likelihood、position/rate、
  resampling decisionの差は診断値として全て保存するが、gateには使わない。
- well数、実行量、finite coverage、clip decision、stable seed、truth-free、
  artifact readbackは引き続き必須gateとする。
- CSV readbackは17桁float出力とround-trip parserで検証する。
- Stage 0A改訂gate PASS時だけ、同じversion 2 run内で事前登録済みStage 0Bへ進む。
- Stage 0Bの科学variant、設定、mechanism gateは変更しない。
- Stage 1、inference、submissionは引き続き別承認とする。
