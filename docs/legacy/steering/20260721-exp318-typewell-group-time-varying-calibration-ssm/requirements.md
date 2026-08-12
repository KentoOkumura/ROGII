# 要件

## 依頼

同一well内でGR校正が時間/深度とともに変化する可能性を、Type Well群priorで初期化した低次元state-spaceとして検証する。exp295のstate explosionを繰り返さず、まず既知prefix mask backtestで可同定性とruntimeを判定する。設計のみ。

## 制約

- Route: `pf_beam`。exp311/313 PASSが先行条件。
- stateはinterceptとlog-scaleの2次元local-levelだけ。
- current wellのvisible prefixだけでfilterし、boundaryを越えるsmoothingは禁止する。
- TVT pathとのjoint training/decode、suffix truth fit、parameter gridは禁止する。

## 受け入れ基準

- Stage 0はlast 640 known-prefix rowsをmaskしたcausal backtestに固定する。
- process noiseはouter-train empirical Bayesで一度だけ決定する。
- gain 0.05 ft、4/5 folds、boundary jump p95≤3σ、hidden-like非悪化、worst +0.25 ft、8.5h以下を全要求する。
- Stage 0 PASSまでStage 1 suffix extrapolationを実装しない。
