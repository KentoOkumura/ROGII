# 要件

## 初回依頼（設計）

物理モデルの状態遷移を改善する第1案として、exp209 の持続状態
`(TVT, U-rate)`を維持したまま、rate と TVT の遷移を連続運動学から導いた
相関 joint transition へ置き換える
`exp439_continuous_kinematic_joint_transition_exact_hmm`を設計する。

初回はアイデアバックログ、steering 3文書、実験ディレクトリ、設定・記録文書までを
作成し、設計を確定する。構造検証用の markdown-only notebook placeholder 以外の
実装、実行可能 notebook、test、Kaggle package/push/run、推論、提出は行わない。

## 追加依頼（実装）

2026-07-29 のユーザー依頼「exp439を実装してください」を実装承認として扱う。
Jupytext percent形式のcompact self-contained Stage 0候補、非負moment projection、
correlated joint HMM、contract test、fail-closed inference候補を実装する。

既存の正規notebookは明示採用なしに上書きせず、compact候補を別名で生成する。
Kaggle package/push/run、Stage 1、inference実行、submissionは引き続き未承認とする。

## 仮説

現行 exp209 は先に rate を隣接3状態へ遷移させ、その後 destination rate だけで

```text
delta_TVT = r_t * delta_MD - delta_Z
```

を作り、0.35 ft の TVT grid へ5点 Gaussian で投影する。この分離により、
区間中の rate 変化が位置変位へ正しく積分されず、grid 投影後の一次・二次 moment も
連続モデルと一致しない。

各 legal rate edge `r_{t-1} -> r_t`について、

```text
r_t = r_{t-1} + eta_r
delta_TVT = 0.5 * (r_{t-1} + r_t) * delta_MD - delta_Z + eta_p
```

を1つの joint edge として離散化し、rate marginal、条件付き平均変位、条件付き分散、
`Cov(delta_TVT, delta_r)`を保存すれば、rate履歴を失わずに区間内の連続運動と
position gridのmomentを整合させられる。

## 制約

- Route は `pf_beam`。
- 親/control は
  `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- 状態、41 rate support、rate step、`sig_r`、`mom`、rate boundary semantics、
  TVT grid、`sig_p`、GR emission、prior、readout、forward/backward は固定する。
- 科学差分は、destination-rate position kernel を相関 joint edgeへ置き換えることだけ。
- rate marginal は exp209 の隣接3状態 kernel と完全一致させる。
- `eta_p`はrate edgeを条件とした独立process noiseとし、exp209の
  `max(sig_p, 0.35 * position_step)`を固定する。
- 離散 position kernel は、確率和、平均、分散を満たす非負 moment projection とする。
- position support は`5 -> 7 -> 9`セルの順に最小の実行可能な奇数幅を選び、
  9セルでも実行不能ならfail-closeする。support選択規則を同一OOFで変更しない。
- fixed32 の1 variantだけを先に評価し、保存済みexp209 predictionをcontrolに使う。
- hard trigger、ML、selector、blend、reset、re-anchor、rate/position/emission tuningを禁止する。
- 実装承認は2026-07-29に取得済み。実行は別のユーザー承認を必要とする。

## 受け入れ基準

- steering 3文書、実験配下の`config.yaml`、`README.md`、`SESSION_NOTES.md`、
  `result.md`、`metrics.json`が同じimplemented-but-unrun契約を持つ。
- `KAGGLE_DIRECTION.md`の未着手バックログへ第1案としてexp439を登録する。
- `experiment_summary.md`へ`implemented_pending_stage0_approval`として登録する。
- exp437 の neighbor-geometry TVT-only HMM は別仮説として維持し、第1案に数えない。
- Stage 0 は fixed32 の1 variant ×32 wells、parent rerun 0 とする。
- rate marginal parity、moment/covariance parity、forward/backward kernel parity、
  brute-force小規模HMM parity、truth-late、runtime/RSSをtechnical AND gateにする。
- Stage 0 mechanism gateは、forward-cause SSE 10%以上、persistent SSE 5%以上、
  persistent 10/16 wells、4/5 folds、matched-control pooled/p95をすべて満たす。
- Stage 1 は Stage 0 全PASSと別承認後だけ773 wellsで行う。
- deterministic anchorは同一設定rerunでlogical content SHAが一致するまで主張しない。
- compact self-contained train/inference候補とcontract testが存在する。
- moment projectionは5/7/9の順で探索し、全supportで不可能な位相をfail-closeする。
- 正規notebookはplaceholderのまま保持し、compact候補だけを実行可能Notebookへ変換する。
