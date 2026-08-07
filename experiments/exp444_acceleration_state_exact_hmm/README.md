# exp444_acceleration_state_exact_hmm

## 状態

- ルート: `pf_beam`
- 状態: Stage 0A technical FAIL、runtime projectionによりterminal close
- 優先度: P4、高リスク
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-29
- 構造参照・一要因control: `exp441_full_support_ou_rate_transition_hmm`
- root比較対象: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

rateを各行で独立に追い直すだけでなく、rateが増加中・一定・減少中という
小さな持続状態を持てば、GRが弱い行をまたいでもrate trendを維持できる。

## 固定状態

```text
acceleration = [-0.0005, 0, +0.0005] rate/MD-ft
transition   = [0.08, 0.84, 0.08]
initial      = [0, 1, 0]
```

境界から外向きの0.08は境界stayへ加える。acceleration更新後、その値でexp441
OU rate平均を`a_t*delta_MD`だけ動かし、更新後rateで親どおりTVTを進める。

## 独立仮説

当初はexp441/442の結果を先行条件にしていたが、2026-07-30のユーザー判断で
撤回した。exp441のStage 0 FAILはnegative contextとして固定し、
「full-support OU単体では不足したが、明示trend-memory stateを加えると
persistent lagを回復できるか」という独立した組合せ仮説として扱う。
exp441/exp442の結果をpositive evidence、parameter変更、gate救済に使わない。

## 検証方針

- Stage 0A: fixed32 identityからtarget-free hashで4 wellsを選び、runtime/memory/
  exactnessだけを評価する。
- PASS後のStage 0B: 残り28 wellsを足してfixed32全体をmechanism評価する。
- fixed32では保存exp441を一要因readout、保存exp209を安全性controlに使う。
- 全PASS・別承認時のみ773 wells。
- model / booster / PF / Beam / GPUは0。

## リスク

state数が3倍になり、runtime/memoryと識別不能性のリスクが大きい。
accelerationはwrong trendも持続させ得る。Stage 0Aでfixed32/full換算runtimeが
上限を超えた場合は実装方式やstate数を救済せず閉じる。

## 実装

- Jupytext percent形式のcompact self-contained train/inference候補を作成。
- `acceleration -> rate -> TVT -> GR emission`を因子化した厳密
  forward/backwardを実装。
- boundary transition、initial zero prior、zero-acceleration exp441 OU parity、
  small-state dense referenceを実装。
- fixed32 manifestは`well`列だけを読み、固定SHA順で4 wellsを選ぶ。
- prediction、acceleration posterior、diagnostic、joint transition SHAを
  target-freeでfreezeする。
- compact trainを正規train Notebookへ採用した。正規inference Notebookは
  scaffoldのまま変更していない。

## 所見

### 検証結果

- 専用pytest: `14 passed`。
- py_compile、Ruff、Jupytext train/inference round-trip: PASS。
- strict experiment validation、template validation: PASS。
- 親exp441 / exp444 train sourceは`3,070 / 2,537 lines`、どちらも10章構成。
- forward正規化監査はpredictive/filtered全状態の正規化済み確率を
  独立に再加算して誤差を測る。
- Kaggle version 1は4 wells / 21,962行を完走した。
- exactness、normalization、leakage、RSSはPASS。
- fixed32/full換算runtimeは`5,970.830 / 144,232.851 sec`で、
  上限`3,600 / 30,600 sec`をFAILした。
- 実装検証だけであり、acceleration stateを支持する科学結果ではない。
- 4案中で計算・識別リスクが最大のためP4とする。
- 2026-07-30に正規train Notebook採用、package、private CPU Stage 0Aが
  明示承認された。Stage 0B/1、inference、submissionは未承認。
- Kernel: `kentookumura/exp444-acceleration-state-exact-hmm-train`
  version 1、id_no `129154702`。
- 事前契約どおりStage 0B/1、runtime/state/kernel/parameter/gate救済、
  inference、submissionへ進まない。

## 次

exp444をterminal closeとして保持する。Stage 0B/1、inference、submissionは
実行せず、同じbranch内でruntimeやstate設計を救済しない。
