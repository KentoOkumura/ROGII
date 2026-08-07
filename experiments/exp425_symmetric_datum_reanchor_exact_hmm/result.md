# exp425_symmetric_datum_reanchor_exact_hmm 結果

## 状態

Kaggle private CPU Stage 0 version 1を完了し、`stage0_fail_closed`。
固定32 wellのmechanism preflightであり、CV、Public LB、Private LB、提出はない。

- kernel:
  `kentookumura/exp425-symmetric-datum-reanchor-exact-hmm-train`
- version / id_no: `1 / 128930925`
- Stage 1 eligible: false
- inference / submission: 未実施

## 仮説

persistent beta-filter rate disagreementをactivationとしてのみ使い、rateを変えずに
negative / parent / positiveのabsolute-datum枝を生成すると、将来GRを含む
exact backward likelihoodがtranslation-gauge lockから正しいdatum枝へsoftに
質量を移せる可能性がある。

## 固定設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Route: `pf_beam`
- Stage 0: exp412 fixed32のmechanism preflight
- branch states / prior: 3 / `0.10, 0.80, 0.10`
- branch shift: `±max(filtered position std, 0.35 ft)`
- event: first persistent beta-filter rate-gap activation、1 well最大1回
- rate process / GR emission / support / posterior mean: 親から固定
- RNG: なし

## 結果

technical gateは12 / 13 PASS、mechanism gateは3 / 7 PASSだった。全AND条件を
満たさないため、Stage 1へ進まず閉じた。

### Technical

- fixed32 role / unique well / cause fold coverage: PASS
- event wells: `21 / 32`、全5 foldsをcoverしてPASS
- finite coverage: `1.0`
- baseline saved-exp209 parity max abs: `0.0 ft`
- parent-only branch parity max abs: `0.0 ft`
- normalization max abs error: `2.438424884232404e-08`
- truth / episode reads before all prediction freeze: `0 / 0`
- peak RSS: `1.0018768310546875 GB`、PASS
- Stage 0実時間: `2,684.506174854秒`（約44.74分）
- full 773-well投影:
  `64,847.60228631694秒 > 30,600秒`、FAIL

### Mechanism

- soft datum direction agreement:
  `0.3965775516538685 < 0.60`、FAIL
- fold別direction:
  `0.439484 / 0.299473 / 0.579400 / 0.200294 / 0.480838`
- `> 0.50`のfold:
  `1 / 5 < 4 / 5`、FAIL
- backward-cause SSE reduction:
  `0.0006975834994096264`（約0.07%）`< 0.10`、FAIL
- forward-cause SSE regression:
  `-0.0029171639393186233 <= 0.02`、PASS
- matched-control RMSE delta:
  `-0.002165636334951593 ft <= +0.02 ft`、PASS
- matched-control reanchor mass:
  `0.28563503481676566 > 0.10`、FAIL
- active reanchor mass:
  `0.3288139217119144 >= 0.05`、PASS

branch identityを条件にした3つのexact HMMを固定priorとfull-sequence log evidenceで
周辺化する実装は、parity、normalization、finite、truth-late、SHAのtechnical契約を
満たした。科学的には、future GRを含むexact evidenceでもdatum方向を識別できず、
matched controlにも大きなreanchor massを割り当てた。

## 再現性

- deterministic submission anchor: false
- seed policy: RNGなし、固定well / row / position / rate / branch順
- Kaggle kernel version / id_no: `1 / 128930925`
- executed config SHA:
  `ff22cf694e3e7f2afc4fb9780f43825775f156192a29a47cc302e7eb990aa046`
- scientific contract SHA:
  `7471662e4ef0e347db76f17e0443e52a17e3f181d6b891667bc01b6323a994c1`
- first-pass message manifest SHA:
  `3f46ffa166364a5078b0f7137e583eb98bca2234b70f3e633b613bd0c83c1992`
- activation schedule manifest SHA:
  `210ec11a46279a1b7e32ad965471be6d4be7dd3ceabbb6a319c6b8ae86993d76`
- shift schedule manifest SHA:
  `fc023dc681a045f54089dca42c74febcdcdc15da0e123090da5b729292ab0f36`
- branch posterior manifest SHA:
  `06c478a546afbcda059d37f58c69d7f817ddfde32098117c5e53c8e1abe25697`
- prediction decompressed SHA:
  `728cf7448ae52147719dcb5cc16e95e4349a9afdc610fe00bbe0d74fa3545319`
- event schedule decompressed SHA:
  `87a405ff92804b078b19d9b1e9b9f01b2d67d3012388f1778c8aca28e8452559`
- Kaggle output metrics SHA:
  `32431724e5e8308d4ef457794cad4f15a423d421cc0c55e6515e3f749d7b17a7`
- summary SHA:
  `914ad2575fd294e4ac38740c1de03330de3aa77fcf321072f1168cc128a4816e`
- model / submission SHA: 対象外

## 解釈

設計補助で観測したrate差符号とdatum修正符号のSSE加重一致`0.396557`に対し、
exact future evidenceによるsoft datum方向一致も`0.396578`だった。方向を対称化して
rate符号の誤写像を避けても、現行GR likelihoodはabsolute datumを識別する情報を
増やせなかった。

control pooled RMSEは悪化しなかったが、control reanchor massが`0.2856`まで上がって
おり、branch posteriorは必要wellを選別できていない。backward-cause SSEも約0.07%しか
改善せず、主目的のtranslation-gauge lock修復には届かない。runtimeもfull投影で
約18.0時間となるため、科学・運用の両面で現行branchを棄却する。

## 次

trigger、shift、prior、branch readout、gateをfixed32上で救済せず、このbranchを閉じる。
Stage 1、inference、submissionは行わない。同じtrigger / symmetric-datum scheduleを
使う後続案は、exp425のFAILを成功根拠として継承せず、独立した機構証拠と別承認が
ある場合だけ再検討する。
