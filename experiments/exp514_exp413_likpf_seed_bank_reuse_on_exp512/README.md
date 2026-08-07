# exp514_exp413_likpf_seed_bank_reuse_on_exp512

## 状態

- ルート: `ensemble`
- 状態: Stage A PASS、Stage B v2 scientific FAIL、Stage D v3 hidden rerun ERROR、Stage D v4 visible technical PASS
- 親実験: `exp512_hjyact_v2_final_10pct_hedge_on_exp413`
- CV / Public LB / Private LB: なし / error・scoreなし / なし
- 実装 / Stage A Kaggle run / submission: `1 / 1 / 0`
- 作成日: 2026-08-05

## 仮説

exp413のwell別stable seed likelihood-PFを1回だけ生成し、同じ128軌跡からscale 3/5/8/12と
SP45 branch統計を派生させれば、exp413 scale 5を維持しながらSP45の重複PFを除去できる。

SP45 legacyとはRNG、seed、最初のMD差分、typewell補間が異なるため、同値高速化ではなく
精度検証が必要な新variantとして扱う。

## 固定した変更

1. exp413 likelihood-PFをSP45より前にwellごとに1回生成する。
2. 同じseed bankからscale 3/5/8/12を集約する。
3. known prefixを連結してSP45 full-length配列を作る。
4. 同じraw seed paths / likelihoodからbranch統計を作る。
5. scale 5だけをexp413へ、全scaleとbranch統計をSP45へ渡す。

learned x1.3 PF、Gold masked-prefix PF、`pf_ancc`、`pf_z`、Beamは共有しない。

## 検証方針

- Stage A: target-free fixed32、scale5 parity、adapter、ledger、thread parity、2-run SHA。
- Stage B: Stage Aと同じtarget-free fixed32のlegacy/shared paired精度screening。
- Stage C: ユーザー指示により不要。実装・実行せず、PASSとも扱わない。
- Stage D: 別承認後にsubmission-readyコードをvisible testで実行し、readinessと工程別runtimeを確認する。
- 200-well runtime: Stage Dの工程別時間を4-way並列工程と逐次工程に分けて外挿し、上限推定が9時間以内かを判定する。

visible 3 wellは親SP45のphysical overrideによりPF差が最終出力へ現れないため、精度のpositive gateには使わない。
工程別runtimeは見積もり入力に使うが、hidden 200 wellsの実測や完走保証にはしない。

## 実行入口

- train Notebook: `exp514_exp413_likpf_seed_bank_reuse_on_exp512_train.ipynb`（placeholder）
- inference Notebook: `exp514_exp413_likpf_seed_bank_reuse_on_exp512_inference.ipynb`（placeholder）
- Stage A: `exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_a_fixed32.ipynb`（Kaggle v1 PASS）
- Stage B: `exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_b_fixed32.ipynb`（v1採点ERROR、v2 COMPLETE / scientific FAIL）
- Stage D: `exp514_exp413_likpf_seed_bank_reuse_on_exp512_stage_d_visible.ipynb`（v1 ERROR、v2/v3/v4 COMPLETE）
- full inference候補: `exp514_exp413_likpf_seed_bank_reuse_on_exp512_compact_selfcontained_inference.ipynb`
- Stage D v3のvisible提出形式とruntime見積もりはPASSしたが、user code submission ref `55266559`は
  hidden rerun ERROR。末尾のvisible v2固定SHA guardがhidden分岐されていない欠陥を確認した。
- Stage D v4はRidge中間物解放、shared PF/SP45の4-thread well streaming、SP45→HJYACT→exp413の
  DataFrame所有権移譲、hidden-safe visible SHA guardを実装した。Kaggle version 4はvisibleでCOMPLETEし、
  v2の5出力SHA完全一致、submit-check PASS、200-well推定`6.289658--8.057147h`。再提出は未実施。

## 実装状態

- exp073/exp413とAST一致するstable seed / Numba likelihood-PF coreをself-contained化した。
- raw 128-seed bankはwell内だけに保持し、all-scale集約とbranch summary後に解放する。
- SP45 legacy bankとexp413後段`build_likpf`はcandidate実行経路から除外した。
- ledgerはproducer/core/SP45/exp413を各1回、legacy/duplicate/fallbackを0回としてfail-closeする。
- dedicated contract testを含む19件、構文、Ruff F821、Jupytext round-trip、
  strict validationをPASSした。
- Kaggle T4上の実Numba経路を32 wells × thread 1/4 × 2 rerunで実行し、aggregate / branch / ledgerの
  3 SHAが4 run完全一致した。合計実行時間は`2,363.410299秒`。

## 所見

- exp413 scale 5はexact parityとして検証できる。
- SP45側はseed/RNG等が変わるため、visible final parityではなくfixed32 paired精度screeningが必要。
- Stage D v4の工程別外挿は`6.289658--8.057147h`でruntime estimated PASSだが、hidden runtime実測ではない。
- Stage Bはpooled delta `+0.049680 ft`、nonworse fold `2/5`、by-well p95 `+0.647871 ft`で
  scientific all-ANDをFAILした。高速化できてもSP45の科学的置換は不採用。

## 次

exp514の科学評価は救済なしで終端したまま。Stage D v4のvisible技術検証はPASSしたが、hidden OOMを保証せず、
Stage B scientific FAILも変えない。competition再提出は未承認・未実施。
