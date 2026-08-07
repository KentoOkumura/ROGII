# exp295_prefix_anchored_wholewell_gr_alignment_ssm

## 状態

- Route: `ensemble`
- 状態: Stage A fold 0 version 3 runtime timeout・branch close
- CV: なし
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-19
- primary parent: `exp202_heatmap_mdn_candidate_generator_probe`
- transition parent: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

対象井自身のcomplete-well horizontal GRと対応Type Well GRを、known prefixから得たwell contextで条件付けてcontinuous TVT unaryへ変換し、固定した物理state-space grammarでsuffix全体を同時推論すれば、neighbor well dataやcandidate selectorなしでpooled OOF 6.0 ft以下、stretch 5.0 ft以下へ到達できる。

## 変更点

- 1 sampleを1 complete wellとするnon-tabular model。
- shared multi-scale GR encoder + prefix-conditioned FiLM + row x TVT learned unary。
- exp209のstate grid/transitionを固定したexact forward-backward posterior mean。
- inference時のhorizontal sourceは対象井1本だけ。neighbor、same-typewell donor、spatial priorを使わない。
- candidate bank、hard top1、PF/Beam/ML blend、test-time backpropを使わない。
- pseudo-cutはouter-train augmentationだけに使い、confidence/risk gateにはしない。
- 学習objectiveはfixed decoder上のGaussian soft-label structured NLL（`sigma=0.35 ft`）1.0 + local true-state CE 0.25。version 2で不可能だったhard truth pathは要求しない。

詳細な不変条件、数式、stage分岐は[architecture_contract.md](architecture_contract.md)とsteeringを正とする。

## 検証方針

- Fold: 5-fold complete-well GroupKFold。exp202 fold identityを再利用してfreezeする。
- Valid input: organizerと同じofficial `TVT_input` visible-prefix/hidden-suffix mask。
- Stage A: fold 0の1 neural modelだけを学習するGPU smoke。
- Stage B: Stage A全PASS後、fold 0を再利用して残り4 modelsを追加するfull OOF。
- Controls: 同一trained modelのreal GR、Type Well circular shuffle、zero-GR/geometry-only。
- Primary promotion: pooled OOF `<=6.0 ft`、GR attribution、5/5 fold、1000+、hidden-like、worst-well guardを全PASS。
- Leakage: outer-valid truthはmodel/unary/posterior/control manifest freeze後のreadoutだけに使う。

## 実行規模

現在はStage A実装1、active architecture 1、running fold/model `0 / 0`、学習完了fold/model `0 / 0`、LightGBM config / booster / PF-Beam well-run `0 / 0 / 0`。Kaggle version 2はhard truth path infeasible、version 3はsoft-label objectiveでepoch 1完了前にruntime timeoutした。control model再学習は0。

Stage Aは1 architecture x 1 fold x 1 seed = 1 neural model。Stage Bはfold 0を再学習せず、残り4 modelsを追加して合計5 modelsとする。Stage B以降のGPU pushは別承認を必要とする。

## 実行入口

- Stage A実装候補: `exp295_prefix_anchored_wholewell_gr_alignment_ssm_compact_selfcontained_train.py` / `.ipynb`
- fail-closed inference候補: `exp295_prefix_anchored_wholewell_gr_alignment_ssm_compact_selfcontained_inference.py` / `.ipynb`
- canonical train: `exp295_prefix_anchored_wholewell_gr_alignment_ssm_train.ipynb`（compact self-contained候補を採用済み）
- canonical inference: `_inference.ipynb`（未採用scaffoldのまま）
- canonical trainはmask-first loader、fold/pseudo-cut freeze、prefix context、multi-scale FiLM encoder、fixed exp209 exact forward-backward/Viterbi、truth-late readout、Stage A guardまでをself-containedに持つ。
- `execution.kaggle_push_approved=true`はStage A fold 0だけに限定し、inference/submissionはfail closedのまま維持する。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage A | version 2 objective failure、version 3 runtime timeout（model未学習） |
| Full 5-fold OOF | 未実行 |
| Public / Private LB | 未提出 |

## 所見

### 現時点で確定したこと

- GR matchingをcandidate confidenceではなくcontinuous TVT emissionそのものとして学習する。
- complete-well global continuityとknown prefix anchorをexact state-space posteriorへ統合する。
- neighbor-free境界、GPU段階実行、negative controls、LB 5.x promotion gateを実装前に固定した。
- fold 0の1-model実装と10本の専用contract testを追加し、ローカルにPyTorchがない環境ではneural/DP実行テストだけを明示skipする。

### 未確認のこと

- Stage AのGR attribution、runtime、memory、band coverage。
- full 5-fold OOFとCV/LB転移。
- current-test inference再生成性。

## リスク

- GRではなくtrajectory/prefix shortcutを学ぶ可能性。
- outer-valid suffix truth、same-typewell horizontal、neighbor pathが混入するleakage。
- exp209 state band外の真値を回収できないcoverage制約。
- row x TVT unaryとstructured lossのGPU memory/runtime。
- CUDA/AMPによる非byte-determinism。
- 3 public wellsに由来するCV/LB分散。

## 次

固定runtime gate FAILとしてexp295をbranch closeする。whole-well unaryを再訪する場合は、計算可能な学習objectiveを別expとして設計し、GPU実行前にユーザー確認する。
