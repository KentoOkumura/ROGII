# 要件

## 依頼

Kaggle discussion 726465 の「neighbor well dataなしでもGR matchingをかなり進められ、complete wellを1 sampleとするnon-tabular modelで pooled 5-fold CV 4.xが可能」という示唆に沿い、対象井自身のhorizontal GR、対応Type Well GR、known `TVT_input` prefix、軌跡だけからhidden suffixのTVT alignmentを推定する実験を設計する。

今回の作業範囲は、`KAGGLE_DIRECTION.md` backlog、steering、実験scaffold、設定、数理・実行・判定契約の確定までとする。実験ロジック、Jupytext source、Notebook実装、test、Kaggle package、学習、推論、提出は行わない。

## 2026-07-19 Stage A実装承認

設計確定後、ユーザーの「exp295を実装してください」をStage A実装承認として受領した。追加範囲は別名compact self-contained train候補、fail-closed inference候補、専用contract tests、設定・記録更新までとする。canonical Notebook採用、Kaggle package、GPU push、Stage B、Stage C、submissionは承認範囲に含めない。

## 2026-07-20 runtime contract修復・再実行承認

Kaggle version 2のhard truth path infeasible失敗を診断し、固定decoderを維持したGaussian soft-label structured likelihood（`sigma=0.35 ft`）を推奨した。ユーザーの「実行してください」を、この単一objective修復の実装、canonical train Notebookへの反映、同じfold 0・1 neural modelのversion 3再実行承認として受領した。Stage B、Stage C、inference、submissionの承認は含まない。

## 仮説

対象井自身のcomplete-well horizontal GRと対応Type Well GRをknown prefixから得たwell contextで条件付け、continuous TVT unaryと固定state-space transitionを全井で同時に解けば、neighbor well dataやcandidate selectorなしでもpooled OOF 6.0 ft以下、stretch 5.0 ft以下へ到達できる。

## 制約

- Route: `ensemble`。learned GR emissionと物理的state-space decoderの双方が予測生成に本質的であるため、このrouteを使う。
- 1 sampleは1 complete wellとし、station単位tabular regression、独立local-window top1、candidate selectorを主モデルにしない。
- inference時に使えるhorizontal情報は対象井自身の`MD/X/Y/Z/GR/TVT_input`だけとする。
- 対象井以外のhorizontal wellのTVT、TVT_input、GR、path、XY neighbor、same-typewell donorをinference featureまたはpriorに使わない。outer-train wellsは共有モデル学習にだけ使用できる。
- Type Wellは対象井に対応して与えられる参照curveだけを使う。neighbor wellの意味で使わない。
- known prefixはhard anchorとwell context生成に使うが、test-time gradient updateは行わない。
- hidden suffixのtrue TVTはouter-validのloss、normalization、band、temperature、early stopping、model selectionに使わず、modelとposterior生成物をfreezeした後のreadoutだけに使う。
- outer-valid wellはwhole-well GroupKFoldで完全に分離する。window/random row splitは禁止する。
- exp209のTVT/rate state grid、transition grammar、initial-rate priorを固定し、最初の実験ではlearned transitionへ変更しない。
- learned emissionはcontinuous TVT stateのunaryを生成し、PF/Beam/ML candidate bank、exp263/exp293 candidate、hard top1、softmax candidate average、neighbor copyを使わない。
- final outputはexact forward-backwardによるTVT posterior meanとする。Viterbi/MAPは診断に限定する。
- pseudo-cutはouter-train wellsのtraining view作成だけに使う。exp244型のpseudo-start risk gateやlocal-linear correctionとして使わない。
- Stage Aはfold 0の1 architecture / 1 seed / 1 neural modelだけとし、既存controlを再学習しない。Stage BはStage A modelを再利用し、残り4 foldsの4 neural modelsだけを追加する。
- LightGBM/CatBoost/XGBoost config、booster、PF/Beam well-runは全stageで0とする。
- Stage Aの実装・GPU push、Stage Bの追加4-fold学習、inference、submissionはそれぞれ別のユーザー承認を必要とする。
- 再現性は`docs/06_reproducibility.md`に従い、fold map、pseudo-cut manifest、input、emission、posterior、model、predictionのcontent SHAを記録する。gzipはdecompressed content SHAを主証拠にする。

## 受け入れ基準

- steering 3文書、実験`architecture_contract.md`、`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`に未記入placeholderが残っていない。
- complete-well入力、neighbor-free境界、known-prefix conditioning、learned emission、固定SSM、posterior meanの数理契約が明記されている。
- Stage Aの1-fold smoke、Stage Bのfull 5-fold OOF、Stage Cの条件付きinferenceが分離され、各開始条件・モデル数・成功条件・禁止事項が固定されている。
- real GR、stable circular-shuffled Type Well GR、zero-GR/geometry-onlyを同じtrained modelで比較し、negative-control modelを別学習しない設計になっている。
- pooled OOF `<=6.0 ft`をLB 5.x promotion gate、`<=5.0 ft`をstretch gateとして固定している。
- OOFが`6.0～6.75 ft`でGR attributionが全fold成立した場合は別expでのarchitecture iterationだけを許し、exp295内のposthoc rescue gridやinferenceへ進まない。
- `KAGGLE_DIRECTION.md`で旧`heatmap_unary_exact_hmm_redecode_probe`をexp295へ統合し、既存backlog全体との優先順位を更新している。
- `make validate-exp EXP=exp295_prefix_anchored_wholewell_gr_alignment_ssm`が通る。
- canonical scaffold Notebookは上書きせず、別名compact self-contained Stage A候補とfail-closed inference候補が実装されている。
- Kaggle package、学習、Stage B、Stage C、submissionが作成・実行されていない。

## 次

Stage A実装レビュー後、ユーザーが明示承認した場合だけcanonical train Notebook採用とKaggle GPU pushへ進む。実装承認はKaggle GPU push承認を兼ねない。
