# 要件

## 依頼

`exp226_prefix_masked_offset_predictability_readout`の実験ディレクトリとsteeringを作成し、
実装前に科学的契約を確定する。今回は設計と記録だけを行い、notebook実装、Kaggle package作成、
Kaggle実行、補正prediction、推論、提出は行わない。

## 仮説

Assumption: exp226 geometry-only `tvt_geop`は局所形状を捉えているが、wellごとの低周波offset / driftを
残している。official known `TVT_input` prefixの末尾をpseudo suffixとして一度隠し、group-safeな
exp226 geometryを再生した後にその既知区間のoffsetを測れば、test-timeにも作れるprefix-only evidenceとして
official evaluation suffixのexp226 residual median / slope / block driftを予測できる可能性がある。

## 制約

- 実験名: `exp285_exp226_prefix_masked_offset_predictability_readout`
- Route: `pf_beam`
- 科学的親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- negative references: `exp279_exp226_geop_centered_exact_hmm_redecode`、
  `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- supporting diagnostic: `exp280_exp226_shift_likelihood_separability_readout`
- validation wellはexp226と同じ保存5 foldsを使い、当該foldのdonor field / kappa fitからwell全体を除外する。
- pseudo cutはofficial known prefixの末尾640行を隠す1点だけとし、cut以前のvisible rowを最低512行要求する。
- 最後のknown rowで終わる連続finite `TVT_input` blockが1,152行以上あるwellだけをeligibleとする。
- pseudo viewではcutより後の全`TVT_input`をgeneratorへ渡さない。target geometryはcut以後もtest-timeで
  観測可能な`X/Y/Z/MD`だけを使い、cutからwell末尾までをunknownとしてexp226 K=16 geometryを再生する。
- pseudo offsetは再生path freeze後にmasked 640行の`TVT_input`だけを戻して算出する。raw true `TVT`は使わない。
- official suffix true TVTはpseudo pathとpseudo summaryのschema/content SHAを確定するまで読まない。
- exp226 full predictionの`gr_delta` / `tvt_pred`は使わず、geometry-only `tvt_geop`だけを対象にする。
- cut、mask長、visible最小長、block数、summary、clip、閾値、guardは同じreadout結果で変更しない。
- 1 fixed readout variant / LightGBM config 0 / trained fold 0 / booster 0 / HMM・PF regeneration 0。

## 固定pseudo summary契約

- masked 640行をrow順に5個の非重複128行blockへ分ける。
- row residualは`TVT_input - pseudo_tvt_geop`とする。
- `offset_median`: 640行全体のfinite residual median。
- `offset_slope`: 5 block residual medianをblock-center MDへ切片付き最小二乗した傾き。
- `block_drift_rate`: `(last block median - first block median) / (last center MD - first center MD)`。
- raw first/last block median、raw block drift、finite fraction、MD spanも診断列として保存する。
- slope / driftの分母が非finiteまたは0、または任意blockのfinite coverageが1.0未満ならwellを黙って落とさず
  technical guardをFAILさせる。

## 固定official target / readout契約

- official residualは保存済みexp226 OOFのofficial `tvt_geop`に対する
  `tvt_true - tvt_geop`とする。
- official suffix全体をrow数で5個の連続blockへ`array_split`し、pseudo側と同じ3 summaryを作る。
- primary pairはpseudo `offset_median`対official full-suffix `offset_median`。
- supporting pairは対応する`offset_slope`と`block_drift_rate`。
- metricはwell単位Pearson、Spearman、符号accuracy / balanced accuracyとする。符号はresidual summaryが
  0以上ならpositive、0未満ならnegativeに固定する。
- negative controlはfold内well assignmentを256回stable SHA256 local RNGで置換し、primary Spearmanの
  add-one permutation p-valueとnull p95を記録する。各permutationは全fold内で独立に置換した後、
  全eligible wellを再結合したpooled Spearmanを1値として計算する。
- H256 / H512 / H640、near `0-250 ft`、1000+、hidden-like spatial、hidden-like typewell-purged、
  fold、by-wellを固定scopeとして記録する。scopeは結果後に追加選択しない。

## 受け入れ基準

- technical guard:
  - canonical official OOF `3,783,989 rows / 773 wells / 5 folds`。
  - eligible wells `>=750`、全5 foldsにeligible wellが存在する。
  - target well donor exclusion、pseudo mask identity、pseudo path finite、pseudo summary finite coverageが全て1.0。
  - pseudo path生成前のmasked `TVT_input` access 0、pseudo summary freeze前のofficial suffix true TVT access 0。
  - input SHA、fold identity、saved kappa identity、row identityが固定契約と一致する。
- primary predictability guard:
  - pseudo median対official full-suffix medianのpooled Spearman `>=0.30`。
  - Spearmanが5/5 foldsで正、かつ4/5 folds以上で`>=0.20`。
  - pooled sign balanced accuracy `>=0.60`。
  - 256回fold内stable permutationに対するprimary Spearman p-value `<=0.01`。
- supporting guard:
  - slope / block-drift-rateの少なくとも1 familyがpooled Spearman `>=0.20`かつ4/5 foldsで正。
  - near、1000+、hidden-like spatial、hidden-like typewell-purgedのpooled primary Spearmanが全て正。
- 全guard PASSだけが、別実験でprefix-calibrated candidate correctionを設計する根拠になる。
- FAIL時はcut / mask / block / summary / clip / threshold grid、exp281 blend/selector救済、補正、
  current-test生成、raw-test inference、submissionへ進まずnegative resultとして閉じる。

## 初回設計の完了条件

- steering 3文書と実験ディレクトリが作成されている。
- `config.yaml`、README、SESSION_NOTES、result、metricsに未確定の科学的項目がない。
- mask、freeze順序、summary、negative control、guard、禁止事項、実行量が固定されている。
- 正規train/inference notebookはtemplate stubのままで、Jupytext sourceや実験ロジックを実装しない。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`へ設計確定・実装前として登録する。

## 2026-07-19 追加実装依頼

ユーザーの「実装してください」により、正規stub notebookを上書きせず、別名compact self-contained
Jupytext train / inference、freeze境界、summary、correlation/permutation/guard、専用合成testまでを
実装対象へ追加する。Kaggle package作成、Kaggle実行、補正、current-test生成、推論、提出は対象外のまま。

追加完了条件:

- train notebook上で入力、fold-safe donor replay、pseudo mask、freeze、official readout、guard、保存先を追える。
- inferenceはfail-closedとし、prediction / submissionを生成できない。
- mask 640、well-end replay、5 x 128 summary、truth freeze、stable permutationを合成testで確認する。
- Jupytext round-trip、`py_compile`、ruff、専用pytest、strict experiment validationを通す。
- 追加実行依頼を受けるまでは`execution.kaggle_push_approved=false`を維持する。

## 2026-07-19 追加実行依頼と結果

ユーザーの「実行してください」により、1 variant / 0 config / 0 trained fold / 0 boosterのKaggle CPU
readoutを承認対象へ追加した。compact trainを正規trainへ採用し、version 1のraw `id`列契約不一致だけを
修正してversion 2を完走した。766 eligible wellsでtechnical guardは全PASSしたが、primary Spearman
`-0.004135`、balanced sign `0.488567`、permutation p `0.599222`のためscientific guardはFAILした。
固定受け入れ基準に従い、parameter rescue、補正、current-test生成、推論、提出へ進まず閉じる。
