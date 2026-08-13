# exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264

## 状態

- ルート: `ml_model`
- 状態: Stage B technical PASS・scientific FAIL・no-rescue閉鎖
- CV: hard-primary OOF RMSE `8.668141`
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-26
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`

## 仮説

exp264の共有dual selectorは、12候補をcandidate-longで同数ずつ学習する。
候補単体RMSEが高いBeam/PF側も同じ学習量を持つことが共有modelのnoiseとなるなら、
候補を削除せず、各fit partition内の候補別RMSEの逆数でtask weightを掛けることで、
unweighted selector scoreとhard readoutを改善できる。

## 変更点

- 変更するのはLightGBM fitへ渡すcandidate-long sample weightだけ。
- 候補別RMSEは各modelのfit rowsだけで計算し、inverse、mean-one normalize、
  `[0.5, 1.5]` clip、再normalizeする。
- 同じ候補weightを`pred_abs_error`と`p_within10`へ適用する。
- validation、early stopping、OOF metric、gateはunweightedのままとする。
- `beam_mean`を含む12候補、88特徴、2 legal domain、fold、sampling、model paramsは固定する。

## 検証方針

- Fold: exp264 corrected outer 5 foldsを固定
- Group: well
- Stratification: 親のdeterministic sampled row contractを固定
- Leakage check: Stage Bはouter-train fit rows、将来Stage Cはinner-train fit rowsだけでweightを計算
- Control: 保存済みexp264 corrected Stage B v5。control再学習0
- 初回予算: 1 variant × 2 objectives × 5 folds = 10 CPU boosters
- 成功条件: steeringと`config.yaml`に固定したtechnical/scientific全AND gate

## 実装済み

- 親と同一SHAの`candidate_contract.yaml`
- 別名のJupytext compact self-contained train候補（8章）
- fit labelsだけを受け取るinverse-RMSE weight生成、最終range fail-closed
- 共有Stage Bのoptional sample-weight hook。既存実験はdefault unweightedのまま
- fold別weight table、sampling manifest、truth-read ledger、feature/model/OOF SHA
- 保存済みparent v5とのfold / bucket / hidden-like / by-well全AND gate
- 専用synthetic contract test

compact self-contained train候補を正規`*_train.ipynb`へ採用した。
inference notebookは既存placeholderを維持している。

## 実行入口

正規train notebookをcanonical kernel
`kentookumura/exp407-inverse-rmse-dual-selector-exp264-train`へpushし、
Kaggle CPU version 1を1,531.430秒で完了した。ローカル学習は行っていない。

## 結果

| メトリック | 値 |
| --- | --- |
| Kaggle train | v1 COMPLETE |
| technical gate | PASS |
| scientific gate | FAIL |
| expected-error MAE | 3.798670（親比+0.002869） |
| within10 logloss | 0.360461（親比+0.000489） |
| within10 Brier | 0.112648（親比+0.000197） |
| hard-primary RMSE | 8.668141（親比+0.081137） |
| Public LB | - |
| Private LB | - |

## 所見

weight/leakage/model/SHAのtechnical契約は全PASSした。一方、scoreのfold再現性、
hard-primary、1000+、hidden-like両面、worst-wellがFAILした。候補単体global qualityの
一様task weightでは、row-local selectorのnegative transferを減らせなかった。

## リスク / 注意

- global OOF RMSEをfit weightへ使うとfold leakageになるため禁止。
- RMSE weightはclassification objectiveの最適weightとは限らないため、
  within10 logloss/Brierのnon-regressionを必須とする。
- 弱い候補も局所的にはoracle-bestになり得るため、候補削除を行わない。
- FAIL後のinverse-square、clip/exponent grid、Beam削除、候補subsetは行わない。
- Stage C、Stage D、inference、submissionはscientific FAILにより閉鎖する。

## 次

exp407をweight変更やcandidate subsetで救済せず閉鎖し、exp264 corrected Stage B v5を
selector anchorとして維持する。原因診断は別の0-booster saved-OOF readoutとして扱う。

## 表記

用語は`backlog/KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせ、
実験名や設定名を除いて日本語優先で記録する。
