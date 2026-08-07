# exp252_pf_seed_medoid_selectability_audit

## 状態

- ルート: PF/Beam
- 状態: Kaggle CPU audit完了。candidate likelihood signalは部分支持、bank gateは不採用
- 学習: なし（0 config / 0 fold / 0 booster）
- PF再実行: なし
- Public / Private LB: 対象外
- 作成日: 2026-07-15
- 親実験: `exp243_pf_seed_medoids`

## 仮説

exp243 K8 medoidはbase8+K8 whole-well oracleを6.592426から5.499587へ改善した。
このheadroomが単なるtarget-side oracleではなく、cluster mass、likelihood、trajectory
distance、entropy/HHI、ESS/resampling、base path disagreementというtarget-free診断で
識別できるなら、別実験のouter-well-fold selectorへ進む根拠になる。

## 変更点

- exp243 canonical v3の保存済み生成物だけを固定入力にする。
- K8だけを残し、K3/K5は読まない。
- bank-selectabilityとcandidate-selectabilityを分離する。
- row / block 128/256/512 / whole-wellでAUC、固定top-decile coverage、top1 regretを読む。
- 全scoreへstable shuffled-score negative controlを付ける。
- selector学習、候補平均、PF再実行、raw-test inference、submissionは行わない。

## Leakage contract

score tableはtrue TVTを引数に取らない関数で先に固定する。true TVTはその後のloss / label
生成と評価にだけ渡す。scoreの符号、K、scope、tie、coverage fractionは実行前にconfigへ固定し、
結果を見たgridやscore選択は行わない。

## 検証方針

- Fold: 0。保存済み候補に対するno-training diagnostic。
- Group: well。blockとwhole-wellはwell境界をまたがない。
- primary: best-source labelのAUC、固定top-decile coverage、candidate top1 regret。
- negative control: scope/scoreごとのstable shuffled-score。
- leakage check: score freeze関数はtrue TVTを受け取らず、label stageだけがtargetを読む。

## 所見

Kaggle train v1は3,783,989 rows / 773 wellsを86.053秒で完走し、exp243の4入力SHA guardも
すべて通過した。K8内では`cluster_likelihood_mass`、`medoid_likelihood_rank_score`、
`medoid_likelihood_gap_from_best`が5 scopeすべてでshuffled controlを上回り、whole-well AUCは
それぞれ0.675214 / 0.655102 / 0.654235だった。

一方、K8 bankがbase8を上回るunitを見分ける最良固定scoreは`resampling_rate`でも
whole-well AUC 0.560593に留まった。`cluster_likelihood_mass` top1はuseful 374 wellsの
51.604%を回収したが、best base8比のlossは全well平均+3.194947 ftだった。したがって
medoid内のtarget-free順位付け信号は認めるが、base8を捨てるgateやdirect selectorには不十分。
likelihood mass / rank / gapは、base8 fallbackを維持するfold-safe selectorへadd-onlyで入れる
candidate-ranking特徴量候補にはなり得る。ただし3 score単独の固定selectorにはせず、現時点では
raw-test inferenceやsubmissionへ進めない。

生成時間は、保存済み候補からのexp252 readoutが全773 wellsで86.053秒。raw入力から
128-seed PFとmedoid bankを作るexp243 v3は773 wellsで37,067.406秒（約10時間18分）だった。
hidden test約200 wellsの単純well比例は約2時間40分だが、raw-test未実測の参考値である。

## 実行入口

- train-side audit: `exp252_pf_seed_medoid_selectability_audit_train.ipynb`
- inference guard: `exp252_pf_seed_medoid_selectability_audit_inference.ipynb`
- Kaggle package: `make prepare-kaggle-notebooks EXP=exp252_pf_seed_medoid_selectability_audit ...`

Kaggle Notebook実行を正とし、ローカルnotebook実行は行わない。
