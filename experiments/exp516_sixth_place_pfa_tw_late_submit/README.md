# exp516 6位解法 `pfA × twGR` LATE SUBMIT再現監査

Kaggle discussion 733226と公開submission Notebookを一次資料として、6位解法の単体Particle Filter componentを`faithful`に再現する。

対象は`pfA × typewell GR`、GR-free物理anchor、learned emission、600 particles、32 seeds、whole-interval ancestral smoother、seed likelihood soft averageである。91候補のrow-level bagging、TCN、GBM、de-shrinkはユーザー確認により対象外とした。

この実験はコンペ終了後の再現監査である。技術gateを通った固定版を1回だけ提出し、Notebook titleとsubmission messageに`LATE SUBMIT`を明記する。6位チームの最終スコアや正式順位と、exp516のlate-sub scoreを混同しない。

## 状態

実装と静的契約検証まで完了。Kaggle T4 x2でのfull run、提出前検証、late submissionは未実行。Active Sessions確認はpush前gateにしない。

## 仮説

公開された`pfA × twGR` componentを、GR-free anchor、learned emission、600 particles、32 seeds、full ancestral smoothingを省略せず再生成すれば、作者報告の単体componentと比較可能なlate-sub scoreを得られる。

## 検証方針

公開Notebook、v96 config、5本のencoder checkpointをSHA固定する。current runtimeのtrain/testからanchorとsimilarityを再生成し、`pfA × twGR`のsmoothed meanだけをsample submissionのID・順序・target列へ1対1整列する。固定版を`LATE SUBMIT`として1回だけ提出し、結果を再調整には使わない。

## 所見

公開実装から単体branchを抽出できた。現時点の数値は作者報告の外部参照値だけであり、exp516の結果はKaggle full runとlate submissionの完了後に記録する。

- Route: `pf_beam`
- Status: `implemented_static_validation_passed`
- Source: `k256net/public20th-private6th-pf-pf-pf-pf-and-bagging`
- External component reference: CV 7.8 / Public 7.88 / Private 7.78
- Late submission attempts allowed: 1
