# exp516 6位解法 `pfA × twGR` LATE SUBMIT再現監査

Kaggle discussion 733226と公開submission Notebookを一次資料として、6位解法の単体Particle Filter componentを`faithful`に再現する。

対象は`pfA × typewell GR`、GR-free物理anchor、learned emission、600 particles、32 seeds、whole-interval ancestral smoother、seed likelihood soft averageである。91候補のrow-level bagging、TCN、GBM、de-shrinkはユーザー確認により対象外とした。

この実験はコンペ終了後の再現監査である。技術gateを通った固定版を1回だけ提出し、Notebook titleとsubmission messageに`LATE SUBMIT`を明記する。6位チームの最終スコアや正式順位と、exp516のlate-sub scoreを混同しない。

## 状態

完了。Kaggle T4 x2のversion 2公開commit runが完走し、submit-check PASS後に固定版を1回だけ`LATE SUBMIT`した。ref `55326266`のhidden rerunも完走し、Public `10.056` / Private `8.552`。

## 仮説

公開された`pfA × twGR` componentを、GR-free anchor、learned emission、600 particles、32 seeds、full ancestral smoothingを省略せず再生成すれば、作者報告の単体componentと比較可能なlate-sub scoreを得られる。

## 検証方針

公開Notebook、v96 config、5本のencoder checkpointをSHA固定する。current runtimeのtrain/testからanchorとsimilarityを再生成し、`pfA × twGR`のsmoothed meanだけをsample submissionのID・順序・target列へ1対1整列する。固定版を`LATE SUBMIT`として1回だけ提出し、結果を再調整には使わない。

## 基準と変更点

基準はdiscussion 733226と公開kernel id_no `126919690`のfinal submission source。そこから`pfA × twGR`だけを抽出し、他の5 PF bank、4 representation、candidate-curve NN、TCN、GBM、de-shrinkを除外した。PF/anchor/emissionの数値契約は変更していない。

## 所見

公開実装から単体branchを抽出して完走したが、作者報告stage 2-4 standalone Public `7.88` / Private `7.78`に対し`+2.176 / +0.772`悪化し、報告スコアは再現しなかった。writeupのstage 2-4とfinal v96 sourceの契約同一性は未証明なので、91候補final systemやPF family全体は閉じない。

- Route: `pf_beam`
- Status: `completed`（作者報告スコアは未再現）
- Source: `k256net/public20th-private6th-pf-pf-pf-pf-and-bagging`
- External component reference: CV 7.8 / Public 7.88 / Private 7.78
- Late submission attempts: 1 / 1（追加提出なし）
- Public commit output submission SHA256: `feee82f8ec8d24390fe0478a983fef42054c127487ac793612ead3ab61fc080c`（hidden scored output SHAはAPI非公開）
