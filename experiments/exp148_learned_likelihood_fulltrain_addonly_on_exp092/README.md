# exp148_learned_likelihood_fulltrain_addonly_on_exp092

## 目的

exp127 では exp112 learned likelihood feature を exp092 に add-only した shared-row 評価が改善したが、155 wells subset に限定されていた。exp145 で full train / raw test の target-free learned likelihood feature generator が通ったため、本実験では exp145 full-train cache を使って exp092 全行 surface 上で add-only 特徴量を再評価する。

## 状態

Kaggle train v1 / inference v7 完了。GPU inference v7 は Public LB 7.960。CPU runtime inference v1 ref `54183122` は Public LB 7.921 で、現行の ML route submitted anchor。

## 仮説

exp145 の full-train learned likelihood confidence features を exp092 surface に add-only すれば、exp127 subset で見えた候補信頼度 signal が full-row CV でも改善として残る可能性がある。

## 変更点

- 親: `exp092_u_projection_correction_disagreement_fullrun`
- feature 親: `exp145_learned_likelihood_rawtest_feature_generator_parity`
- control: 再学習しない。保存済み exp092 metrics を historical baseline として参照する。
- variant: `learned_likelihood_confidence_addonly`
- route: `ml_model`

exp092 の U-projection correction / disagreement feature と residual target は固定し、exp145 の probability、expected-error、margin、entropy、candidate TVT disagreement、multi-observation score、weighted TVT proxy を add-only feature として追加する。

## 検証方針

GroupKFold 5 folds を well group で実行し、`learned_likelihood_confidence_addonly` だけを学習する。pooled RMSE、distance bucket、by-well regression、feature importance を確認し、保存済み exp092 `lgb1` CV 9.322479896 / Public LB 8.350 を historical baseline として参照する。

## 所見

Kaggle train v1 は `learned_likelihood_confidence_addonly` だけを 15 boosters で実行し、`lgb_mean` pooled RMSE 8.50128118189582 を得た。保存済み exp092 `lgb1` CV 9.322479895503927 との historical 比較では -0.821198713608107 改善した。

Feature join coverage は 3,783,989 rows / 773 wells で pass、drop rows / wells は 0。Inference v7 は current-test learned likelihood features を生成し、14,151 rows の `submission.csv` を生成した。fallback rows は 0。submission SHA256 は `45a8b1787fd80213c158d9af04fb596750d8025802d1328ab9d075432bcb6e4b`。

v5 submission rerun は public raw-test cache 依存により hidden test で `Notebook Threw Exception` になった。v7 で hidden-safe generation に直し、submit-check は PASS。GPU inference v7 の提出 ref `54124882` は Public LB 7.960 で、exp092 Public LB 8.350 から改善した。

その後、ユーザー確認により CPU runtime inference v1 の提出 ref `54183122` を exp148 CPU runtime として扱う。Public LB は 7.921 で、GPU inference v7 の 7.960 と exp193 の 7.946 を上回るため、現行の ML route submitted anchor は exp148 CPU runtime inference v1 / Public LB 7.921 とする。Control 再学習なしのため同一実行 ablation ではないが、ユーザー判断により追加 trust audit は不要とする。

## 実行状態

Train / inference は Kaggle Notebook 上で完了。提出済み。
