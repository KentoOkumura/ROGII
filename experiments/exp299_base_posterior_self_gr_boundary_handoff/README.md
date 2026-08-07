# exp299_base_posterior_self_gr_boundary_handoff

## 状態

- ルート: `ensemble`
- 状態: Kaggle private CPU version 2完了、train-side guard FAILでbranch close
- CV / LB: `11.789577561` / 未提出
- 作成日: 2026-07-20
- 親実験: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- negative/reference: `exp296_exp223_self_gr_known_tvt_support_gate`
- base-HMM parity reference: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`

## 仮説

exp296はrange外self-GRをexact 0にした一方、range内stateの正boostを残したため、known range内への相対priorを作った。Type Well-only exact HMM posteriorを先に凍結し、境界ではrow全体をself-GR neutralへhandoffし、range内ではsupport総量を変えない条件付きlikelihoodとしてself-GRを使えば、range外参照禁止と境界通過を両立できる。

## 変更点

1. Pass Aでexp223と同じType Well HMMをself-GRなしで実行する。
2. candidate stateがknown range外ならself-GR contributionをexact 0にする。
3. base posterior meanが境界へ近づくほど、既存sigma 12 ftを使ってrow全体のself-GRを0へfadeする。base meanがrange外または境界上なら全state exact 0にする。
4. range内で使うself-GRはbase posterior条件付きで正規化し、inside/outsideの総尤度比を変えず、inside stateの順位だけを変える。
5. Pass B posteriorやfinal predictionをgateへ戻さない。

正確な数式は[handoff_contract.md](handoff_contract.md)を正とする。

## exp296との違い

exp296はstate-wiseに`inside=正boost / outside=0`を作った。exp299はoutside exact 0を維持しつつ、境界ではinside側もrow全体でneutralへhandoffする。さらにconditional normalizationでsupport総量への追加priorを消す。paddingやsoft outside supportではない。

## 検証方針

- score: official train unknown suffix 3,783,989 rows / 773 wells。
- primary control: saved exp223 `hmm_selfgr_boost_only_a070_c100`、RMSE 11.349942946、decompressed SHA `0eb48b55...aa0c`。
- negative comparison: exp296 RMSE 12.159749140。
- Pass A parity: saved exp209 exact HMM、decompressed SHA `8e2f4236...7ae5`。
- reporting fold: stable SHA256 well hash 5分割。学習foldは0。
- readout: overall、fold、true-TVT inside/outside、upper-boundary 0-12/12-24/24+、distance、hidden-like 2面、by-well、step delta、base/controller calibration。
- technical/performance gateを全必須とし、1項目でもFAILなら救済せず閉じる。

## 実行規模

- scientific variant: 1。
- internal passes: Pass A base-only + Pass B handoff。
- planned HMM well-runs: `773 + 773 = 1,546`。
- LightGBM config / trained fold / booster: `0 / 0 / 0`。
- GPU / parent-control retraining: `0 / 0`。
- v1は約11.1時間後にparity比較bugで停止した。修正版v2は`22,481.454 sec`（約6.245時間）で完了した。承認は消費済みで、v3は未承認。

## 実行入口

- 正規train Notebook: `exp299_base_posterior_self_gr_boundary_handoff_train.ipynb`
- inference Notebook scaffold: `exp299_base_posterior_self_gr_boundary_handoff_inference.ipynb`
- 別名train候補: `exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_train.{py,ipynb}`
- 別名fail-closed inference候補: `exp299_base_posterior_self_gr_boundary_handoff_compact_selfcontained_inference.{py,ipynb}`
- 別名compact train候補を正規train Notebookへ採用した。正規inference Notebookはtemplateのままで、fail-closed候補はraw testを読まず常に停止する。
- `implementation=true`、`canonical_train_notebook_adopted=true`、`run_train=true`、`run_variant=true`、`kaggle_cpu_push_approved=false`（v2開始後に承認を消費）、`run_inference=false`、`write_submission=false`。

## 結果

| メトリック | 値 |
| --- | --- |
| 設計 | 確定 |
| 実装 | compact候補・専用tests完了 |
| Kaggle train | version 2 COMPLETE（id_no `127957958`） |
| exp209 base parity | max / mean abs `0.0 / 0.0 ft`、PASS |
| RMSE | candidate `11.789577561` / exp223 `11.349942946` / delta `+0.439634615 ft` |
| reporting folds | 改善 `0/5` |
| technical / performance gate | `24/25` / `2/11` PASS、総合FAIL |
| Public / Private LB | 未提出 |

## 所見

exp299はexp296より`-0.370171579 ft`回復したが、exp223より`+0.439634615 ft`悪化した。inside/outside、upper-boundary、1000+、p95、worst-wellも悪化しており、base posterior handoffとconditional normalizationの一体policyは不採用。technical唯一のFAILはrow gate maxの`2.9e-15`丸め超過だが、performance 9/11 FAILの結論には影響しない。

## 次

[Kaggle version 2](https://www.kaggle.com/code/kentookumura/exp299-base-post-self-gr-boundary-handoff-train)をnegative resultとして確定し、事前固定fail actionどおりbranchを閉じる。handoff/fade/normalizer/alpha/clip/support/threshold救済、version 3 repush、inference、submissionは行わない。
