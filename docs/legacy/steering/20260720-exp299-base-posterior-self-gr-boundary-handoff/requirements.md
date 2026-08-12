# 要件

## 依頼

exp296で確認したstrict candidate-state support gateの境界障害を解消しつつ、base-only HMMがknown TVT range外を予測する行・candidate stateではself-GRを参照しない新実験を実装する。2026-07-20の設計確定後、ユーザーの明示的な実装指示を受け、別名compact self-contained train候補、fail-closed inference候補、契約tests、設定・記録更新までを行う。正規Notebook採用、Kaggle実行、実推論、submissionは行わない。

## 仮説

exp296はoutside stateを0にしてinside stateの正boostだけを残したため、known range内への相対priorを作り、未来TVTが`known_tvt_max`を越える際のmode transitionを妨げた。self-GRを「base posteriorが安全にrange内にあるときの条件付きstate順位付け」に限定し、境界ではrow全体をneutralへhandoffすれば、outside参照をexact 0にしたままexp223のinside signalを保持できる。

## Assumption

target-freeなType Well exact HMM posteriorは、self-GRを使う前に境界接近・range外を判定するcontrollerとして利用できる。base posteriorは完全な正解判定器とは仮定せず、self-GRがinside/outsideの総尤度比を変更しないための参照分布としてだけ使う。

## 制約

- 実験名は`exp299_base_posterior_self_gr_boundary_handoff`、Routeは`ensemble`とする。
- primary parentは`exp223_joint_typewell_self_gr_hmm_likelihood_probe`、negative/referenceは`exp296_exp223_self_gr_known_tvt_support_gate`、base-HMM parity referenceは`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`とする。
- exp223 best `hmm_selfgr_boost_only_a070_c100`のHMM grid/transition、Type Well emission、descriptor、anchor、top-k、surface、quality、alpha `0.07`、clip `1.0`、Gaussian sigma `12 ft`を固定する。
- Pass Aでself-GRなしのType Well exact HMM posteriorを作る。unknown-suffix true TVT、error、exp223/296 predictionをcontrollerに使わない。
- supportはfinite visible-prefix `TVT_input`のinclusive `[known_tvt_min, known_tvt_max]`とし、outside candidate stateのself-GR contributionはexact `0.0`とする。
- row handoffはPass Aのbase posteriorだけから作る。base posterior meanがsupport外または境界上なら、そのrowのself-GR contributionを全stateでexact `0.0`にする。
- 境界へのfade幅は新しいgridにせず、exp223既存`gaussian_sigma_tvt=12 ft`を1倍で再利用する。
- range内でself-GRを使う場合も、base posterior重み付きconditional normalizationによりsupport全体の尤度massを変えず、support内stateの相対順位だけを変える。
- final variant prediction、variant posterior、true TVTをgateへ戻さない。Pass AからPass Bへの一方向依存にする。
- scientific variantは1本。Pass Aはcontroller生成用の内部base passであり、比較controlを再学習するvariantではない。
- planned HMM well-runsはPass A 773 + Pass B 773 = 1,546。LightGBM config / trained fold / booster、GPU、parent-control retrainingは`0 / 0 / 0 / 0 / 0`。
- 保存済みexp223 controlとexp296結果を比較に使い、再実行しない。Pass Aはexp209 saved exact-HMM predictionとのparityを照合する。
- handoff式、fade幅、normalizer、alpha、clip、window、top-k、support padding、thresholdの救済gridは禁止する。
- Stage 0 FAILならbranchを閉じ、inference/submissionへ進めない。PASSでもexp209 HMM/likPF blend 10.269696以下、raw-test-safe設計、別承認なしにinferenceへ進めない。
- 再現性は`docs/06_reproducibility.md`に従い、pushed source/config、raw input、saved controls、base posterior、row gate、conditional contribution、prediction、metricsのSHAを記録する。gzipはdecompressed content SHAを主証拠にする。

## 受け入れ基準

- steering 3文書、`handoff_contract.md`、`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`に未記入placeholderがない。
- Pass A / Pass B、support、row handoff、conditional normalization、一方向依存が数式で固定されている。
- outside candidate contribution exact 0、base mean outside/boundary rowの全state contribution exact 0、conditional support-mass preservationがtechnical gateに含まれる。
- 1 scientific variant / 2 HMM passes / 1,546 well-runs / 0 booster / parent control再実行0が明記されている。
- exp223/exp296/exp209との比較、fold/scope/hidden-like/by-well/worst-well performance gate、FAIL-closeが実装前に固定されている。
- `implementation=true`、`run_variant=false`、`kaggle_cpu_push_approved=false`、`run_inference=false`、`write_submission=false`で停止する。
- 正規train/inference Notebookと`settings.py`はtemplate scaffoldのまま維持し、別名compact self-contained train/inference候補と専用testsだけを追加する。Kaggle package/outputは作らない。
- compact train候補はPass A posterior/mean/grid/input identity SHA、row gate、conditional contribution、Pass B prediction、schema、metricsの記録経路を持つ。
- inference候補はraw testを読まず常にfail-closeし、実推論や提出ファイル生成を実装しない。
- `make validate-exp EXP=exp299_base_posterior_self_gr_boundary_handoff`とproject validationがPASSする。

## 次

当初は実装と静的検証で停止した。その後、ユーザーの別指示で正規train Notebook採用、Kaggle CPU version 1、parity修正後のversion 2再実行を承認された。version 2はcandidate RMSE `11.789577561`、exp223比`+0.439634615 ft`、改善0/5 foldsでguard FAIL。停止条件どおりbranchを閉じ、version 3、inference、submissionへ進めない。
