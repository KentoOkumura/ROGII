# 要件

## 依頼

`exp223_joint_typewell_self_gr_hmm_likelihood_probe`を完全固定し、HMM candidate state `grid[j]`がvisible prefixの`[known_tvt_min, known_tvt_max]`外にある場合だけself-GR boostを厳密に0にする1 variantの新実験を設計する。

初回設計ターンの作業範囲は、`KAGGLE_DIRECTION.md` backlog、steering 3文書、実験scaffold、support gate契約、設定、評価・実行・停止条件の確定までとした。その後2026-07-19のユーザー指示で、別名compact self-contained Jupytext train候補とcontract testsを実装し、続く実行指示で正規Notebook採用とKaggle CPU pushを行った。version 3で完走後、performance guard FAILによりbranchを閉じた。inference、submissionは行っていない。

## 仮説

exp223のdescriptor-motif self-GRはknown prefixに存在するTVT stateにだけ経験的根拠を持つ。candidate stateがknown TVT range外のときself-GR contributionをneutralにすれば、support内の有効なmotif boostを保ったままwrong-depth attractionとworst-well regressionを減らせる。

## 制約

- Route: `ensemble`。exp209 Type Well HMM emissionとsame-well self-GR motif emissionの双方がposterior生成に寄与するため、このrouteを使う。
- primary parentは`exp223_joint_typewell_self_gr_hmm_likelihood_probe`、negative/referenceは`exp225_state_known_tvt_self_gr_hmm_emission`とする。
- exp223 best variant `hmm_selfgr_boost_only_a070_c100`のHMM grid、transition、Type Well GR emission、GR補間、descriptor、anchor selection、top-k、surface、centering、quality、alpha、clip、modeを変更しない。
- 変更は、exp223でcenter・normalize・positive clipしたself-GR boostへcandidate-state support maskを掛ける処理だけとする。
- `known_tvt_min/max`は同一wellのvisible prefixにあるfinite `TVT_input`全行から計算する。unknown suffix true `TVT`、予測値、Type Well TVT range、matched top-k anchorだけから計算しない。
- supportはinclusiveな`known_tvt_min <= grid[j] <= known_tvt_max`とする。padding、tolerance、nearest-distance threshold、hole filling設定、support soft weightを追加しない。
- support外stateはself-GR contributionをbitwise `0.0`にする。base Type Well emission、transition、posterior stateはsupport外でも有効であり、最終予測TVTをknown rangeへclipしない。
- support内stateのself-GR boostはmask前exp223 boostとbitwise一致させる。support内だけで再centering・renormalizationしない。
- final predicted TVTをgate入力に使わない。gateはdecode前のcandidate stateにだけ適用する。
- exp225のstate-known `TVT_input -> GR`曲線、curve smoothing、GR sigma、state-valid qualityは使わない。
- variantは1本だけ。alpha、clip、sigma、range padding、support定義のgridを作らない。
- 保存済みexp223 feature cache / metrics / by-wellをcontrolとして参照し、exp223 controlを再実行しない。
- 将来の実行規模は新variant 1、773 wells、773 HMM well-runs、LightGBM config / trained fold / booster `0 / 0 / 0`、GPU 0、親control再学習0とする。
- compact候補の実装、正規Notebook採用、Kaggle CPU pushは各ユーザー承認に基づき完了した。
- Stage 0 FAIL後のsupport/alpha/clip/window/top-k/threshold救済grid、Stage 2、inference、submissionは禁止する。
- 再現性は`docs/06_reproducibility.md`に従い、pushed source/config、raw input、saved control、support mask、prediction、metricsのSHAを記録する。gzipはdecompressed content SHAを主証拠にする。

## 受け入れ基準

- steering 3文書、`support_gate_contract.md`、`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`に未記入placeholderが残っていない。
- support maskの入力、数式、適用順序、inclusive boundary、neutral fallback、inside-support parity、outside-support zeroが明記されている。
- exp223からの変更がcandidate-state mask 1点だけで、exp225との非重複性が明記されている。
- 実行は1 variant / 773 HMM well-runs / LightGBM config・trained fold・booster `0/0/0` / control再実行0で完了している。
- 保存済みexp223 controlを再実行しないことと、expected decompressed SHA `0eb48b5516276b0ab7b2191a52a39ebb89d9997363cc7839ede519c7863baa0c`が固定されている。
- technical gate、primary performance gate、subgroup/worst-well gate、FAIL-closeが実装前に固定されている。
- `make validate-exp EXP=exp296_exp223_self_gr_known_tvt_support_gate`とproject validationが通る。
- 正規train Notebookは承認後にcompact sourceから採用済み。inferenceロジック、trained model、submissionは存在しない。

## 次

Kaggle CPU version 3はtechnical 12/12 PASS、performance 2/10 PASS、pooled RMSE delta `+0.809806 ft`で完了した。FAIL-closeを適用し、救済variant、inference、submissionへ進めない。
