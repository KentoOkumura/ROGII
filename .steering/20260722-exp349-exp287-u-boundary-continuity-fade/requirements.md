# 要件

## 依頼

公開Notebook `phuongncn/kdrill-f594-ucont8` のうち、`U = TVT + Z` の既知prefix／未知suffix境界差を最大8 ftだけ打ち消し、MD方向へ240 ftで指数減衰させるtarget-free後処理だけを、現行ML submitted anchor `exp287_fold_safe_formation_74_addonly_on_exp264` の保存済みOOF上で監査する実験として設計固定する。

初回依頼の範囲は、バックログ、steering、実験scaffold、config、設計記録の作成までとした。補正コード、Jupytext source、正規Notebook実装、Kaggle package/push/run、raw-test inference、submissionは初回時点では作成・実行しない契約だった。

## 2026-07-22 実装承認

ユーザーの`exp349を実装してください`により、事前登録済み仕様を変更しないcompact self-contained train候補、fail-closed inference候補、synthetic contract testsの実装を承認した。既存canonical Notebookの上書き、Kaggle package/push/run、raw-test inference、submissionは引き続き承認対象外とする。

## 2026-07-22 実行承認と2026-07-23完了確認

ユーザーの`実行してください`により、正規train Notebook採用とKaggle private CPU Stage 0を承認した。version 1はfreeze前のpandas型互換technical errorで停止し、科学契約を変えない修正だけを行ったversion 2が完了した。全technical gateとscientific gate 11/12件はPASSしたが、pooled改善`0.001611295 ft`が下限`0.020 ft`に届かず、`FAIL_CLOSE_NO_RESCUE`で終端した。raw-test inferenceとsubmissionは承認も実行もしていない。

## 仮説

exp287の未知suffix先頭で、親予測から得る `U_pred = TVT_pred + Z` と既知prefix末端の `U_last = TVT_input_last + Z_last` に局所的なdatum段差が残っている。境界差の符号と大きさだけを現在wellの既知情報から測り、固定式で境界直後を補正して奥へ減衰させれば、親の遠方形状をほぼ維持したまま境界近傍RMSEを改善できる。

## 制約

- Route: `ml_model`。最終予測の親はexp287のdownstream LightGBMであり、PF/Beam/HMMを本実験で再生成・直接blendしない。
- 親実験: `exp287_fold_safe_formation_74_addonly_on_exp264`。保存済みOOF SHA `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`、3,783,989 rows、773 wells、CV `8.136708220359452`を固定する。
- 比較対象は保存済みexp287 OOFだけとし、exp264、exp334、公開6.594 anchorを同一実験内の選択候補にしない。
- 変更は `cap=8.0 ft`、`tau=240.0 MD-ft` のalways-on U境界fade 1 variantだけとする。cap、tau、符号、減衰形、適用率、well別thresholdを探索しない。
- 公開Notebookのcontact reconstruction、same-well overlap、formation列、PF/Beam、visible-prefix profile、model-package correction、branch hedgeは移植しない。
- 生成時に使える列は、保存済み親予測と現在wellの`MD`、`Z`、`TVT_input`、ID／well／row identityだけとする。unknown suffixのtrue TVT、target、error、oracle、same-name train well、training-only formation列を使わない。
- target-free candidate、境界診断、config、input identityをSHA freezeしてから、評価用`actual_tvt`、outer fold、hidden-like assignmentをlate joinする。
- Stage 0予定量は1 deterministic postprocess variant、5 reporting folds、trained fold/model/config/LightGBM booster/PF/Beam/HMM/control再学習/GPUすべて0。実装承認は取得済みで、canonical採用とKaggle実行には別承認を必要とする。
- Stage 0 FAIL後にcap/tau grid、gap threshold、well selector、blend weight、別親、far-only化、Public LB probeで救済しない。
- inferenceは全technical/scientific gate PASSと別承認後に、同じexp349内でのみ検討する。新しいinference専用expは作らない。
- 再現性は`docs/06_reproducibility.md`に従い、親OOF、raw-horizontal manifest、target-free prediction、diagnostic、metrics、package/kernel SHAを記録する。

## 受け入れ基準

- `.steering/20260722-exp349-exp287-u-boundary-continuity-fade/`と`experiments/exp349_exp287_u_boundary_continuity_fade/`が存在し、仮説、式、入力境界、gate、禁止事項、実行量が記録されている。
- 実装時点では`implementation_complete_no_run`でfail-closedとし、実行承認後だけStage 0 flagsを有効化した。完了後はrun flagsをfalseへ戻し、結果を`kaggle_cpu_v2_complete_fail_close_no_rescue`として固定する。
- wellごとの式を次で固定している。
  - `U_last = TVT_input[last_visible] + Z[last_visible]`
  - `gap_U = parent_pred[first_hidden] + Z[first_hidden] - U_last`
  - `move(d) = -clip(gap_U, -8, 8) * exp(-d / 240)`
  - `candidate(d) = parent_pred(d) + move(d)`
- finite `TVT_input` prefixの直後にcontiguousなNaN suffixが続き、OOF IDがそのsuffixへ1対1対応し、`d = MD_hidden - MD_last_visible > 0`であることを全773 wellsで要求する。1 wellでも違反したらskipせずfail-closedする。
- technical gateは親SHA／row／well／ID／CV parity、raw schema、truth-before-freeze 0、finite、最大移動8 ft、補正絶対値の単調減衰、境界U gap非増加、prediction／diagnostic SHAをすべて要求する。
- scientific gateは親比pooled RMSE `>=0.020 ft`改善、改善fold `>=4/5`、0--240 ft RMSE `>=0.050 ft`改善、240--480 ft `<=+0.020 ft`、480--1000 ft `<=+0.010 ft`、1000+ `<=+0.005 ft`、hidden-like 2面各`<=+0.020 ft`、by-well median delta `<=0`、p95 delta `<=+0.10 ft`、worst-well delta `<=+0.50 ft`をANDで要求する。
- `|gap_U|`の固定bucket `[0,1) / [1,2) / [2,4) / [4,8) / [8,+inf)`をtruth join前に確定し、coverage、移動量、late-join後のRMSE deltaを診断するが、bucketで発火や採否を変えない。
- compact self-contained train/inference候補、正規train Notebook、実験固有tests、Kaggle package/output evidenceが作成済みである。hidden-test predictionとsubmissionは未作成であることを明記する。

## Assumption

- train raw horizontal wellsの`TVT_input` finite prefix／NaN suffix構造がcurrent testと同じ境界契約を表すと仮定する。Stage 0 preflightで全wellを検証し、成立しなければ科学評価へ進まない。
- 公開Notebook記載の495-well選択／773-well再監査は外部主張としてのみ扱い、本実験の採用根拠にはしない。
