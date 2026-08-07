# exp353_typewell_group_quality_feature_preflight セッションノート

## 目的

旧exp314のadd-only ML feature仮説を、exp311/313のpromotionから分離し、
0-booster preflightを必須にした独立後継として設計固定する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle CPU version 1完了、固定Stage 0 gate FAIL、branch closed
- CV: Stage 0 5/8 checks PASS、総合FAIL
- LB: まだなし

## コマンドログ

- 2026-07-23: ユーザー承認により、設計確定までの独立後継scaffoldを作成した。
- 2026-07-23: 実装、Notebook編集、Kaggle実行、Stage 1学習は行っていない。
- 2026-07-23: exp352はsame-group平均`+0.381540 GR API`だったがworst
  `+12.914716 GR API`でFAILした。ユーザーの「平均で改善しているのなら次に進む」指示を、
  exp352直接補正を採用せずexp353 Stage 0実装とKaggle CPU実行へ進む承認として受領した。
- 2026-07-23: compact self-contained train/inferenceと専用contract testsを実装し、
  compact trainだけを正規train Notebookへ採用した。
- 2026-07-23: target-free行数監査で水平well全行が`5,092,255`、exp148 score surfaceの
  `TVT_input`欠損行が`3,783,989`と確認した。fold row weightは後者だけを使う。
- 2026-07-23:
  `make prepare-kaggle-notebooks ... --run-on-push --strict`をPASSし、
  config/package/bootstrap SHA一致を確認した。
- 2026-07-23:
  `make push-kaggle-train EXP=exp353_typewell_group_quality_feature_preflight`で
  `kentookumura/exp353-typewell-quality-preflight-train` version 1をpushした。
- 2026-07-23: pull metadataでid_no `128362932`、private CPU、GPU/internet off、
  exp065/exp148 kernel sourcesを確認した。
- 2026-07-23: Kaggle logsで正常完了と固定gate FAILを確認した。
- SHA実ファイル確認のため小さいStage 0生成物だけを
  `/tmp/exp353-kaggle-output-v1`へ取得した。repositoryには生成物を保存しない。

## 変更点

- 6列schema、outer-fold join、global fallbackを固定した。
- Stage 0は1 preflight + 1 stable shuffle / 5 folds / model・booster各0。
- Stage 1予約は1 variant × 3 configs × 5 folds = 15 boosters、control再学習0。
- Stage 0/1、raw-test regeneration、inference、submissionを別承認境界にした。
- exp148 GroupKFoldはscore-row countを使うsklearnのlargest-first/lightest-fold規則を
  self-containedに再現する。保存errorはfeature manifest freeze後だけjoinする。
- group qualityはexp311と同じ固定Huber/identity-shrink calibrationを独立再計算し、
  exp311保存priorやpromotion decisionを入力にしない。

## Kaggle train push前ガード

- Stage 0 primary preflight: 1。
- negative control: stable group-label shuffle 1。
- reporting folds: 5。
- LightGBM config / trained fold / booster: `0 / 0 / 0`。
- 親exp148 control再学習: 0。保存済み`lgb_mean` by-well OOF errorだけを読む。
- Stage 1予約: 1 variant × 3 configs × 5 folds = 15 GPU boostersだが、未承認で実行flagはfalse。
- inference / submission: 0 / 0。

## 再現性メモ

- seed policy: Stage 0 RNGなし。Stage 1はexp148 seed/fold/threadを継承する。
- stochastic components: Stage 1 GPU LightGBMのみ。現在は未承認。
- CPU/GPU runtime: Stage 0 Kaggle CPU、GPU/internet off、30分上限。Stage 1 GPU予約。
- Kaggle kernel id: `kentookumura/exp353-typewell-quality-preflight-train`。
- Kaggle kernel version / id_no: `1 / 128362932`。
- exp065 membership raw SHA:
  `dcda8588cc1dd9261bafae7de00c890393e38b8a0ca0eb86fbba18a2cffc4a50`。
- exp148 summary raw SHA:
  `18d4491f70629d49bb9d9f4ca9c77ca8479d69df6f8c715d317c0e7eaa56ea86`。
- exp148 by-well raw SHA:
  `3fd88c32de63c5df14756ef4ea55f0fb3dd8b050eaa2aa87b97c3e7bd5f2dc87`。
- Kaggle fold manifest content / raw SHA:
  `9fd83e273d994762068ed5182c8e61397b78f51b4cd9489113dea4240edc1515` /
  `1d3aa5d689a5a010841715e88cca0805017bfbee3ef5c0355823905a6a35c74a`。
- host previewは同じ773行・fold・row countと完全一致した。host pandas dtypeを含む
  in-memory content SHAは`a8274038...96df`だったため、Kaggle runtime SHAを正とする。
- feature manifest freeze SHA:
  `6a90ee2aa35029c4910e93c7476aa5cff1cce82af0e17fce41d7e34e85e256fe`。
- group prior content / raw SHA:
  `7533c94abfeef9e804cdd1bf9d4cd4af774bac368173e1ac6dce44b6304a1cca` /
  `b1bd4894baf93884df913a6febc36f7b161858e2ee2b1ae1a3b916e2f3a00e65`。
- feature manifest content / raw SHA:
  `b1bcb4729e78d24b2e0dc596e23dd83b14bd5b7e8442bd188e2bfef2894ac852` /
  `420a2a3bc8e8b30af053a8ebd38396a3557f975d43b0b694032f0831dc1081c1`。
- error association content / raw SHA:
  `a5322178bbedb2c2f5c39b95dbb05b99c851b045b90097c4fa5e69da0936d128` /
  `ead37d9a767ce759b7622dc0086b02b22dc2d3bf1ffee9d8ad60ea447a5a51d3`。
- gate / summary raw SHA:
  `5d1592177cb55ce977e40e18dad0ff62a06eeefe7c8b52150935b7fe5ebf91cb` /
  `f395de2a5a4cc2ca6b57472210fe87681daa9bc032d7a0ceac15d58949a8aa35`。
- model manifest / model SHA: Stage 0非該当。Stage 1時は15 model。
- prediction SHA: Stage 1 OOFのみ。現在非該当。
- submission SHA: inference/submission未承認のため非該当。
- rerun check: 未実行。

## Kaggle Stage 0結果

- diagnostic runtime: `112.107959 sec`。
- 3,783,989 score rows / 773 wells / native groups 54。
- exact group coverage: `0.9805950841`、PASS。
- fallback fraction: `0.0194049159`、PASS。
- 6 feature finite: 100%、PASS。
- outer-valid truth before feature freeze: `0`、PASS。
- real residual sigma vs exp148 well-RMSE Spearman:
  `0.0061344681`、閾値`0.15`に届かずFAIL。
- fold Spearman:
  `0.044483 / 0.015301 / -0.104195 / 0.021856 / 0.098298`。
  正方向4/5 foldsはPASS。
- q4-q1 exp148 well-RMSE:
  `+0.2027010060 ft`、閾値`+0.25 ft`に届かずFAIL。
- shuffle Spearman:
  `0.0653007982`、real-minus-shuffleは`-0.0591663301`でFAIL。
- 8 checks中5 PASS、総合FAIL。

## 実装SHA

- final config / metrics:
  `4f11bbf37757686789241de01fa3d9f0d0d3b777af8b7e7539a19aa0b99537c7` /
  `9ef28d9f7a38a702f5f18576dda1490dcd485e5223ef5b2c49faab74ef42d864`。
- compact train source:
  `90ed288ac28eac470975721295ddb1c7b3924163c61f7ad1e576ee552b2d43b2`。
- compact inference source:
  `b0774ab5211a7e7dbcf59ca6230accccaf5c5f73d114499f4c3cd275f6537a0f`。
- contract test:
  `120afbec6accc14864a152bff8366eaf12ccca4982e7e5659d0f23c93b339998`。
- canonical train Notebook:
  `5edb585617e704593dcee658a216aa9917158795cd16c6e3eda018da86c62c54`。
- execution package config:
  `f395cd0ce92ae9666f660d5ab442120d54904132a5f36552eb761c3048829a0b`。
- execution package metadata / Notebook:
  `8c6fd00927bafd41d4fc30e9db2996fb96f6a56606e9f76cc4fd45dbafbac289` /
  `df0a89a9e43a5dcd87447bde43ba4d96193fc1dfeb106ba866a64afbcc4ffa6a`。

## 静的検証

- compact train/inferenceの`py_compile`、Ruff
  `F821,F401,F841,E722,E501`: PASS。
- exp353専用contract tests: 6 passed。
- exp311/352/353関連tests: 18 passed。
- compact train/inferenceのJupytext変換と`--test`: PASS。
- `make validate-exp EXP=exp353_typewell_group_quality_feature_preflight`: PASS。
- `make validate-template`: PASS。
- repository全体: 705 passed / 5 skipped / 2 failed。2失敗は既存exp296の
  完了後status/run flagと旧test期待値の不一致で、exp353専用6件はすべてPASSした。

## 次のアクション

1. 列選択、group/fallback/閾値救済、再実行を行わずbranchを閉じる。
2. Stage 1の15 GPU boosters、raw-test再生成、inference、submissionへ進まない。
3. exp352のdirect prior不採用と旧exp314閉鎖を維持し、同family救済backlogを追加しない。
