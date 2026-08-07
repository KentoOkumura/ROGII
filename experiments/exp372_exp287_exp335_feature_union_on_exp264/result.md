# exp372_exp287_exp335_feature_union_on_exp264 結果

## 状態

2026-07-25、Kaggle T4 train version 2で15/15 boosterを完走した。technical gateは
PASSしたが、固定incremental utility gateとtail promotion gateをFAILした。
事前契約どおりtrain branchは`train_complete_guard_failed_closed`として閉じた。
その後、2026-07-25のユーザー明示overrideによりsaved-model CPU inferenceだけを
別途実行対象へ戻した。推論workflowではcompetition submitを行っていない。
後にユーザーがscoring完了を連絡し、Kaggle上の完了提出を確認した。

## 仮説

corrected exp264へ、exp287 formation 74列とexp335 signed residual 23列を同時に
add-onlyで追加すると、各単独親のfold相補性をdownstream LightGBMが利用し、
best standalone CVを上回る。

## 固定設定

- Route: `ml_model`
- 親: `exp264_exp263_candidate_confidence_dual_selector`
- 特徴: `clean273 + saved74 + formation74 + signed23 = 444`
- 検証: corrected exp264 outer 5 group fold
- メトリック: RMSE
- seed: 42
- 学習量: 1 variant / 3 configs / 5 folds / 15 GPU boosters
- control・単独親・selector再学習: 0
- formation / signed train feature再生成: 0

## 変更点

単独親の仮説、feature生成、selectorは変更せず、保存済みのformation74とsigned23を
同じouter foldのdownstream feature matrixへ同時に追加することだけを変更した。

## 比較基準

| 実験 | CV | Public LB | train-side guard |
| --- | ---: | ---: | --- |
| exp264 | 8.460811238 | 7.562 | FAIL |
| exp287 | 8.136708220 | 7.530 | FAIL |
| exp335 | 8.146107756 | 7.517 | FAIL |

Public LBはprovenanceとしてのみ記録し、本実験の特徴・parameter・gate選択には使っていない。

## 結果

| メトリック | 値 |
| --- | ---: |
| pooled CV RMSE | 8.071563865 |
| exp287比改善 | 0.065144355 ft |
| 完了booster | 15 / 15 |
| Public LB | 7.587 |
| Private LB | 未確定 |

fold ensemble RMSEと各foldのbest standaloneとの差は次のとおり。

| fold | union RMSE | best standalone | 差 |
| ---: | ---: | --- | ---: |
| 0 | 7.822580 | exp335 | -0.177982 |
| 1 | 8.676376 | exp287 | +0.420537 |
| 2 | 7.662126 | exp335 | -0.010559 |
| 3 | 7.847103 | exp335 | +0.014383 |
| 4 | 8.306610 | exp287 | -0.043016 |

固定`<=+0.02 ft`条件を満たすfoldは4/5で、fold条件自体はPASSした。

## ゲート判定

### Technical gate: PASS

- 3入力manifestと全partition SHA、formation logical SHAを検証した。
- 3,783,989 rows / 773 wells、`id/well/fold/role` alignmentを確認した。
- schema freeze前truth/error読込0、444 unique features、finite matrixを確認した。
- 15 unique model slotsと全15 model SHAを確認した。

### Incremental utility gate: FAIL

- pooled CV上限`8.116708220`に対し`8.071563865`: PASS。
- best standalone比`<=+0.02 ft`のfold数4/5: PASS。
- near / mid / 1000+ / hidden-like 2面の全固定scope非悪化: FAIL。
  `mid_250_1000`がexp335比`+0.048399545 ft`で、上限`+0.02 ft`を超えた。
- formation / signed familyはともにtotal gain正、positive gain fold 5/5: PASS。

### Tail promotion gate: FAIL

- exp264比by-well delta p95: `+2.198026177 ft`（上限`0.0`）。
- exp264比worst-well delta: `fb03ae90 +13.023263266 ft`（上限`+0.25`）。
- clean273比悪化well数:
  `+1 ft=157` / `+3 ft=53` / `+5 ft=23`
  （上限`135 / 39 / 14`）。

平均CVは改善したが、mid-rangeとtailの固定安全条件を満たさない。
promotion gateはFAILであり、LB候補へ昇格しない。

## Kaggle train履歴

### Version 1

- Kernel: `kentookumura/exp372-exp287-exp335-feature-union-train` version 1
- 終了状態: `ERROR`
- log最終時刻: 619.970949秒
- 失敗箇所: prefit parent compact fold load
- 例外: `KeyError: compact_features`
- LightGBM booster開始数: 0 / 15
- logs SHA256:
  `89d00097a675bb0373f3e1d4b464aeaa6a4d1d403f89043e369d3458607da8d8`

`verify_parent_compact_root`の`features`を、再利用したexp264 loaderの
`compact_features`契約へ変換するadapterを追加し、専用9 testsと関連44 testsを通した。

### Version 2

- Kernel: `kentookumura/exp372-exp287-exp335-feature-union-train` version 2
- Kaggle id_no: `128530478`
- Runtime: `NvidiaTeslaT4`、internet無効
- 終了状態: `COMPLETE`
- log最終時刻: 18,425.058990秒
- LightGBM booster完了数: 15 / 15
- logs SHA256:
  `0f8134297af145f7d4cb2da9bed6fef7c795bac9201e38adb151b16251a5704f`

## 成果物と再現性

実ファイル確認が必要なOOF、metrics、model manifest、SHAを監査するため、
version 2 output archiveを
`kaggle/output/train_v2/artifacts/`へ取得した。28 files、457,240,998 bytesで、
manifest記載の主要10成果物と15 model fileのSHAはすべて一致した。

- feature schema SHA256:
  `049800d626b04f16fbf08eb33e8a980ecbe62008402ff7b24f3e77e04e6ef4e9`
- model manifest SHA256:
  `e0d7f85c34d5c64410fe1b2e641669ee1887346a4cbd754579d0dd7e15875b5a`
- OOF SHA256:
  `635dea78b9bf7ad07a1bef267d37e4e2d1707f648799c1590715d4255c02e6f8`
- reproducibility manifest SHA256:
  `90eeede79b13d39ec3fcf6cb08268e6b396db42cb0e1f9e62fd9dca712ccdb5d`
- GPU LightGBMのbitwise reproducibility: 主張しない
- submission生成: false

## 結論

formation74とsigned23のunionはpooled CVを改善し、両familyも全foldで利用されたため、
平均的な相補性は確認できた。一方、fold 1、mid-range、exp264比tailが不安定であり、
安全に昇格できるunionではない。事前登録どおり同じOOFでのfeature/config/weight/gate救済を
行わずtrain branchを閉じる。

## 推論override

2026-07-25の明示指示により、保存済み40 parent selector、20 signed selector、
15 union TVT modelを使うCPU inferenceと提出形式検証用`submission.csv`生成を承認した。
raw testからcandidate/confidence、clean273、saved74、formation74、signed23を再生成し、
444列順序で予測する。model fitは0。competition submitは未承認である。

推論version 1はmodel manifestのartifact完了statusと、別の科学判定statusを誤比較して
25.549087秒でtechnical errorとなった。予測・fit前の失敗であり、保存modelや特徴契約の
不一致ではない。両statusを分離して検証する修正後、同じ0-fit契約でversion 2へ進む。

version 2はraw-test候補12本、14,151 rows / 3 wellsの生成を完了したが、exp335由来の
chunk設定をexp372に存在しない階層から参照して434.628235秒でtechnical errorとなった。
同系統の未到達参照も監査し、chunk sizeとsigned top1 toleranceをexp372 inference契約へ
明示した。同じ保存model・0-fit・submit無効契約でversion 3へ進む。

version 3はfixed 88-selectorのraw-context allowlistがexp372 configから欠落していたため、
expected dense列7本がNaNとなりmissingness guardで停止した。公開test 3 wellに
`MD/X/Y/Z/GR`が全件存在することと、成功済みexp264/exp335の同一契約を確認し、その5列
allowlistとtarget forbiddenを明示した。補完やguard緩和はせずversion 4へ進む。

## Kaggle CPU inference version 4

version 4は`KernelWorkerStatus.COMPLETE`。runtimeは`459.376 sec`、log最終eventは
`490.494217 sec`だった。raw test 14,151 rows / 3 wellsを処理し、候補12本、
selector 88列、`clean273 + saved74 + formation74 + signed23 = 444`を生成した。
保存済みmodelはparent selector 40、signed selector 20、union TVT 15を全slot使用し、
model fit / booster学習は0。formula parityとsigned top1 parityの最大絶対誤差は0だった。

提出形式検証はskill checkerとrepository checkerがともにPASSした。`submission.csv`は
14,151行、列`id,tvt`、sampleとheader・ID内容・順序が完全一致し、重複、NaN、Infは0。
予測はmin `11591.696289`、max `12239.309570`、mean `11905.273438`、
std `278.501831`。

- logs SHA:
  `5e405b2015a80a16c8262c54cfadd749d9bcba675e22fab21955359009fbf811`
- inference metrics / reproducibility manifest SHA:
  `f5436406513d306fef8e75c63f10b9446fb8cc5ca0204e975eb9f27daa72cd8d`
- prediction decompressed SHA:
  `5f18bcaf8cdd6952652155c6029c8045272b0b052a69ac8157bbf170aad4bc54`
- feature schema SHA:
  `bac4d7c539ea6b647c9393c75f00c99349c83887bf644a4a093c3fa89a0116f2`
- formation logical content SHA:
  `cc974f8cc4bd3976b42767fc690a8085389d39d249d73ff3f8e6bdf0c44c9d8c`
- submission SHA:
  `3688de824db2ae0ff1002fb9c2c9ed8543ed09d4e5bbfdd45d7bbf3c9c7eacdd`

trainのincremental utility / tail promotion / promotion FAILは維持する。推論成功は
科学gateの再分類ではない。推論NotebookとCodexはcompetition submitを実行していない。

## Kaggle scoring

ユーザーのscoring完了連絡後、Kaggle CLIとsubmission monitorで最新の完了提出を照合した。
推論version 4の完了時刻`2026-07-25 12:15:24 UTC`より後のCode submissionであり、
次の結果を確認した。

- submission ref: `54975325`
- submitted: `2026-07-25 12:28:12.460000 UTC`
- status: `COMPLETE`
- Public LB: `7.587`
- Private LB: 未確定
- monitor log: `logs/submission_exp372_exp287_exp335_feature_union_on_exp264.log`

Public LBはexp335 `7.517`より`+0.070`、exp287 `7.530`より`+0.057`、
exp264 `7.562`より`+0.025`悪い。別routeのexp082 `7.601`は`0.014`上回るが、
ML routeのanchorはexp335のまま更新しない。unionはCVでは単独親を上回った一方、
Public LBへ改善が転移しなかった。train科学gate FAIL、same-OOF rescue禁止、
非昇格判断をすべて維持する。外部提出はユーザー確認後に観測したもので、Codexは実行していない。
