# exp264_exp263_candidate_confidence_dual_selector 結果

> **旧結果無効:** 旧Stage A/B/CとStage D `selector_compact_addonly`は、hidden testに存在しないtraining-only
> formation 6列のraw値・差分12特徴をouter-validで使用した。feature availability leakageのため、以下の
> 旧score guard、RMSE、fold/bucket/hidden-like/by-well差、feature importanceは比較・採用判断に使用できない。
> `7.805644`も有効なOOF CVではない。修正版Stage B version 5の88列selector score OOFだけは有効で、
> 旧結果とは分離して記録する。

## 仮説

exp263の12 deployable candidate surfaceと候補別confidenceを整理済みcandidate-long featureへ入れ、候補別dual scoreをfold-safe compact meta-featureにすれば、固定blendより候補品質を詳しく表現できる。

## 設定

- 候補親: exp263
- 方法親: exp251 dual-objective candidate-long
- adapter参照: exp238 compact add-only
- 候補: 12 score surface、2 legal domain
- 検証: outer well 5 folds。downstreamは条件付きouter 5 × inner 4。
- selector objective: `pred_abs_error` / `p_within10`
- シード: 42
- 旧実行済み（無効）: Stage B 10 CPU selector boosters、Stage C 40 CPU nested selector boosters、
  Stage D matched control 15 + compact add-only 15 = 30 GPU TVT boosters
- 修正版実行済み: Stage A v4は0 booster、Stage B v5は10 CPU selector boosters、
  Stage C v6は40 CPU nested selector boosters、Stage D v3は30 GPU TVT boosters

## 修正版availability監査

2026-07-18にfeature-level lineageを再監査した。

| 対象 | 旧列数 | hidden-safe | 無効 | 判断 |
| --- | ---: | ---: | ---: | --- |
| exp264 selector | 100 | 88 | 12 | formation raw/deltaを削除してStage Aから再構築 |
| exp218 downstream | 380 | 273 | 107 | 既存control/OOFを再利用しない |

exp218の107列は、full-train formation reference依存74列、exp111 fold0 target-trained scoreを全trainへ
適用した非nested stacking依存27列、その推移依存GRWR 6列である。380列すべてがcurrent-testでfiniteに
なった事実はschema coverageでしかなく、fold-safeの証拠ではなかった。

selector側はraw allowlistを`MD/X/Y/Z/GR`へ縮小し、actual train/current-test全ファイルのheaderをfit前に
照合するgateを実装した。修正版Stage A version 4はKaggle CPUで完了し、train 773/773 file、current-test
3/3 fileで5列すべてのavailabilityがPASSした。88列は暫定値ではなく、logical schema SHA
`aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`で凍結した。

### 修正版Stage A version 4

| 項目 | 実測値 |
| --- | ---: |
| Kaggle status | COMPLETE |
| 生成物表示まで | 152.891秒 |
| 学習booster | 0 |
| 監査candidate-long rows | 600,000 |
| 変換後監査前特徴 | 150 |
| 採用特徴 | 88 |
| 全欠損除外 | 41 |
| 定数除外 | 5 |
| 完全重複除外 | 16 |
| 採用側の完全重複 | 0 |
| 高相関report-only | 14組 |
| 学習時に構造的NaNを持つ採用特徴 | 25 |

高相関14組は、候補TVT・last-known・候補bank中央値の近似関係、bank spreadの近似関係、candidate one-hotと
confidence/formula validityの決定的関係である。差分値や欠損patternに意味があるため、事前契約どおり
自動削除せずStage Bで重要度を確認する。完全一致16列はすでに除外され、採用88列内には残っていない。

## 修正版Stage B version 5

Kaggle CPU version 5は1 variant × 2 objectives × 5 folds = 10 boostersを完走した。
expected-error MAEは5.788783から3.795801、within10 logloss/Brierは0.510131/0.165095から
0.359972/0.112451へ改善し、3指標すべてpooledかつ5/5 folds改善でscore guardをPASSした。

一方、hard top1 RMSEは8.587004でfixed 8.238332より+0.348673悪化し、改善foldは0/5。
near +0.079326、1000+ +0.389208、worst-well +14.684481、独立hidden-likeはspatial +0.768585 /
typewell-purged +0.721137であり、hard selectorは明確に不採用。scoreを74列compactの内部表現としてのみ残す。

candidate-long 45,407,868行とcompact 3,783,989行×74列を全行監査し、欠損・nonfinite・確率範囲外・
fold/model対応違反は0。10/10 model SHAとOOF/manifest SHAも一致した。confidence groupは
pred-abs-error gain 4.267%、`sigma_tvt`は4位・2.958%。全88特徴の説明・重要度・重複・相関は
`selector_feature_readout_corrected_stage_b_v5.md`を正とする。

## 修正版Stage C version 6

Kaggle CPU version 6は1 variant × 2 objectives × outer 5 × inner 4 = 40 boostersを完走した。
expected-error MAEは5.788783から3.798819、within10 logloss/Brierは0.510131/0.165095から
0.359412/0.111830へ改善し、3指標すべてpooledかつ5/5 folds改善でscore guardをPASSした。
Stage B比はMAE +0.003018、logloss -0.000560、Brier -0.000621で、nested化しても校正品質を維持した。

outer-valid wellをinner assignment、fit、early stoppingから除外し、inner train/validはwell-disjoint。
outer-train compactはinner OOF、outer-valid compactは4-inner-model ensembleから生成され、nested leakage
auditはPASSした。40/40 model byte SHA、40期待組合せ、25 partition manifest、18,919,945 compact rows、
45,407,868 outer-valid candidate-long rowsを監査した。

一方、hard top1 RMSEは8.652532でfixed 8.238332より+0.414200悪化し、改善foldは1/5。
fold差は+0.800053 / -0.059620 / +1.041497 / +0.068543 / +0.216519でhard guardはFAILした。
したがってStage Cの合格範囲は74列compactの後段add-only入力までであり、hard selector、Viterbi、
softmax TVT平均、submissionは不採用を維持する。

model manifest SHAは`3f28b04a...e2d2`、compact manifest SHAは`f4855726...f1c1`、
nested metrics SHAは`421376ab...478b`。巨大compact Parquetはローカル取得せず、Stage D fit前に
Kaggle入力上の25/25 partitionをmanifest byte SHAで再検証する。

## 修正版Stage D version 3

canonical Kaggle T4 version 3はclean 273 matched control 15本と、273 + nested compact 74 = 347列の
add-only 15本、合計30/30 GPU boostersを完走した。Stage C v6の25 compact partitionは1本目のfit前に
byte SHAを全件検証し、2 variantで行、fold、3 LightGBM config、runtimeを一致させた。

| 指標 | clean 273 control | 347 add-only | delta | 判定 |
| --- | ---: | ---: | ---: | --- |
| pooled `lgb_mean` RMSE | 10.476169 | 8.460811 | -2.015358 | PASS |
| near 0-250 RMSE | 2.029054 | 1.583151 | -0.445903 | PASS |
| 250-1000 RMSE | 4.856472 | 4.099686 | -0.756786 | PASS |
| 1000+ RMSE | 11.535491 | 9.302283 | -2.233208 | PASS |
| hidden-like spatial | 12.493329 | 9.420315 | -3.073014 | PASS |
| hidden-like typewell-purged | 12.433031 | 9.341391 | -3.091639 | PASS |

fold deltaは-2.103882 / -1.322387 / -1.432684 / -2.090369 / -2.989973で、5/5 folds改善した。
一方、773 well中518改善・255悪化、+1 ft超135、+3 ft超39、+5 ft超14だった。worst `70925e23`は
11.825487から26.308360へ+14.482873悪化し、事前上限+0.25を超えた。したがってoverall、fold、near、
1000+、hidden-likeのcheckはPASSしたが、総合guardは**FAIL**である。

compact 74列はadd-only全体の15-model平均正規化gain 76.9258%、split 25.2013%を占めた。上位4列は
2 legal domainの`p_within10` / `pred_abs_error` top1候補値-minus-anchorで、合計gain 61.0343%。5位は
`selector__pred_abs_error__beam_mean`の5.8196%だった。これはhard `beam_mean`採用ではなく、候補値と
selector scoreを連続meta-featureとして後段が強く利用した結果である。全74列の説明と重要度は
`stage_d_feature_importance_readout_corrected_stage_d_v3.md`を正とする。

結果は有効なmatched ablationとして保持し、worst-well guardを緩めない。2026-07-19のユーザー明示overrideにより、
corrected Stage C v6 / Stage D v3を使うhidden-safe inferenceとsubmit-check後の参考提出だけを例外実行する。
hard selector、Viterbi、softmax TVT平均は引き続き実行しない。
小さい根拠生成物は`kaggle/output/stage_d_v3_corrected/artifacts/`へ保存し、metrics / OOF / model manifest SHAは
`29cecbf7...0acb` / `b11c5005...9ae2` / `c3b22481...5fcc`。

## 修正版推論・提出（完了）

- selector: corrected Stage C v6、88特徴、40 saved models。
- TVT: corrected Stage D v3、clean 273 + compact 74 = 347特徴、add-only 15 saved models。
- 学習: 0 booster。matched control推論、hard selector、Viterbi、softmax TVT平均は0。
- Stage C private Datasetをv6 bundle SHA `0e1ae1a5...b5adf`へ更新済み。
- inference v4は424.511秒で完了し、formula parity 0、submit-check PASS。ref `54818932`は
  Public LB 7.562でCOMPLETEし、直前ML anchor exp274の7.715を-0.153更新した。
  別routeのexp082 ensemble 7.601も-0.039で上回る。Stage D guard FAILは保持する。

## 旧結果（feature availability leakageにより無効）

| メトリック | 値 |
| --- | --- |
| Stage A audit | **INVALID**、raw-test availability監査漏れ |
| 特徴量 | 162候補 → 100採用 |
| 除外 | 全欠損41、定数5、完全重複16 |
| 高相関 | 35組、report-only |
| feature schema SHA | `766cfcf10a14fdcd0aa6f6ff78f347c1fb4f3eb86f95138b8676d11da96d4deb` |
| compact schema | 74列、`23614916...725` |
| native confidence依存 | 採用21列、exp263 v3実値parity完了 |
| expected-error MAE | **3.742231**、prior 5.788783から改善、5/5 folds改善 |
| within10 logloss / Brier | **0.355298 / 0.110596**、prior 0.510131 / 0.165095から改善、各5/5 folds改善 |
| score guard | **INVALID**、feature availability leakage |
| diagnostic hard top1 RMSE | **8.362844**、fixed `exp226_w500_50_50` 8.238332より+0.124512 |
| hard readout guard | **FAIL**、near +0.088746、1000+ +0.135728、worst-well +18.258274 |
| hidden-like spatial / typewell-purged | hard-fixed **+0.438111 / +0.407604** |
| model | 10本、全model SHA一致 |
| Stage C expected-error MAE | **3.762776**、prior 5.788783から改善、5/5 folds改善 |
| Stage C within10 logloss / Brier | **0.354702 / 0.110137**、prior 0.510131 / 0.165095から改善、各5/5 folds改善 |
| Stage C score / leakage guard | **PASS / PASS** |
| Stage C hard top1 RMSE | **8.420613**、fixed 8.238332より**+0.182281**、FAIL |
| Stage C出力 | 40 models、25 compact partitions、18,919,945 rows、outer-valid score 45,407,868 long rows |
| Stage D matched control / add-only | **INVALID**、旧計算 8.545568 / 7.805644 |
| Stage D fold | **5/5改善**、最小改善fold 3も-0.083249 |
| Stage D near / mid / 1000+ | **-0.222916 / -0.419414 / -0.807155** |
| Stage D hidden-like spatial / typewell-purged | **-1.174830 / -1.193025** |
| Stage D worst-well | `70925e23` **+17.446742**、guard上限+0.25 |
| Stage D guard | **FAIL**、worst-well以外の全checkはPASS |
| Stage D成果物 | 30 models、3,783,989 OOF rows、30/30 model SHA・8/8 output SHA一致 |
| Public LB | - |
| Private LB | - |

全100特徴の説明、objective別5-fold mean gain重要度、candidate選択率、16完全重複、41全欠損、
5定数、35高相関組は`selector_feature_readout.md`へ記録した。
Stage Dの全74 compact特徴の説明と15-model正規化gain/split重要度は
`stage_d_feature_importance_readout.md`へ記録した。

## 再現性

- deterministic anchor: 単発Kaggle runとしてSHA固定済み。独立rerunは未実行。
- seed policy: fixed 42 + stable SHA256 sampling。
- kernel: `kentookumura/exp264-exp263-confidence-dual-selector-train` version 2、id_no `127485868`、CPU、internet off
- runtime: 生成物一覧表示まで1,633.209秒、notebook変換完了まで1,644.601秒
- exp263 manifest SHA: `85e60ac1...a26bb9e`、catalog SHA: `7cd74866...e9e6e0`
- feature content SHA: audit-long `62d84bcf...a4896d`
- model manifest SHA: `12375038...4c9a`。10 modelを個別取得し、全SHA一致。
- candidate score OOF SHA: `e51bb674...45a5a`
- compact meta OOF SHA: `1ab4cff4...45ba`
- Stage C kernel: 同一kernel version 3、CPU、internet off。生成物表示まで4,329.795秒、
  notebook変換完了まで4,338.238秒。
- Stage C model manifest SHA: `b2d8def7...aab1`。outer 5 × inner 4 × 2 objectivesの40組と
  40個の一意SHAをmanifestで確認し、ログ上の40実ファイルを確認した。選択取得した3 modelはbyte-level SHA一致。
- Stage C compact manifest SHA: `c95d9ea4...c06e`。25 partition、18,919,945 rows、
  5 valid partitionは4-model ensemble、20 train partitionはinner OOFの1 modelであることを確認した。
- Stage C outer-valid candidate score SHA: `4a77ceb7...2777`。巨大Parquet本体は取得せず、
  Kaggle生成ログ、metrics、manifestのSHA連鎖を監査根拠とした。
- Stage D kernel: `kentookumura/exp264-exp263-confidence-dual-selector-tvt-train` version 2、
  id_no `127577193`、Nvidia Tesla T4、internet off。30本目完了まで32,344.668秒、
  生成物一覧まで32,352.136秒、notebook変換完了まで32,361.299秒。
- Stage D OOF SHA: `7367983f...dafee`。3,783,989 rows、fold coverage
  `757738/756650/756255/757101/756245`、予測・targetの欠損/nonfinite 0。8 pooled RMSEを再計算して一致した。
- Stage D model manifest SHA: `064684b6...6159`。2 variants × 3 configs × 5 foldsの30組は一意で、
  取得した30 model全件のbyte SHAがmanifestと一致した。reproducibility manifestの8 output SHAも全件一致した。
- submission SHA: scope外
- rerun result: -

## 解釈

修正版Stage Bでは候補別scoreの校正を学習できた。expected-error MAE、within10 logloss/Brierは全foldで
candidate別outer-train priorを上回ったため、`pred_abs_error[12]`と`p_within10[12]`はcompact metaの
内部表現として残す価値がある。重要度はbank disagreementが両objectiveで約54%を占め、confidence
groupはpred-abs-errorで4.267%、within10で1.461%だった。`conf__native__sigma_tvt`は予測誤差重要度4位で、
候補固有confidenceを追加した狙いは支持された。candidate ID one-hotも小さいが非ゼロの寄与を持つ。

一方、予測誤差最小候補をそのまま選ぶhard top1は固定blendより悪い。全体+0.348673、0/5 folds改善で、
near、1000+、hidden-like、worst-wellの全guardも外した。したがってhard selector、Viterbi、softmax平均、submissionへは
進めない。これはselector scoreが無意味という意味ではなく、「top1へ離散化するとtail riskが大きい」
という結果である。

修正版Stage B OOF compactをそのままdownstream outer-trainへ使うとnested leakageになる。したがって、
compact metaの価値を後段で検証するには、修正版Stage Cでouter 5 × inner 4 × 2 objectivesのscoreを
作り直す必要がある。outer-trainはinner OOF、outer-validは4 inner model ensembleから生成する。

旧Stage Cでは同じ構造で18,919,945 rowsの74列compactを25 partitionへ保存したが、旧100列schemaの
feature availability leakageによりmodel・compact・score・承認をすべて再利用しない。

旧Stage C hard top1、連続score、compactの全判定はfeature availability leakageにより無効である。
hard selector、Viterbi、softmax TVT平均、submissionの禁止は維持する。COPCFは同一生成parityを
満たす入力がないため引き続きdeferする。

旧Stage D version 2では74列のadd-onlyが改善して見えたが、feature availability leakage判明後は後段へscoreを
渡す仮説も支持されたとは扱わない。旧compact重要度70.96%や上位4列59.96%も無効な入力上の挙動であり、
特徴価値の根拠にしない。一方、88列hidden-safe selectorとclean 273 downstreamで再構築した修正版version 3は
同じ仮説を有効なmatched ablationとして支持し、compact gain share 76.93%と5/5 folds改善を確認した。

旧version 2のwell単位改善・悪化数とworst-well値も同じ理由で比較根拠にしない。旧version 2の総合判定は
guard FAILではなく、その前段のfeature availability contract不成立による`INVALID`とする。修正版version 3は
有効だが、worst `+14.482873`により事前guardがFAILしたため、推論採用には至らない。

## 次

修正版Stage D version 3は有効なOOF比較としてglobal/fold/bucket/hidden-likeを大幅改善したが、
worst-well guardに失敗した。現時点では推論へ進めず、local run gateを閉じたままにする。

次の妥当な一手は、旧親のleakageで無効化された`exp276`の固定target-free tail-risk readoutを、修正版
Stage C v6 / Stage D v3へ入力だけ差し替えて再監査すること。これは0 boosterのCPU監査であり、feature、
family weight、q70/q80/q90、guardを今回の結果に合わせて変更しない。再開する場合も別途ユーザー確認後に
同じexp276で行い、guard通過前のcorrected inference / submissionは禁止を維持する。

## 2026-07-18 hidden-safe inference（version 2/3失敗・deployment契約無効）

上記のStage D guard FAILは変更しない。その後のユーザー明示指示により、例外scopeを未提出の推論成果物
生成だけに限定して`kentookumura/exp264-stage-d-hidden-safe-inference`を開始した。version 1はStage D
manifest期待SHAの転記誤りをfail-closed guardが27秒で検出し、候補/model推論前にERRORとなった。実manifestと
reproducibility manifestで正しいSHAを二重確認し、誤記だけを直したversion 2をpushした。version 2は
395.586秒、候補生成後・selector predict前に`current-test selector matrix contains non-finite values`でERRORとなった。

原因は入力parity違反ではなく、推論側の一律finite guardである。Stage A採用100列中29列は学習時からNaNを
持ち、特に候補固有`conf__`とformula非該当slotはNaNをLightGBM missing semanticsとして使っていた。
修正版ではStage A feature catalogをSHA固定で同梱し、期待NaNを0補完せず保持する。`±inf`、training-dense列の
新規NaN、`conf__`/`formula__`構造欠損率ずれ、current-test全欠損化だけをfail-closedにする。exp218 base 380列、
compact 74列、後段454列は学習時どおりfinite必須を維持する。

- 新規学習: 0 variant / 0 config / 0 fold / 0 booster。
- selector: Stage C保存済み40本。各outer foldでinner 4本 × 2 objectivesを平均し、74 compact列へ即変換する。
- TVT: Stage D保存済み`selector_compact_addonly` 15本だけを使用する。matched control 15本は使わない。
- current test: exp263の6 primitive、5 pair、fixed 1、21 confidence列をraw competition testから同じrunで再生成する。
- base feature: exp263 replayを基にexp218の380列を再生成し、outer別compactと連結した454列をmodel順で検証する。
- output: prediction、outer別compact、監査sample、`submission.csv`、metrics/reproducibility SHA。competition submitは行わない。
- missingness audit: feature別・candidate別CSVを保存し、Stage A catalog SHA `83c8b953...639d`へ連鎖する。
- Stage C model bundle: private Dataset `kentookumura/exp264-stage-c-selector-models`、13,029,920 bytes、SHA `1697e1f7...6b21`。

修正版version 3完了後にrow/sample order、candidate/confidence coverage、selector欠損契約、40/40 selector SHA、
15/15 TVT SHA、454列と予測のfinite、prediction/submission SHAを監査する。推論完了はStage D guard PASSや
提出採用を意味しない。

同じkernel IDへversion 3をpushし、直後のKaggle statusは`KernelWorkerStatus.RUNNING`だった。pushしたpackageの
notebook/config/metadata SHAは`9a01bea9...6d63` / `b4495b05...7321` / `27012472...906`。
実行scopeは保存済み40 selector + 15 add-only TVT modelのpredictionだけで、新規variant/config/fold/学習boosterは
すべて0、competition submitも0である。ユーザー方針どおり継続監視は行わず、完了連絡後に成果物を監査する。

完了連絡後のログ確認でversion 3は378.938秒に`ERROR`と確定した。公開test horizontalは
`MD/X/Y/Z/GR/TVT_input`だけで、公式にもtraining onlyとされる`ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA`を持たない。
しかしStage A/Cはこの6列のraw値とlast-known差の12特徴を選択していた。うちtraining missing率0の8特徴を
新guardが検出して停止し、残る4特徴もcurrent-testでは全欠損になるため既存schemaはhidden-safeではない。
12特徴のselector gain share合計はpred-abs-error 5.657%、within10 7.622%で、単純なNaN許可・0/median補完・
testだけのKNN補完では学習/推論parityを回復できない。Stage C/Dの数値は診断・比較にも使わず、失敗再現値
としてだけ残す。正規修正はdirect formation特徴を除去するかfold-safe imputed formationへ置換し、
selectorとdownstream add-onlyを再学習することである。

## 2026-07-19 corrected inference version 4・参考提出

上記version 3失敗後、direct formation 12特徴を除いたStage C v6（88列）と、非fold-safe downstream
107特徴を除いたStage D v3（clean 273 + compact 74 = 347列）を再学習済みartifactとして固定した。
worst-well guard FAILを保持する参考提出であることを条件に、ユーザー明示承認でinference version 4を実行した。

version 4は424.511秒で`COMPLETE`。14,151 rows / 3 wells / 12 candidates、21 confidence列、88 selector
features、40 selector models、74 compact features、380 source baseから273 clean base、347 final features、
15 TVT models、0 training boosterを実測確認した。formula parity最大絶対誤差は0である。hard selector、
Viterbi、候補softmax平均は使わず、Stage D add-only 15 model predictionの等重み平均だけを最終値とした。

`submission.csv`はsampleとheader・行数・ID順が完全一致し、重複ID、empty、NaN、Infは0。
submit-checkはPASS、SHAは`cbaad9a3603008f4adaaf0c53a3369aa47f0fd95db8711ad0d005116663297b7`。
competition submission ref `54818932`はPublic LB **7.562**で`COMPLETE`。直前ML submitted anchor
exp274 / 7.715を-0.153改善したため、Public-LB上のML anchorをexp264へ更新する。
別routeのensemble anchor exp082 ref `53885305` / 7.601も-0.039で上回るが、ensemble anchorはexp082に維持する。

kernel version 4完了時にdescriptionなしの自動run record ref `54818883`も作成され、同じ7.562だった。
明示submit ref `54818932`と同scoreであり、推論出力の再現結果として扱う。worst-well `+14.482873`
guard FAILは維持し、これは「LB anchor更新」であって「train-side guard PASS・モデル採用」ではない。

routeは`ml_model`へ修正した。PF/HMM/Beam候補はselectorの補助meta featureであり、direct blend、
hard-path、Viterbi、softmax TVT平均は使わず、最終予測はdownstream LightGBMが生成するためである。

## 2026-07-19 OOF診断成果物

exp238のselector-confidence / LikPF 128-path probeをexp264のcorrected artifactへ移植した。
selector表示はStage C v6 strict nested outer-valid score、最終ML表示とviewer CSVはStage D v3 add-only OOFを使う。
12候補はprimary 11候補とfixed比較7候補の2 legal domainを分離し、hard top1 guard FAILとStage D worst-well
guard FAILを注記した。LikPFは500 particles × 128 seedsのexact replay契約を維持する。

viewer CSVは`id,tvt`の3,783,989行 / 773 wells、unique ID、NaN/Inf 0、RMSE 8.460811で、repository viewerへ
読込可能。出力SHAは`9fe0cfceda8b8e3d852c74352e0e4d7d6748f057b79354133b110e77173ce04b`。

selector probe version 3とLikPF probe version 1はともにKaggleで`COMPLETE`。selectorは3,783,989行 / 773 wells、
773 plotsを生成し、final OOF RMSE 8.460811、primary hard top1 RMSE 8.652532だった。version 2のplotは無効のままとし、
version 3でexp238と同じ3段構成・比較パス・色・線種・exact HMM ±2sigma帯を復元した。summary SHAは
`70eab1a000caf89fed7f6bc9f2138806f5180b27f75d95c993c1b304e0f3f869`。

LikPFは500 particles × 128 seeds × 773 wellsを完走し、773 plotsを生成した。保存済みexp072 meanとのparityは
全wellでexact、最大絶対差0。PF seed-mean RMSEは11.594898、summary SHAは
`ffc85e804d564dc0c1ade8245dceafb87b7e78b446e05495d5c36ceb6bec94d0`。両notebookとも学習・blend・submitは0である。
