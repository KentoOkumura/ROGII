# exp264_exp263_candidate_confidence_dual_selector

## 状態

- ルート: `ml_model`
- 状態: 修正版88列Stage Bとnested Stage C v6を完了。Stage Cはscore/leakage PASS、hard top1 FAIL。
  修正版Stage Dのclean 273 control / 347 add-only、合計30 GPU boostersをKaggle T4 version 3で完走。
  pooledは大幅改善したがworst-well guard FAIL。ユーザー明示overrideによるcorrected inference v4は
  COMPLETE、submit-check PASS、参考提出ref 54818932はPublic LB 7.562でCOMPLETE
- CV: 修正版selector OOF / nested compact / downstream TVT OOFは有効。Stage Dは10.476169 → 8.460811。
  旧7.805644は無効
- Public LB: 7.562（直前ML anchor exp274 7.715から-0.153、新ML LB anchor。別routeのexp082 ensemble 7.601も-0.039で上回る）
- Private LB: -
- Submit ID: 54818932（同一runの自動record 54818883も7.562）
- 作成日: 2026-07-16
- 候補親: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 方法親: `exp251_raw_test_safe_dual_objective_candidate_ranker`
- downstream参照: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`

## 仮説

exp263が生成するraw-test対応12候補を、候補固有confidence、全候補共通のtarget-free proxy、整理済みraw-test-safe contextと一緒にcandidate-long selectorへ入れれば、旧候補集合を使うより候補誤差とwithin10確率を安定して推定できる。候補別dual scoreをhard pathへせずcompact meta-featureへ変換すれば、downstream TVT LightGBMが候補情報を安全に使える。

> **旧実装の無効化:** 旧100列実装はtraining-only formation 6列のraw値と差分12特徴をouter-validへ直接与えていた。
> これはfeature availability leakageであり、Stage A/B/Cのguard、Stage C compact、Stage D add-only OOF、
> feature importance、bucket/by-well比較をすべて意思決定に使用しない。数値は失敗履歴としてのみ残す。

## 変更点

- 候補source of truthをexp263に固定した。
- score対象を6 primitive + 5 pair + fixed 1 = 12本にした。
- score対象12本はすべてexp263 Stage 1でcurrent-test生成済みである。Stage 0 OOFにのみ収録された追加surfaceは、現行Stage 1出力へ追加してparity確認するまで別inventoryとして扱う。
- `blend_likpf_hmm_w500`はpair aliasとして重複登録しない。
- 12 candidate IDをone-hotにし、ordinal `candidate_index`を禁止した。
- native confidenceがない候補にもshape、anchor距離、bank/formula disagreement等のuniversal proxyを付ける。
- exp263 Stage 1に21 namespaced confidence列を固定し、推論adapterで列欠損・非finite・invalidをfail-closedにした。
- exp251 v4 295列をseedに、`ctx/cand/conf/bank/formula/id`へ再編する。
- exp251 v4 295列は分類表へ展開し、COPCFは同等のtrain cross-fit/current-test generatorを接続するまでdeferする。初回はraw horizontal/typewellから共通contextを生成する。
- selector出力を`pred_abs_error[12]`と`p_within10[12]`に固定し、同じfold/process内でcompact化する。
- pairを含む11本domainとfixedを含む7本domainを分け、exp263 lineage guardを維持する。
- HMM+LGB、Viterbi、hard-path提出、softmax TVT平均をscope外にした。

物理・幾何候補の状態方程式、観測モデル、特徴量、候補別/最終精度は
[`physical_model_summary.md`](physical_model_summary.md)にまとめた。契約の正は`candidate_contract.yaml`、
`feature_contract.yaml`、`output_contract.md`とする。

## 2026-07-18 修正版availability監査

- 旧selector 100列は、formation raw/delta 12列を削除して88列をallowlistとした。
- configのraw allowlistを`MD/X/Y/Z/GR`だけへ変更し、actual train/current-test全horizontal fileの
  headerをfit前に照合するfail-closed gateを追加した。
- exp218 380列も独立監査し、current-testでfinite生成できることとfold-safeであることを分離した。
- exp218はbase formation依存74列、非nested exp111 score依存27列、推移依存GRWR 6列の計107列が
  fold-safeでない。既存380列matched controlとOOFは再利用しない。
- 修正版Stage A version 4はKaggle CPUで152.891秒、0 boosterで完了した。600,000 candidate-long rowsを
  監査し、150列から全欠損41、定数5、完全重複16を除いた88列をlogical schema SHA
  `aaef4ffd...ddd3a4`で凍結した。train 773/773・current-test 3/3 fileで`MD/X/Y/Z/GR`が全件存在した。
- 高相関14組はreport-only。採用88列内の完全重複は0で、構造的NaNを持つ採用特徴は25列。
- 詳細は`artifacts/feature_availability_audit/README.md`と2つのfeature-level CSVを正とする。

## 検証方針

- Fold: outer GroupKFold 5。downstream時だけouter 5 × inner 4。
- Group: exp263 cacheのcanonical key `well`（raw dataの`well_id`に対応）。
- Score rows: `TVT_input`欠損相当のevaluation rows。
- Stage A: 0 boosterでmanifest、候補identity、coverage、formula、feature provenance、欠損、重複、相関を監査する。
- Stage B: 1 variant × 2 objectives × 5 folds = 10 CPU boosters。
- Stage C: outer 5 × inner 4 × 2 objectives = 40 CPU selector boosters。修正版version 6完了。
- Stage D: downstream baseは107列を落としたclean 273列、compact add-onlyは347列へ固定。修正版version 3の30 GPU boosters完了。
- Leakage Check: labelはfeature固定後に別join。outer-valid labelをfit/calibration/selectionへ使わない。downstream outer-trainはinner OOF scoreだけを使う。
- Inference Check: exp263 Stage 1の12 surface parity成立が前提。保存済みmodelを使い、公開test row artifactや再学習に依存しない。
- Missingness Check: 修正版Stage A catalogの学習時欠損率を正とし、25 sparse selector特徴のNaNを0補完せず保持する。`±inf`、dense列の新規NaN、構造欠損率ずれ、全欠損化は停止する。

## 成功条件

- expected-error MAE、within10 logloss/Brierがcandidate別outer-train priorよりpooledかつ4/5 folds以上で改善する。
- confidence coverageとcalibrationをcandidate/fold/distance/validity別に説明できる。
- diagnostic hard readoutはfixed `exp226_w500_50_50` 8.238331からoverall -0.02以上、3/5 folds改善、near/1000+/hidden-like各+0.02以内、worst-well +0.25以内を目安とする。
- hard readoutが不通過でもscore品質が通れば、Stage Cのcompact add-only候補には残せる。

## 実行入口

- 学習notebook: `exp264_exp263_candidate_confidence_dual_selector_train.ipynb`
- 推論notebook: `exp264_exp263_candidate_confidence_dual_selector_inference.ipynb`
- Kaggle準備: `task prepare-kaggle-notebooks EXP=exp264_exp263_candidate_confidence_dual_selector`
- notebook実行: Kaggle kernel runを正とする。旧Stage B version 2、Stage C version 3、Stage D version 2は履歴として保持するが再利用しない。
  Stage D version 1はschema SHA契約ミスで22.5秒時点に学習前停止し、version 2で30/30 boostersを完走した。
  修正版Stage Bは1 variant × 2 objectives × 5 folds = 10 CPU boostersでKaggle version 5完了。
  重複実行防止のため、push後のlocal `run_approved=false`に戻した。
  hidden-safe inferenceは未提出artifact生成だけ例外承認済み。v2は想定NaNを一律finite guardが拒否した。
  v3は378.938秒、公開/hidden testに存在しないtrain-only formation 6列から作ったraw/delta 12特徴を
  Stage C schemaが要求していることをfail-closed検出し、selector predict前に停止した。欠損許可や0補完では
  修正にならない。新規学習は0 booster、Viterbiとcompetition submitも0。修正版Stage A version 4は完了し、
  88列Stage B 10 CPU boostersは完走し、clean 273列downstream方針を固定済み。
  clean 273 allowlist SHAは`d01a73cc...77bf`。修正版Stage C version 6は40 CPU boostersで完了。
  Stage Dはcontrol 273 / add-only 347列の30 GPU boostersを2026-07-18に承認し、version 3で完走した。

## 修正版Stage B version 5結果

| 指標 | selector | prior / fixed | 判定 |
| --- | ---: | ---: | --- |
| expected-error MAE | 3.795801 | 5.788783 | PASS、5/5 folds改善 |
| within10 logloss | 0.359972 | 0.510131 | PASS、5/5 folds改善 |
| within10 Brier | 0.112451 | 0.165095 | PASS、5/5 folds改善 |
| hard top1 RMSE | 8.587004 | 8.238332 | FAIL、+0.348673、0/5 folds改善 |

- 10/10 model SHA、candidate-long 45,407,868行、compact 3,783,989行×74列を実ファイルで監査し、
  行数・候補数・fold対応・欠損・nonfinite・SHAはすべてPASS。
- hardはnear +0.079326、1000+ +0.389208、worst-well +14.684481、hidden-likeも
  spatial +0.768585 / typewell-purged +0.721137で不採用。
- confidence groupは予測誤差gain 4.267%、`sigma_tvt`は4位・2.958%で、候補信頼度を入れる価値は確認できた。
- 修正版88特徴すべての説明・重要度・重複・相関は
  `selector_feature_readout_corrected_stage_b_v5.md`を正とする。

## 修正版Stage C version 6結果

| 指標 | nested selector | prior / fixed | 判定 |
| --- | ---: | ---: | --- |
| expected-error MAE | 3.798819 | 5.788783 | PASS、5/5 folds改善 |
| within10 logloss | 0.359412 | 0.510131 | PASS、5/5 folds改善 |
| within10 Brier | 0.111830 | 0.165095 | PASS、5/5 folds改善 |
| hard top1 RMSE | 8.652532 | 8.238332 | FAIL、+0.414200、1/5 folds改善 |

- outer-validをinner fitから除外し、outer-trainはinner OOF、outer-validは4-inner-model ensembleから
  compact化した。well disjoint、model/partition/row coverageを含むnested leakage auditはPASS。
- 40/40 model byte SHA、40期待組合せ、25 partition manifest、18,919,945 compact rows、
  45,407,868 outer-valid candidate-long rowsを監査した。巨大Parquet本体はStage D入力時に25/25 byte SHAを再検証する。
- 重要度group shareはpred-abs-errorでbank 54.909%、ctx 26.848%、formula 8.756%、cand 4.902%、
  conf 4.247%、id 0.337%。`sigma_tvt`は誤差objectiveの5位・2.841%。
- score/leakageは合格したため74列compactはStage Dの入力候補になるが、hard selector、Viterbi、
  softmax TVT平均、submissionには使わない。

## 修正版Stage D version 3結果

| 指標 | clean 273 control | 347 add-only | delta | 判定 |
| --- | ---: | ---: | ---: | --- |
| pooled `lgb_mean` RMSE | 10.476169 | 8.460811 | -2.015358 | PASS |
| near 0-250 | 2.029054 | 1.583151 | -0.445903 | PASS |
| 250-1000 | 4.856472 | 4.099686 | -0.756786 | PASS |
| 1000+ | 11.535491 | 9.302283 | -2.233208 | PASS |
| hidden-like spatial | 12.493329 | 9.420315 | -3.073014 | PASS |
| hidden-like typewell-purged | 12.433031 | 9.341391 | -3.091639 | PASS |

- 30/30 GPU boostersを完走し、fold deltaは全5 foldで負。Stage C v6の25 compact partition byte SHAもfit前に通過した。
- 773 well中518改善・255悪化。+1 ft超135、+3 ft超39、+5 ft超14。
- worst `70925e23`は11.825487 → 26.308360、+14.482873で事前上限+0.25を超えた。
- compact 74列の15-model平均正規化gain shareは76.9258%、split shareは25.2013%。上位4つの
  top1-minus-anchorがgain 61.0343%、5位は`beam_mean`予測誤差score 5.8196%。全74列は
  `stage_d_feature_importance_readout_corrected_stage_d_v3.md`に記録した。
- overall/fold/bucket/hidden-likeは改善したが、worst-well guardにより総合FAIL。後続のユーザー明示overrideで
  corrected inferenceとreference submissionだけを実行した。hard selector、Viterbi、softmax TVT平均は使わない。

## 旧結果（feature availability leakageにより全性能値無効）

| メトリック | 値 |
| --- | --- |
| Stage A audit | **INVALID**、600,000 candidate-long rows |
| 特徴量 | 162候補 → 100採用 |
| 除外 | 全欠損41、定数5、完全重複16 |
| 高相関 | 35組、報告のみ |
| feature schema SHA | `766cfcf1...d4deb` |
| native confidence依存 | 採用21列、exp263 v3実値parity完了 |
| Selector score OOF | expected-error MAE 3.742231、within10 logloss/Brier 0.355298/0.110596、3指標とも5/5 foldsでprior改善 |
| Score guard | **INVALID** |
| Hard top1 diagnostic | 8.362844、fixed 8.238332より+0.124512、guard FAIL |
| Hidden-like | spatial +0.438111、typewell-purged +0.407604 |
| Model | 10本、全SHA一致 |
| Stage C nested score | expected-error MAE 3.762776、within10 logloss/Brier 0.354702/0.110137、3指標とも5/5 foldsでprior改善 |
| Stage C leakage | PASS、40 models、25 partitions、18,919,945 compact rows |
| Stage C hard diagnostic | 8.420613、fixed 8.238332より+0.182281、FAIL |
| Stage C artifacts | outer-valid score 45,407,868 candidate-long rows、manifest SHA一致 |
| Stage D pooled RMSE | **INVALID**（旧計算 8.545568 → 7.805644） |
| Stage D fold | **5/5改善**、fold差 -1.272861 / -0.670234 / -0.358390 / -0.083249 / -1.241044 |
| Stage D distance | near -0.222916、mid -0.419414、1000+ -0.807155 |
| Stage D hidden-like | spatial -1.174830、typewell-purged -1.193025 |
| Stage D worst-well | `70925e23` **+17.446742**、事前上限+0.25を超過 |
| Stage D guard | **FAIL**。overall/fold/distance/hidden-likeはPASS、worst-wellのみFAIL |
| Stage D artifacts | 30 model全SHA一致、OOF 3,783,989行、欠損・非finite 0、8 output SHA一致 |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- 旧7候補案を廃止し、exp263が生成する12 surfaceを情報損失なくscore対象にした。
- native confidenceが全候補に同じ形で存在しない問題を、universal proxyとlearned dual scoreで分離した。
- 候補別scoreとdownstream compact featureの役割を分離した。
- 修正版Stage C v6で88列nested selectorを40 boosters完走し、score/leakage guardをPASSした。
  40/40 model SHAと25 partition manifestを監査し、74列compactのfold-safeな後段入力を確立した。
- Stage A/B、streaming OOF、calibration、重要度、74列compact adapter、current-test model ensembleを実装した。
- Kaggle CPU Stage A v1を0 boosterで完走し、exp263 3,783,989 rows / 773 wells / 5 foldsのmanifest・catalog SHAと100列feature schemaを固定した。
- Kaggle CPU Stage B v2を10 boostersで完走し、dual scoreは3指標すべて5/5 foldsでpriorを改善した。
- native confidenceは予測誤差objectiveでgroup重要度4.03%、`sigma_tvt`は5位だった。
- availability audit前の旧Stage C version 3も40 boostersを完走したが、100列schema依存のため結果は無効。
- Stage Dは承認scopeどおりcontrol 15 + add-only 15 = 30 GPU boostersを完走した。add-onlyは
  旧計算では改善して見えたが、feature availability leakageにより全比較を無効化した。
- 74 compact列はadd-only全体gainの70.96%を占め、上位は2 legal domainのwithin10/error top1候補値と
  anchorの差、`beam_mean`予測誤差、候補値range/stdだった。
- 修正版Stage D v3も30/30 boostersを完走し、有効なmatched ablationで10.476169 → 8.460811、
  5/5 folds改善を確認した。compactを連続meta-featureとして後段へ渡す仮説は支持された。

### 悪かった点

- 修正版Stage C v6のhard top1もfixedより+0.414200悪く、改善foldは1/5だった。
  score校正の合格をhard選択の合格とは扱えない。
- hard top1は固定blendより+0.124512悪く、near/1000+/hidden-like/worst-well guardも不通過だった。
- Stage Cのinner ensemble hard top1も固定blendより+0.182281悪く、hard化の不採用判断は変わらない。
- 773 well中346 wellで悪化し、worst-well回帰は+18.258274だった。
- Stage D add-onlyも773 well中303 wellで悪化し、243 wellが+0.25を超えた。worst-wellは
  `70925e23`でcontrol 5.804539からadd-only 23.251280へ悪化し、差は+17.446742だった。
- Stage Aで出たDataFrame fragmentation warningはconfidence/formula列の一括`pd.concat`化で解消し、回帰テストを追加した。
- pair shortlist自体は同一OOF上の事前監査で固定され、完全に独立なcandidate discoveryではない。

### リスク / 注意

- primitive、pair、fixedは高相関なので、違法な12本top1や見かけの小marginを作らない。
- inferenceではparity確認済みexp263 Stage 1 v3の21 namespaced confidence列と記録SHAを必須にする。
- 旧Stage Dはglobalには改善して見えたがfeature availability leakageで無効。修正版Stage D v3は有効だが、
  worst-well +14.482873でguard FAIL。ユーザー明示overrideによるreference submissionはLB 7.562を得たが、
  train-side guardをPASSへ変更せず、LB anchor更新とtrain-side採用を分ける。
- exp251 295列の最終採用数はStage A audit後にSHA固定し、事後に重要度で削らない。
- 修正版88 selector入力特徴の説明・重要度と重複相関は
  `selector_feature_readout_corrected_stage_b_v5.md`を正とする。旧100列readoutと旧Stage Dの
  74 compact重要度はfeature availability leakageの履歴としてのみ残す。

## 次

1. Public LB 7.562によりexp274 7.715を-0.153改善し、ML routeのLB anchorをexp264へ更新する。
   別routeのexp082 ensemble 7.601も-0.039で上回るが、ensemble anchor自体はexp082に維持する。
2. 修正版Stage D v3のworst-well guard FAILは固定し、hard selector、Viterbi、softmax TVT平均は不採用を維持する。
3. 再訪するなら、旧親leakageで無効化された`exp276`の固定0-booster target-free tail-risk readoutを、
   修正版Stage C/Dへ入力だけ差し替えて再監査する。feature/weight/quantile/guardは変更しない。

## 表記

用語は`KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせ、実験名や設定名を除いて日本語優先で記録する。

## OOF診断成果物

- `exp264_exp263_candidate_confidence_dual_selector_oof_selector_confidence_probe.ipynb`:
  corrected Stage C v6 strict nested scoreとStage D v3 final OOFの全well可視化。
- `exp264_exp263_candidate_confidence_dual_selector_oof_likpf_128_paths_probe.ipynb`:
  exp072 LikPF 500 particles × 128 seedsのexact replayとStage D v3 final OOFの比較。
- `artifacts/exp264_exp263_candidate_confidence_dual_selector_stage_d_v3_oof_viewer.csv`:
  corrected Stage D v3 add-only OOFを`id,tvt`へ変換したviewer用CSV。3,783,989行 / 773 wells、
  SHA `9fe0cfceda8b8e3d852c74352e0e4d7d6748f057b79354133b110e77173ce04b`。

notebookはJupytext sourceを正とし、Kaggle packageはprivate・CPU・internet offで準備済み。未実行・未pushであり、
hard selectorとStage D worst-wellのguard FAILは変更しない。
