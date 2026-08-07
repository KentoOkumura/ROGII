# 調査サマリー

このファイルは、長い調査レポートから主要な知見を抜き出し、エージェントが読みやすい形にまとめるために使います。

個別の完了レポートを実験番号・種類・トピックから探す場合は、[`README.md`](README.md)の生成索引を使います。

## 現在の発見

- 公式タスクは horizontal well の evaluation zone における `TVT` 回帰。
- まずは typewell `GR` と horizontal `GR` の correlation、`MD/X/Y/Z` trajectory、`TVT_input` 既知区間からの外挿を使うベースラインが手堅い。
- train-only formation columns は隠しテストで使えない前提のため、直接特徴に入れない。
- 類似タスクとしては well log correlation、lithology/geosteering、sequence regression が近く、2026-05-28 と 2026-06-04 に関連研究を調査済み。
- 2026-05-27 に Kaggle 公開ノートブック上位 20 件を `scoreAscending` で取得し、`docs/notebooks/rogii-wellbore-geology-prediction/score_ascending/solution_summary.md` に整理した。最優先で読むべきは Mitch の 8.905 系 writeup、次に Ravaghi/DWT/AeroRidge の runnable tree pipeline、h-blend 系は public-probe 色が強く hidden-safe に再構成してから使う。
- 2026-05-28 に Kaggle 公開ノートブック vote 上位 20 件を取得し、EDA / 重要知見軸で `docs/notebooks/rogii-wellbore-geology-prediction/eda_insights_summary.md` に整理した。最重要は Pilkwang の target-free alignment EDA、CDeotte の EDA Starter、Konbu17 の formation/ANCC plane-fit insight、Mitch の drift+NCC writeup。共通する結論は residual target、GroupKFold by well、fold-safe formation imputer、multi-scale NCC、beam/PF/DTW の補助特徴化、保守的 postprocess。
- 2026-05-28 に外部論文・過去ベンチマークを調査し、`docs/surveys/maybe_related_research.md` に整理した。実行価値が高いのは CWT / deterministic DTW の add-only GR alignment features、fold-safe formation top recommender / spatial surface imputer、learned GR-window similarity + constrained DP、candidate path disagreement features。FORCE 2020 は直接転用より blind-shift / validation discipline の参考にする。
- 2026-06-03 に Kaggle 公開 notebook を再取得し、`docs/notebooks/rogii-wellbore-geology-prediction/latest_public_notebooks_20260603.md` に整理した。scoreAscending 最新上位は PF/physical sel15 family、Sunny physical + 生成物を使った積み上げ、AeroRidge/PF/beam/TabICL route に寄っている。`LB 8.860/8.863` と明記された notebook は出ているが、見えている train well 用の物理処理 や public probe に寄りやすいため、まず PF/beam の scale別予測、likelihood、GR interpolation、候補間の食い違い を fold-safe features / diagnostics として使う。
- 2026-06-03 に recent discussion を `docs/discussions/rogii_discussion_insights_20260603.md` に整理した。formation columns / typewell `Geology` は hidden test で使える前提にしない。hidden rerun は visible 3 wells ではなく約 200 wells なので、runtime、id merge、per-well range、prediction-start continuity、candidate pairwise distance を提出前に見る。
- 2026-06-04 に、上記の最新 public methods と関連論文の対応を `docs/surveys/maybe_related_research.md` に追補した。既存調査は大枠で有効だが、PF/beam は geosteering の particle filter / ensemble uncertainty、DWT-DTW は well-log correlation、TabICL/生成物を使った積み上げ は tabular foundation model diversity、SG smoothing は signal-processing postprocess として位置付ける。次に実験化するなら、単体論文実装より PF/beam 候補分布 の fold-safe OOF feature 化を優先する。
- 2026-06-25 に GR matching の deep research を `docs/surveys/gr_matching_deep_research_20260625.md` に整理した。結論は、NCC/DTW/DWT を直接 add-only する方向は exp008/017/042/048 の履歴から弱く、self-GR や same-typewell cross-horizontal GR 直接転写も不採用。一方で exp091/093/099 は候補集合 oracle headroom、exp120 は neighbor drift prior の longtail 改善を示しており、次に試すなら `pf_ancc` を過小評価しない learned observation likelihood / candidate ranker、shape-aware multi-observation GR feature、+/-15ft datum ambiguity detector が優先。
- 2026-07-16 に候補パス、Beam/PF medoid、Self-GR履歴、候補別confidence、selector出力、exp238 TVT特徴の重要度・重複・相関を `docs/surveys/exp238_selector_tvt_feature_audit_20260716.md` に統合した。selector全入力は `docs/surveys/selector_feature_catalog_20260716.md` に分離し、exp237 320列、exp251 v4 295列×2目的、exp238 downstream adapter 35列を全件説明・重要度付きで記録した。
- 2026-07-19 に修正版exp264の12候補パス、hidden-safe selector入力88列、dual scoreと74列compact出力、hard選出結果、最終TVT LightGBM 347列の全特徴説明・15-model normalized gain/split重要度を `docs/surveys/exp264_selector_tvt_feature_audit_20260719.md` に統合した。hard selectorはfixed fallbackより悪化して不採用だが、compact add-onlyはclean 273 control 10.476169から8.460811へ改善しPublic LB 7.562。worst-well +14.482873のtrain-side guard FAILと、availabilityで除外した107列も明記した。
- 2026-07-16 に保存OOF 3,783,989行、81 candidate paths、13 selector/TVT outputsを揃え、全4,371ペア、均等平均190,026組、cross-fit triple 1,140組を`docs/surveys/candidate_path_blend_audit_20260716.md`で監査した。その後のユーザー判断でHMM+LGB系exp221/234/240とmodel outputsをscope外とし、78 raw/path候補で再集計。scope内最良単体はexp226 K16 9.427110、target-free固定最良はexp226 + w500の50/50で8.238331、5/5 folds。raw-test生成可能なcross-fit最良はexp226 + likPF + exact HMMの8.231651だが固定案との差は-0.006680だけ。Beamはexp226との平均を更新せずreserve、exp192/K8は診断候補に留める。後から当初監査sourceに未登録だったexp104 PF-Z seedbag 5本（RMSE 14.145856–14.587060）をsuperseded referenceとして追記し、`last_anchor`より良いknown inventoryは33本になった。後続共通基盤ではPF/Beam 27本を6代表へ縮約したcore 12候補だけをconfidence付きprimitive cache化する。全378 pairは当初監査28本の履歴に留め、有望8 pair・3 named tripleだけをvirtual adapterで扱う `last_anchor_better_candidate_confidence_pair_cache` を `KAGGLE_DIRECTION.md` に追加した。

- 2026-07-16 に上記契約を `exp263_last_anchor_better_candidate_confidence_pair_cache` として実装し、Kaggle CPU Stage 0 version 1を完走した。3,783,989 rows / 773 wells、source 9 groups / 12 gzip、candidate value/confidence各60 partitionsを951.444秒で生成し、manifest `85e60ac1...a26bb9e`、catalog `7cd74866...e9e6e0`を固定した。代表4 Parquetはrow/bytes/file/content/schema SHAが一致し、best-pair virtual loaderも757,738行・最大誤差0。Stage 0を後続OOF canonical inputとして採用する。Stage 1の6 current-test primitive / 5 pair / fixed exp226+w500 parityはsource解決待ちで、完了までraw-test inputにはしない。

- 2026-08-04 に exp490 の予測・アイデア・アルゴリズム再利用を `docs/surveys/exp490_reuse_strategy_20260804.md` へ整理した。exp413との固定10% blendはCV `7.884803→7.734534`、5/5 folds、MD/hidden-like全改善だがby-well p95/worst `+0.549195/+2.657049 ft`で非選択。signed physical-consensus特徴は既存32特徴のbeneficial AUCを`0.659944→0.671898`へ上げたがhard gateはtailを解決せず、exp413 blendの`>+1 ft` harm AUCも`0.552255`だった。したがって直接blend / hard routerを閉じ、K16 segment blocked-Huber model-evidence readout、小さいexp413 add-only mechanism/risk block、position/rate factorial、evidence PASS後だけのswitching dynamicsを順に検討する。

## 未解決の質問

- 隠しテスト分布に最も合う検証戦略はどれか。
- このタスクに最も近い過去コンペの解法はどれか。
- typewell `Geology` ラベルを使うべきか、`GR` シグネチャに限定するべきか。
- `TVT_input` 既知区間の最後の値から、evaluation zone の drift をどうモデル化するか。
- PF / beam / TabICL artifact route の public score と、自前 GroupKFold OOF のどこが一致し、どこが public-specific なのか。
