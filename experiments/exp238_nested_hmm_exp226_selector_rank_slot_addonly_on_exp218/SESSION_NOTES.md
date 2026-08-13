# exp238 セッションノート

## 2026-07-12 実装開始

`hmm_exp226_selector_rank_slot_addonly_on_exp218` を strict nested stacking として実験化した。過去の exp188/184/194 の fixed OOF artifact join は再利用せず、outer-valid wellをselector学習から完全に隔離する。

### Kaggle train 前コスト

- active variant: 1
- selector: 1 config × outer 5 × inner 4 = 20 boosters
- final LightGBM: 3 configs × outer 5 = 15 boosters
- 合計: 35 boosters
- parent/control再学習: なし
- Kaggle push: safety実装と静的検証後も、ユーザー明示承認まで禁止

### 実装内容

- outer 5 / inner 4 well GroupKFold split contract と overlap/coverage assertion。
- inner OOF selector score と inner model ensembleによるouter-valid score。
- selected/top2 relative delta、predicted-error rank/margin/ratio、source one-hot、candidate spreadのfold別特徴生成。
- global / near 000_050 / 1000+ / worst-well safety guard。
- guard通過時だけexp218 feature surfaceを再構築し、fold固有のnested selector featureを加えたlgb0/lgb1/lgb2を学習する条件分岐。
- guard不通過時はfinal 15 boostersを実行しない。

### 静的検証

- Jupytext train/inference変換: pass
- `jupytext --to ipynb --test`: pass
- `py_compile`: pass
- `ruff --select F821`: pass
- `make validate-exp EXP=exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`: strict pass
- ローカルnotebook実行は行っていない。初回実行はKaggleとする。

### 状態

実装・静的検証完了、Kaggle train未push。35 boostersの実行承認待ち。

## 2026-07-12 selector / final notebook分割

- `*_selector_train.ipynb`: CPU、nested selector 20 boosters、各modelのcandidate-long train/validを各120,000行に制限、full OOF/outer-valid予測は50,000 base rows chunk、fold別score artifactとsafety summaryを保存。
- `*_train.ipynb`: GPU。selector summaryの`guard_pass=true`を必須とし、5個のfold別score artifactをID/well/row/fold contractで検証してからexp218 lgb0/lgb1/lgb2の15 boostersを学習。
- 同じexp238内で管理し、実験番号は分けない。
- selector guard不通過ならGPU notebookは明示的に停止する。

## Kaggle selector v1/v2 failure

- v1: exp237 helper sourceがKaggle notebook inputに含まれず、import時にFileNotFoundError。親helper/settings/configをbootstrap dependencyとして同梱するpackage拡張を追加。
- v2: 89秒で`test base frame is missing auxiliary columns`。exp237 canonical configが後続raw-test inference用に`inference.use_test_base_as_dense_auxiliary=true`へ更新されていたことが原因。
- exp238はtrain-side OOF rebuildなので、selector/final notebookの両方で同flagをfalseへ明示上書きする。
- v1/v2ともselector booster学習開始前に停止。OOMや学習時間超過ではない。

## 2026-07-13 Kaggle selector v3完了

- Kernel: `kentookumura/exp238-nested-selector-train` v3
- Runtime: 14,049.055秒
- 3,783,989 rows / 773 wells / 11 candidates / 184 context features
- 20 selector boosters完了
- global delta RMSE: -3.089911
- near `000_050` delta RMSE: -0.609540
- `1000_plus` delta RMSE: -3.372225
- worst-well最大回帰: +37.680897
- `guard_pass=false`
- exp218 final GPU 15 boosters: 未実行
- inference / submit: 未実行
- 判断: worst-well上限+0.25を大幅超過したため不採用。guardを緩和しない。

## 2026-07-13 user-authorized guard override

- ユーザーがworst-well `+37.680897` のguard不通過確認後、GPU本学習へ進むよう明示指示した。
- selector artifactのID/well/row/fold/SHA contractは維持する。
- override対象はfinal train開始条件のみ。safety結果自体は変更しない。
- 実行予定: active variant 1、LightGBM 3 configs、5 outer folds、15 boosters。
- parent/control再学習なし。inference / submitは未承認。

### GPU train push

- Kernel: `kentookumura/exp238-nested-rank-slot-exp218-train` v1
- Kaggle id_no: `126872154`
- Accelerator: `NvidiaTeslaT4`（pull metadataで確認）
- run_on_push: true
- 初期CLI logsは空。実行中logsが空になる既知挙動のため、再pushせず同じkernelの完了連絡を待つ。

### GPU train v1 failure

- Runtime: 約717秒、booster学習開始前。
- guard overrideとselector artifact読込は成功。
- exp218 feature再構築開始時に`AttributeError: module 'exp218_source' has no attribute 'load_config'`。
- exp218 helperはfeature/model関数だけを持ち、config loaderを公開しない。bootstrap済み`exp218_source/config.yaml`をnotebook側で直接読み、local `cfg_get`で参照するよう修正。

### GPU train v2 failure

- Runtime: 約803秒、booster学習開始前。
- exp072 feature cache読込は成功。
- `add_anchor_columns`へconfigのローカル相対`data/raw/train`を渡し、raw horizontal fileが見つからず停止。
- bootstrap済みexp218 `settings.py`の`ExperimentPaths().train_data_dir`でcompetition inputのtrain directoryを解決し、anchor/GRWR双方へ同じ実在pathを渡すよう修正。

### GPU train v3 failure

- Runtime: 約1,098秒、booster学習開始前。
- competition raw train 773 files、exp218 380 featuresの再構築は成功。
- selector score artifact読込時、GPU runtimeで再計算したGroupKFold rowsとCPU selector artifactのroleが不一致でKeyError。
- selector artifactの`row_index + role`はnested学習時に保存した正のfold contractである。runtime GroupKFold再計算は診断だけに下げ、artifact roleからouter train/valid rowsを復元して最終学習へ渡すよう修正。

### GPU train v4 failure

- Runtime: 約1,940秒。exp218 380特徴再構築、5 foldのartifact role contract検証、fold 0 / lgb0 の学習700 iterationまでは成功。
- 最初のbooster終盤で`DeadKernelError: Kernel died`。Python例外ではなく、5 fold分score matrix、selector全context、base train/valid DataFrame、concat後DataFrame、LightGBM Datasetが同時常駐したことによるRAM OOMと判断。
- nested stackingのsplit/score契約、3 configs × 5 folds = 15 boosters、parent/control再学習なしは変更しない。
- v5修正: selector scoreはfold contract列だけ先に読み、11 score列は各foldの直前に1個だけ遅延読込。selector frameはcandidate/anchor/key列へ縮小。base+rank-slot学習行列は32列chunkで最終float32領域へ直接充填し、巨大なbase copyとconcat copyの二重保持を廃止。各booster/fold後にmodel・行列を明示解放する。
- Kernel v5を同じcanonical id `kentookumura/exp238-nested-rank-slot-exp218-train`へpush。`run_on_push=true`、`NvidiaTeslaT4`、15 boosters、parent/control再学習なし。push後pullで同kernelへの反映を確認。

### GPU train v5完了

- Kernel: `kentookumura/exp238-nested-rank-slot-exp218-train` v5
- Runtime: 15,635.579秒（約4時間20分）。3 configs × 5 outer folds = 15 boosters完走。
- OOM対策（fold単位score遅延読込、selector frame縮小、32列chunk行列構築、明示解放）は有効。
- pooled RMSE TVT: lgb0 7.972601414、lgb1 7.971169949、lgb2 7.990318775、lgb_mean 7.936689854。
- exp218 historical lgb_mean 8.475793752との差は名目-0.539103898。
- 生成物SHA: metrics `8eb1d21cc9ca41f16e4dfec724bf771acd936f3342d169aa6d0b0b7a1764193c`、OOF decompressed `0e7390ac3b3a432b1d432e432cb374cbf38da393a9b95f8f0d6c22732030010c`、model manifest `46ca999f0ee7ccae2dc784656059dc544d67f69d588fcf155dbb68edddefe4aa`。
- selector artifact outer foldsとGPU runtime再構成GroupKFoldは全fold不一致。artifact roleを正とするnested OOFはwell-isolatedだが、historical exp218とは同一fold比較でないため、名目改善をselector特徴の因果効果とは断定しない。
- selector worst-well guard +37.680897は不通過のまま。inference / submitは未実行。
- metrics/summary/manifest/feature importanceだけを`--file-pattern`で取得し、OOF/model archive全体はダウンロードしていない。

## 2026-07-13 user-authorized inference implementation

- ユーザーがselector worst-well guard不通過とhistorical exp218 fold比較caveat確認後、raw-test推論へ進むよう明示指示した。competition submitは未承認。
- 学習時の分離方針を維持し、CPU selector inferenceとGPU final inferenceを別Kaggle notebookにする。
- CPU selector inference: full trainをwell GroupKFold 4分割、selector 4 boosters。各modelのcandidate-long train/validは120,000行上限。raw-test 14,151行を4本で採点して平均し、候補値+anchor+score生成物を保存する。
- GPU final inference: 新規学習0 boosters。exp218 current-test 380特徴とselector rank-slot 35特徴を再生成し、train v5保存済み15 boostersを等重み平均する。
- raw-test parityがないexp109/114 OOF-only selector contextはfull-train中央値、train側も全欠損なら0で補完し、補完列/件数をselector summaryへ保存する。
- 再現性: selector model SHA、selector score decompressed SHA、15 train model SHA再検証、prediction decompressed SHA、feature schema SHA、submission SHAを保存する。PF/replay raw-test生成は親exp218/exp237の固定seed contractを継承するが、rerun一致確認前はdeterministic submission anchorと呼ばない。

### CPU selector inference v1 push

- Kernel: `kentookumura/exp238-nested-selector-inference` v1
- Kaggle id_no: `126922644`
- Runtime: CPU、internet disabled、competition source 1、kernel sources 9（pull metadataで確認）。
- 実行内容: full-train well GroupKFold 4分割、selector 4 boosters、raw-test scoreを4本平均。final LightGBMの再学習は0 boosters。
- push直後status: `RUNNING`。
- final GPU inferenceはこのkernelのselector score生成物に依存するため、v1完了確認後に同じexp238の別kernelとして開始する。competition submitは行わない。

### CPU selector inference v1 failure

- Status: `ERROR`、runtime約1,292秒。selector booster学習開始前。
- 最初の意味のある例外: raw-test enrichmentの`KeyError`。`tvt_dense_d`、`tvt_densew_d`、`tvt_dense50_d`、`dense_std`、`dense_dist`、`pf_vs_dense`の6列がtest frameになかった。
- 原因: train-side cache join用の`use_test_base_as_dense_auxiliary=false`をraw-test構築にも流用したため、exp073 raw-test cacheから6列を読まず、train-side exp072 cacheをtest IDへjoinして全列を生成できなかった。
- v2修正: train-sideは`false`を維持し、raw-test側だけdeep-copyしたconfigで親exp237正規契約の`true`を設定する。exp073 test cacheから6列を直接読み、raw-test enrichmentへ渡す。候補、fold、selector model数、学習上限、target境界は変更しない。
- 同じcanonical kernel `kentookumura/exp238-nested-selector-inference`へv2をpush。CPU、internet disabled、4 selector boosters、parent/final LightGBM再学習なし。push直後statusは`RUNNING`。

### Fold-matched saved-selector inference correction

- ユーザー指摘により、推論時に新規4 selectorをfull-train cross-fitする設計を再監査した。
- selector train v3はouter 5 × inner 4の20本を学習したが、保存したのは行数等のCSV manifestだけで、model本体を保存していなかった。
- selector inference v2の新規4本は全15 LightGBMへ共通のscore面を渡すため、final train v5のfold-specific nested変換と一致しない。v2が生成する出力は完了しても採用しない。
- 正規構成: selector trainで20本を`outer_fold + inner_fold`付きで保存。selector inferenceは再学習0本で、outerごとの4本平均から5 score面を作る。各final LightGBMは自分と同じouter foldのscore面だけを使う。
- raw-test parityがないcontextは中央値補完せず`NaN`にし、学習済みLightGBM selectorのnative missing-value routingを使う。
- selector train修正版の実行予定: active variant 1、selector config 1、outer 5、inner 4、20 CPU boosters。final LightGBM追加学習0、parent/control再学習なし。
- 保存専用再実行がGPU train v5の入力と同一であることを保証するため、selector v3 summaryだけを限定取得した。v3 nested score decompressed SHA（outer 0..4）をconfigへ固定し、修正版で1つでも不一致ならkernelを失敗させて推論へ進めない。
- 静的検証: selector train / selector inference / final inferenceのJupytext test、`py_compile`、`ruff --select F821`、strict experiment validationはpass。生成package上でselector inferenceに`.fit()`がなく、selector train inputを持つこと、final inferenceのouter-fold対応、CPU/GPU metadataを確認した。
- Kernel: `kentookumura/exp238-nested-selector-train` v4。同じcanonical IDへpushし、CPU、internet disabled、kernel sources 8をpull metadataで確認。push直後statusは`RUNNING`。
- v4完了後の必須確認: 20 model files、manifestのouter/inner完全被覆、各model SHA、5 nested scoreのv3 SHA一致。全条件pass後だけ学習0本のselector inferenceをpushする。

### Selector train v4完了・保存モデル監査

- Runtime: 約7,496秒。20 CPU boosters完走、notebookは正常終了。
- 限定取得: selector summary、selector model manifest、20 model txtのみ（約53MB）。nested score本体やoutput archive全体は取得していない。
- outer 0..4 × inner 0..3の20組を完全被覆。全20モデルのfile SHAがmanifestと一致。
- 全モデルのfeature schemaは184 context + candidate 3 = 187で一致。model txtのtree数とmanifest best iterationも一致（min 212、max 1200）。
- v4が再生成した5 nested scoreのdecompressed SHAはselector v3 summaryの固定値と全件一致。`selector_v3_nested_score_sha_contract=pass`。したがってfinal GPU train v5が学習したselector特徴と保存モデルが対応する。
- summary SHA: `f5d56ea3d9d25c45b61d0a40eef73288c5e47ba50929dab1fb858fa5708538cb`。
- model manifest SHA: `5ce9a670bb54e18517ba89b77e43cf7956be66354ad0f0db66fbde0d16e557b5`。
- selector guardはhistorical v3と同じ`false`のまま。推論承認とguard overrideの履歴は変更しない。

### Saved-selector inference v3 push

- Kernel: `kentookumura/exp238-nested-selector-inference` v3、id_no `126922644`。
- 実行内容: 保存済み20 selectorをSHA/schema検証して読み、outer foldごとの4本平均から5 raw-test score面を作る。selector fitは0本。
- CPU、internet disabled、selector train v4を含むkernel sources 10をpull metadataで確認。push直後statusは`RUNNING`。
- v1/v2のfull-train 4-selector再学習出力は不採用。final inferenceはv3固有status、20 model、5 score面、decompressed SHAを要求するため旧出力を受理しない。

### Saved-selector inference v3完了・監査

- Runtime: 約325秒。`selector_training_executed=false`、保存済み20モデル読込、outer 5 × inner 4被覆をログと生成物で確認。
- raw test: 14,151 rows / 3 wells。利用可能context 139、raw-test parityなし45列は`NaN`としてLightGBM native missing routingへ渡した。
- 5 score面は全て14,151行。ID/well/last_known_tvtは面間で完全一致し、candidate値と11 predicted-error scoreは全有限値。
- score decompressed SHA outer0..4: `1cdc3e74b9dd3b1857624f2fe2bcb49866426e64a6105e31726425bf1430246e`、`fb9c68b0eb82d0755cc5fe7cc5bec22ff339040c5ae7636276ceb1f6a3e1b120`、`343342fbcf306c7341714c96d40757bf37ede285de49ed4f2b42ae1a46d55423`、`156ad1451131cda56d467a109579bc31fc79249570870bf93dfe1a5c0e1bad5c`、`53be3c6955f8fe61edc52b22382dc8b38c5b52030e642a07501b83e59ccf87e2`。
- summary SHA: `da20fd8c40829d47ab6921509f5da504063a17e429e2f1fdbd8c75fe154c825b`。loaded-model audit SHA: `7b58aba498dfc1e2e507557f07b95fd7dff88cf5fbd260c086dbdbd6bf749908`。
- 次段: GPU final inference、新規学習0 boosters、保存済み15 LightGBM、outer fold対応selector score、competition submitなし。

### GPU final inference v1 push

- Kernel: `kentookumura/exp238-nested-rank-slot-exp218-inference` v1、id_no `126942754`。
- 新規学習0 boosters。保存済みfinal LightGBM 15本をSHA検証し、各modelへ同じouter foldのselector score面から作る35特徴を渡して等重み平均する。
- exp218 base 380 + selector 35 = 415特徴。T4、internet disabled、kernel sources 7をpush後pull metadataで確認。
- `submission.csv`は生成するがcompetition submit APIは呼ばない。push直後statusは`RUNNING`。

### GPU final inference v1 failure

- Runtime: 約24秒、raw-test replay開始前、新規booster学習なし。
- selector v3 status、20 models、5 score面、final 15 models、380+35 schemaの入力契約はpass。
- 最初の意味のある例外: `ModuleNotFoundError: public_notebook_replay_audit`、fallbackの`src.public_notebook_replay_audit`も未同梱。
- 原因分類: Kaggle bootstrap dependency不足。exp218 helper/config/settingsは同梱したが、self-contained raw-test replay moduleを同梱していなかった。
- v2修正: exp218の`public_notebook_replay_audit.py`を`exp218_source/`へbootstrap。learned likelihoodは未同梱generator fallbackを使わず、信頼済み`kentookumura/exp145-inference` v3をkernel sourceに追加し、既存raw-test特徴を必須利用する。
- exp145 sourceの存在とraw-test features/schema/summaryを限定取得で確認。14,151行のid/well setはselector testと一致。
- 同じcanonical kernelへv2をpush。T4、internet disabled、kernel sources 8、run_on_pushをpull metadataで確認。push直後statusは`RUNNING`。

### GPU final inference v2完了・submit-check

- Runtime: 約130秒、status `inference_completed_not_submitted`。
- 15 saved LightGBM（outer 5 × lgb0/lgb1/lgb2）を全て読込。各modelの`outer_fold`とselector score outer foldが一致。
- exp218 base 380 + selector 35 = 415特徴。5 selector score decompressed SHAとtrain model manifest SHAは入力契約と一致。
- prediction: 14,151 rows / 3 wells、min 11,590.633、max 12,240.465、mean 11,904.923402、std 278.717954。
- submit-check: sample submissionと列`id,tvt`、14,151行、ID順が完全一致。duplicate ID 0、missing target 0、全有限値。predictionとsubmissionのID対応値も一致。
- SHA: prediction decompressed `56ae1c9460597c9bac6a61edd629c8e5312fc4290178d97ddda57964766825f9`、feature schema `5de340e66a0797fb58b23176ab5ea73db8f57cd94cde508f50439cdc251e495f`、submission `dc0eb2e8f4581d0e91a8a6748f13cae17742e86539cbc234fa3a42fad6ec1f9d`、inference summary `f94ffaaa23d782f3882d0d899cf9d9b766d6a81e33856c133f2574fc678ff78e`。
- competition submissionは未承認・未実行。

### Code submission ref 54647064 failure / hidden-safe修正

- Kaggle submissions API詳細: ref `54647064`、kernel `kentookumura/exp238-nested-rank-slot-exp218-inference`、scriptVersionId `334800917`、status `COMPLETE`だがscoreなし。
- `errorDescription`: hidden datasetでnotebook rerun中に未処理例外。APIから具体的tracebackは取得できなかった。
- 静的監査で、public test 14,151行向けのexp145 rawtest feature、selector score CSV、exp073 cache、exp226 submissionのID依存を確認。hidden testではいずれもcoverage/key contractを満たさない。
- 修正後の提出notebookはcurrent test base replayを一度生成し、exact/self-GR HMM、exp226 K16 full-train inference、multiobs、exp145 learned-likelihood、GRWRをcurrent test上で再生成する。
- 保存済みouter 5 × inner 4 selectorは再学習せず、outerごとの4モデル平均scoreをメモリ内生成して同じouterの3 final LightGBMへ渡す。保存済み15 final LightGBMも再学習しない。
- inference kernel sourceはpublic-test selector inference / exp145 inferenceを外し、selector train v4、exp111/112 saved model/schema、exp209/223 HMM sourceを直接読む構成へ変更した。
- static checks: `py_compile` pass、`ruff --select F821` pass、Jupytext `--test` pass、`make validate-exp` strict pass、bootstrap ZIPにexp145/exp226/exp237/exp218 source/configが含まれることを確認。
- 次: canonical inference kernelをKaggle public testでrerunし、v2 predictionとのparity差、runtime、全current-test contractを確認後にcode submissionする。

### Hidden-safe final inference v3 push

- Kernel: `kentookumura/exp238-nested-rank-slot-exp218-inference` v3。
- T4、internet disabled、kernel sources 9。public-test selector inference / exp145 inference outputはsourceから除外し、selector train v4とrow-independent model/schema/sourceだけを使用する。
- 新規selector学習0、final LightGBM学習0。保存済み20 selectorと15 final modelをcurrent-test再生成特徴へ適用する。
- push直後status: `RUNNING`。competition code submissionはまだ再実行しない。

### Hidden-safe inference v3完了・code submission成功

- Kernel: `kentookumura/exp238-nested-rank-slot-exp218-inference` v3、scriptVersionId `334897917`。
- Notebook summary runtime: 407.557秒。kernel log最大時刻433.153秒。
- rows / wells: 14,151 / 3。features: exp218 base 380 + selector 35 = 415。
- 保存済み20 selectorをouter別4本平均で適用し、保存済み15 final LightGBMを推論。selector fit 0、final model fit 0。
- prediction min / max / mean / std: 11,590.6328125 / 12,240.4736328 / 11,904.9238281 / 278.7186584。
- SHA: prediction decompressed `ff7193ecfeab316498344f3a431a2db75bc3137aceb9ec39322f4c5219f3cb29`、submission `829709d6a4a27c7440412ae1b24aeab51734b30b19f59a78e9d0178dadcf6e0e`、selector surface decompressed `8d1bf1b4ca2cbbe72ad4c9dbbe1512eb3df1fd08901b3bd645729fa85557fb99`。
- code submission ref `54662073`は`COMPLETE`、errorDescriptionなし、Public LB `7.775`。
- exp218 ML anchor 7.843を-0.068、exp148 CPU 7.921を-0.146改善。exp082 ensemble anchor 7.601には+0.174届かない。
- 判断: exp238をML route submitted anchorへ更新。ensemble route anchorはexp082を維持する。

## 2026-07-15 raw-test copcf parity実装

exp238 saved-selector inferenceで184 contextのうち45列が全行NaNだった問題を再監査した。
内訳はexp109/exp114のOOF priorをtest IDへjoinできなかった`copcf_*` 41列と、test候補
artifactに含まれなかったexp226診断4列である。exp245は41列をtrain schemaから除外した
ため、正しいparity修正ではなくfeature-removal ablationとして残す。

exp238内に`*_rawtest_copcf_parity.py/ipynb`を追加した。既存の正規selector inference
notebookは上書きしていない。実装契約は以下。

- train context 184列とselector train v4保存済み20 modelを固定。再学習0。
- current test上でexp218 replay、HMM、exp226候補+診断4列、multiobs、enrichmentを再生成。
- test typewellをnative-overlap `1` / `0.999`の固定train clusterへ割り当てる。
- typewell priorは割当clusterのfull train wellだけ、spatial priorはfull train geometryだけをsourceにする。
- test-test edge/neighborは禁止。visible 3 wellとhidden 100+ wellで各wellの変換定義は変わらない。
- exp237と同じ41 `copcf_*`名/派生式を生成。missing context列0を必須とする。
- exp226診断4列は完全finiteを必須とする。trainにも存在するprior/std/distanceの自然なmissingは
  median/0埋めせずLightGBM native missing routingへ渡し、coverageを記録する。
- outerごとのinner 4 model平均から5 predicted-error score面を保存する。
- final LightGBM学習0、submission生成/competition submitなし。

静的検証:

- Jupytext変換 / `--test`: pass
- `py_compile`: pass
- `ruff check`: pass
- strict experiment validation: pass
- prepare helper tests: 10 passed
- package: `kentookumura/exp238-rawtest-copcf-parity`、CPU、internet off、
  `run_on_push=false`、kernel sources 7
- package config SHA: `262859bc6924309c54831166e3f4a9059357737386dcc1a1575cbb5561915ffb`
- bootstrap manifest: configとexp218 replay/config、exp226 source/config、exp237 source/configを確認

Kaggle実行は未実施。push前にはsaved selector 20適用、selector/final学習0、submissionなしを
再確認し、ユーザーの明示依頼後だけ`run_on_push=true`でpushする。

### Raw-test copcf parity Kaggle実行依頼

- 2026-07-15、ユーザーがKaggle実行を明示依頼した。
- 実行対象はCPUのraw-test parity監査1 variantのみ。
- 学習variant 0、LightGBM config 0、fold学習 0、booster学習 0。
- selectorは学習済みouter 5 × inner 4 = 20 modelを読み込む。final LightGBMは学習・推論とも行わない。
- 184 contextのうちtest側で41 `copcf_*`とexp226診断4列を再生成し、outer別5 selector score面とcoverage監査を保存する。
- parent/control再学習なし、GPU不使用、competition submissionなし。
- canonical kernelは`kentookumura/exp238-rawtest-copcf-parity`を維持し、`run_on_push=true`で初回実行する。

### Raw-test copcf parity v1 push

- Kernel: `kentookumura/exp238-rawtest-copcf-parity` v1、id_no `127304223`。
- 2026-07-15、`run_on_push=true`でpush成功。Kaggle URL:
  `https://www.kaggle.com/code/kentookumura/exp238-rawtest-copcf-parity`
- CPU、internet disabled、competition source 1、kernel sources 7。GPU学習・selector学習・final LightGBM学習・submission生成はいずれもなし。
- push前package config SHA: `0654f8821f1693fe593fc13f2a21e486321866798c5bde237c519fb568f676ef`。
- push前の既存kernel pullは`403 Forbidden`。push後pullは成功し、canonical ID、id_no、CPU/internet設定、7 kernel sourcesを確認した。
- local packageとpull後notebookは20 cellすべてのsourceが一致。Kaggle側の実行中ログはCLI取得せず、ユーザーからの完了/失敗連絡後に通常logsを取得して監査する。

### Raw-test copcf parity v1完了・監査

- Status: `rawtest_copcf_parity_completed_not_submitted`。ログ最大時刻560.479秒。
- current testは14,151行 / 3 wells。contextはID/wellを除きtrainと同じ184列。
- missing context列0、全行nonfinite列0、部分nonfinite列0、184列内nonfinite値0。
- 41 `copcf_*`は41/41列にfinite値があり、visible testでは全行finite。exp226診断4列も全行finite。
- typewell native-overlap `1` / `0.999`とspatial 2 priorはいずれも3 wellsでvalid rate 1.0。test-test edge/neighborは不使用。
- 保存済みselectorはouter 5 × inner 4の20 modelを完全被覆し、selector fit 0、final LightGBM fit/inference 0。
- outer別5 score面は各14,151行×25列、ID順一致、重複ID 0、numeric nonfinite値0。
- context decompressed SHA: `91d65d9c86fa9b83f9bcff5ddf509812620c20259865c64af2d9fd3f137f6d48`。
- schema SHA: `5a67fb13d30191bf4cce674ed5ac76cfdb64b2fc6a5c29eb9c16ae4462b99f69`。
- loaded-model manifest SHA: `7b58aba498dfc1e2e507557f07b95fd7dff88cf5fbd260c086dbdbd6bf749908`。
- score outer0..4 decompressed SHA: `16e4e4b34c92966d57fe2232207bb4a71f6f0b30765d99e949c7c9f0445af6b0`、`1b152399be3723c442c37e17470cdf9d468ec00ba13eef42a56194dac4d5d1e5`、`600cb78f3f4c2dc842ee5a6fc7077cf54c88ccd85f2ea1460d4159d37baa9d56`、`e1db0f48eb2a5a0a4a348e93d160f35686e3fd60fce4d3e034f55e8a3f329e78`、`b0cdaf6c917feb8ba97d67bac09d6163da95bac5114b6ae6fcd55614a626f7f6`。
- 結論: 45列NaN問題はvisible testが3 wellsであることではなく旧test generator不足が原因。full-train referenceだけで同じ184列を生成できた。NaN/parity修正はpassしたが、add-only設計とselector worst-well guard不通過は別問題として残る。

### Copcf parity final inference実装

- parity v1で通過した184 context generatorを、exp238 hidden-safe final inferenceへ接続した。
- 既存正規inference notebookは上書きせず、`*_inference_copcf_parity.py/ipynb`を別名で作成した。
- 保存済みselector 20本、保存済みfinal LightGBM 15本をSHA/schema検証して読む。selector/final/control学習は0。
- outerごとのinner 4 selector平均から35 rank-slot特徴を作り、同じouterの3 final modelへ渡す。全15予測を等重み平均する。
- final schemaはexp218 base 380 + selector 35 = 415。missing context列0、41 copcf、exp226診断4列をfail-fastする。
- test-test edge/neighbor、public-test行artifact、事前計算selector score CSVは使用しない。
- `submission.csv`はnotebook outputとして生成するが、competition submit APIは呼ばない。
- 実行コスト: inference variant 1、model config学習0、fold学習0、booster学習0、parent/control再学習なし。T4は推論runtimeとしてのみ使用する。
- 静的検証: Jupytext `--test`、py_compile、ruff、strict experiment validation、関連pytest 10件がpass。
- package: `kentookumura/exp238-copcf-parity-inference`、T4、internet off、run_on_push=false、kernel sources 12。
- bootstrap config SHA: `54e9c774134f714689eb7fbe785823e5025e5c58ad88df5a34a6ba560fcd39d1`。exp145/218/226/237のsource/config同梱を確認した。
- Kaggle実行は未実施。今回の変更はNaN修正のfinal inference portであり、add-onlyモデル構造は維持する。

### Copcf parity final inference Kaggle実行依頼

- 2026-07-15、ユーザーがKaggle実行を明示依頼した。
- 実行対象はparity-integrated final inference 1 variant。LightGBM config学習0、fold学習0、booster学習0。
- 保存済みselector outer 5 × inner 4 = 20本と、保存済みfinal LightGBM outer 5 × 3 configs = 15本を推論に使用する。
- parent/control再学習なし。T4はraw-test特徴生成と保存済みmodel推論にのみ使用する。
- `submission.csv`はnotebook outputとして生成するが、competition submit APIは呼ばない。
- canonical kernel `kentookumura/exp238-copcf-parity-inference`を`run_on_push=true`で初回実行する。

### Copcf parity final inference v1 push

- Kernel: `kentookumura/exp238-copcf-parity-inference` v1、id_no `127309057`。
- 2026-07-15、`run_on_push=true`、`NvidiaTeslaT4`でpush成功。Kaggle URL:
  `https://www.kaggle.com/code/kentookumura/exp238-copcf-parity-inference`
- push直後statusは`RUNNING`。internet disabled、competition source 1、kernel sources 12。
- inference variant 1、LightGBM config学習0、fold学習0、booster学習0、parent/control再学習なし。
- 保存済みselector 20本と保存済みfinal LightGBM 15本だけを使用し、competition submit APIは呼ばない。
- push時config SHA: `5626042c7670d0d4d31dc53254909a04cd1aba7b654aec7bb722ea98f097df9b`。source/packageで一致。
- push前の既存kernel pullは`403 Forbidden`。push後pullは成功し、canonical ID、id_no、T4/internet設定、12 kernel sourcesを確認した。
- local packageとpull後notebookは22 cellすべての結合sourceが一致した。継続監視と実行中logs取得は行わず、ユーザーの完了/失敗連絡後に監査する。

### Copcf parity final inference v1完了・監査

- Kaggle statusは`COMPLETE`。summary runtimeは479.031秒、ログ最大時刻は約502.233秒。
- Status: `hidden_safe_copcf_parity_final_inference_completed_not_submitted`。14,151行 / 3 wells。
- context parityはpass。184 context、41 `copcf_*`、exp226診断4列を生成し、missing context列0、部分nonfinite列0、fallback行0。
- test-test neighborは不使用。public-test行artifactも不使用。
- selector manifestはouter 5 × inner 4の20組を完全被覆。outer別5 score面は各14,151行・4 model平均で、全decompressed SHAがsummaryと一致。
- final schemaはexp218 base 380 + selector 35 = 415列。保存済みfinal LightGBM 15本で推論し、selector/final/control学習は0。
- prediction範囲は11,591.132〜12,240.252、平均11,904.986795。fallback行0。
- `submission.csv`はsample submissionと14,151行・2列・ID内容/順が完全一致し、重複・欠損・NaN/Infなし。submit-checkはFAIL/WARNとも0でPASS。
- competition submit APIは未実行。selector worst-well guardは既知どおり不通過であり、今回のNaN/parity修正では緩和していない。
- output取得時はKaggle CLIの既定paginationでroot `submission.csv`が最初の取得対象から外れたため、`--file-pattern '^submission[.]csv$' --page-size 200 --force`で取得した。notebook側ではsummary保存前に同ファイルのSHAを計算済みで、生成失敗ではない。
- SHA: submission `c1a16392519e14f2b4ca9c1d86668e7f13d0f7bc20088c165f0aedcec6b05d30`、prediction decompressed `d88e9ca83197267b0d749953a0fa9ff506e3a2c2a0ddb6f796bf13c84f0f5fec`、context decompressed `6cde0ad35e9fd4e91d0c6ccbb1a117caa2f8d99113317409cfec22316b28c8fe`、selector surface decompressed `a7b317035eced14c55190b624332e1059aeda4f0a3993abaea77b487332b4c56`。
- feature schema SHA `5de340e66a0797fb58b23176ab5ea73db8f57cd94cde508f50439cdc251e495f`、loaded selector manifest SHA `7b58aba498dfc1e2e507557f07b95fd7dff88cf5fbd260c086dbdbd6bf749908`、summary SHA `fe4644f11dcfa9dd47ce3e15489bea86ba344598ccdab0a4824e6107206beec0`。

### Copcf parity final inference v1 提出前再確認

- ユーザー指示によりexp255より先にexp238を進めるため、既存Kaggle outputの提出前検証を再実行した。
- `.agents/skills/kaggle-submit-check/scripts/check_submission.py`と`validate_submission.py`はいずれもPASS。14,151行 / 2列、sample header・行数・ID順一致、重複・欠損・NaN/Infなし。
- submission SHAは既記録どおり`c1a16392519e14f2b4ca9c1d86668e7f13d0f7bc20088c165f0aedcec6b05d30`。
- kernel `kentookumura/exp238-copcf-parity-inference` v1はT4、internet off、12 kernel sources。再学習は行わない。
- 既存Public LB 7.775のref `54662073`はparity修正前の別kernel/version。184 context finite版v1は未提出であり、competition submitは明示確認待ち。

### Copcf parity final inference v1 code submission

- 2026-07-15、ユーザーの明示指示によりkernel `kentookumura/exp238-copcf-parity-inference` v1の`submission.csv`をcode submissionした。
- Kaggle submission ref: `54725625`。submit直後statusは`PENDING`、descriptionは`exp238 copcf parity finite 184 context`。
- 提出物は14,151行、SHA `c1a16392519e14f2b4ca9c1d86668e7f13d0f7bc20088c165f0aedcec6b05d30`。提出直前の2 checkerはFAIL/WARN 0でPASS。
- このv1は184 selector context、41 `copcf_*`、exp226診断4列がすべてfiniteであるparity修正版。保存済みselector 20本とfinal LightGBM 15本だけを使用し、推論時学習は0。
- ユーザー指示どおり継続監視は行わない。Public LBは未確定であり、既存ref `54662073`の7.775と混同しない。

### Copcf parity final inference v1スコア確定

- 2026-07-16、ユーザーのスコアリング完了連絡後にKaggle submissionsを1回取得した。継続監視は再開していない。
- submission ref `54725625`は`COMPLETE`、Public LB `7.842`。
- parity修正前hidden-safe v3 ref `54662073`の`7.775`より`+0.067`悪化し、exp218 ref `54457577`の`7.843`とは`-0.001`のほぼ同等だった。
- 184 context、41 `copcf_*`、exp226診断4列をfinite生成できることとhidden-safe実行の健全性は維持される。一方、NaNを有限値へ置き換えたparity版がPublic LBを改善した証拠は得られなかった。
- exp238のML route anchorはref `54662073` / Public LB `7.775`を維持し、parity版ref `54725625`は不採用とする。

## 2026-07-16 exp238 OOF selector-confidence plot notebook実装

指定された Kaggle notebook `kentookumura/exp083-v12-ml-oof-known-tvt-probe`
（`scriptVersionId=333830051`）を参照し、同じexp238内に次を追加した。

- Jupytext source: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218_oof_selector_confidence_probe.py`
- 正規notebook: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218_oof_selector_confidence_probe.ipynb`
- package: `kaggle/oof_selector_confidence_probe/`
- canonical kernel metadata: `kentookumura/exp238-oof-selector-confidence-probe`

可視化契約:

- exp148 OOFの代わりにexp238 final train v5の保存済み`lgb_mean_pred_tvt`を青線で表示する。
- selector scoreはcandidateの予測絶対誤差なので、rowごとの最小scoreを「最も信頼したcandidate」と定義する。
- 5 outer foldのscore gzipはchunk読込し、各rowが`role=valid`のfoldだけを使う。outer-train scoreやfold平均は使わない。
- TVT panelにselector top-1 candidate pathを橙破線で重ねる。これはdiagnosticでありexp238 final predictionではない。
- 下段に`top2 predicted error - top1 predicted error` marginと、11 candidateのtop-1色分け帯を表示する。
- title/manifestにはexp238 OOF RMSE、selector top-1 RMSE、dominant top-1 candidate/share、margin、switch数を保存する。
- selector historical worst-well guard不通過をplot注記とsummaryに残し、direct replacement採用と誤認させない。
- 全773 wellsのPNG、plot manifest、global candidate distribution、plots zip、summary JSONをKaggle outputへ保存する設計。

実行契約:

- 新規selector/final LightGBM学習0、PF/Beam/HMM/exp226候補再生成0、submission生成0、competition submit 0。
- CPU、internet disabled、`run_on_push=false`。
- kernel sourcesはexp072 cache、exp238 final train、exp238 selector train、exp209、exp223、exp226の6件。
- selector summaryが宣言する5 scoreのdecompressed SHA契約をsummaryへ引き継ぎ、実行時にfilename/row/fold/id/well/coverageをfail-fastする。

静的検証:

- `py_compile`: pass
- `ruff check --select F821`: pass
- Jupytext convert / `--test`: pass
- canonical/package notebook JSON parse: pass
- canonical/package 16 cell source一致、output 0、execution count 0
- metadata id/title slug一致、CPU/internet-off/run-on-push-false、kernel sources 6: pass
- `make validate-exp EXP=exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`: strict pass
- `__file__`、model `.fit()`、submission生成/submit call: なし

file SHA:

- Jupytext source: `7a4a825ff68b16c436606548a150715868e1cc99de22365e31ace4799f75c5e9`
- 正規notebook: `ba83871a4c4b08ce40d515ab5d4dfb754642a6cbd62f8717cf683e1b56c8a88e`
- package notebook: `e1ec4df98549afe0e8612cb8f590a11b5b39fea92e2b1264540df3c92f3e3fb2`
- metadata: `ac50c3365b9b750850a528dafd4e8c9ccada49a781ad4650b35e6934df76bb52`

Kaggle push / notebook実行は依頼範囲に含めず、未実施。したがってplot/manifest/summary生成物と実行runtimeはまだない。

### OOF selector-confidence plot Kaggle実行依頼

- 2026-07-16、ユーザーがKaggle実行を明示依頼した。
- 実行対象はall-well diagnostic plot 1 variant。CPU、internet disabled。
- model config学習0、fold学習0、booster学習0、parent/control再学習なし。
- 保存済みexp238 final OOF、outer-valid selector score、exp072/209/223/226候補を読み、773 wellsのplotを生成する。
- selectorは各rowが`role=valid`のouter fold scoreだけを使用し、top-1 candidateとtop2−top1 marginを表示する。
- submission生成・competition submitは行わない。
- canonical kernel `kentookumura/exp238-oof-selector-confidence-probe`を維持し、`run_on_push=true`で初回実行する。

### OOF selector-confidence plot v1 push

- Kernel: `kentookumura/exp238-oof-selector-confidence-probe` v1、id_no `127444478`。
- URL: `https://www.kaggle.com/code/kentookumura/exp238-oof-selector-confidence-probe`
- 2026-07-16、CPU / internet disabled / `run_on_push=true`でpush成功。
- 実行variant 1、model config学習0、fold学習0、booster学習0、parent/control再学習なし、submissionなし。
- push前のcanonical pullは`403 Forbidden`。初回作成前のため、push後pullで存在を確認した。
- push後metadataはCPU、internet disabled、competition source 1、kernel sources 6、id/title slug一致。
- local packageとpull後notebookは16 cellすべてのsourceが一致した。
- package notebook SHA `e1ec4df98549afe0e8612cb8f590a11b5b39fea92e2b1264540df3c92f3e3fb2`、push時metadata SHA `145b382e4d46ec7ed379dcce731fe760dde0c1a2acbc4efe2805cc7a80ba9d17`。
- 実行完了まで同じcanonical kernelを監視し、別slugへ再pushしない。

### OOF selector-confidence plot v1完了・監査

- Kernel `kentookumura/exp238-oof-selector-confidence-probe` v1は`COMPLETE`。CPU / internet disabled、ログ最大時刻656.56秒（約11分）。
- summary statusは`diagnostic_plots_completed_not_submitted`。学習、PF/Beam/HMM/exp226候補再生、submission生成、competition submitはすべて0。
- baseは3,783,989行 / 773 wells。outer-valid行数はfold 0〜4で757,738 / 756,650 / 756,255 / 757,101 / 756,245、合計3,783,989で厳密に1回ずつ覆った。
- 773 / 773 wellsのPNG、manifest 773行（well/pathともunique）、top-1 distribution、plots zip、summary JSONを生成した。代表図`000d7d20.png`を取得し、青のexp238 OOF、橙破線のselector top-1、confidence margin、色分けtop-1帯、候補legendの表示を目視確認した。
- global RMSEはexp238 OOF `7.936690030870403`、selector top-1 `8.512264240223669`、LikPF `11.594897672217703`、exp226 K16 `9.42710967407494`。margin mean / p50 / p90は`0.3192525804042816` / `0.15628385543823242` / `0.7142080307006837`。
- top-1の最多候補は`Self-GR HMM`で1,205,794行 / 31.8657%。以下`PF ANCC` 20.3814%、`exp226 K16` 17.5745%、`Likelihood PF mean` 12.8689%、`LikPF/HMM 50:50` 8.3958%。
- selector top-1はあくまでdiagnostic。historical worst-well regression `+37.6808967590332`のguard不通過は維持され、direct replacementやanchor更新は行わない。
- artifact SHA: manifest `0cef08141fe7cb2f2941064c3d5bba2b65f33fef49967b77660f1f30bd3c2f2a`、distribution `639e2fb475bfb5c8859c03622c47b2b56fa0aa10b565ca17c31567e1aef01dae`、plots zip `e4b91e29d38f7f5a6e5a3bfcb8fdb1f326bd26e0800e2b5709930866282e975c`、summary `56af95ee87a2303c8f7c493462292fcdf40dee5a5a5d5fecd5f9177ae0a1e7a0`。
- 必要な実ファイルだけローカル`kaggle/output/oof_selector_confidence_probe_v1/`へ取得し、summaryのscope/status、manifestの773 wells、distributionの行数/share合計、SHAを検証した。

### OOF selector-confidence plot exp083配色修正

- 2026-07-16、ユーザー指示により、参照元exp083 v12と共通系列の色を完全に一致させる。
- true TVT `black`、ML OOF `#e11d48`、PF ANCC `#1f77b4`、Beam `#ff7f0e`、LikPF `#2ca02c`、exp226 `#a16207`、exp209 HMM `#7c3aed`、HMM band `#8b5cf6`、-Z minmax `#db2777`、grid `#e2e8f0`をexp083から固定した。
- selector top-1 pathはBeamと誤認しないよう、exp083のlast-anchorと同じ`#64748b`の灰色破線に変更した。top-1候補の識別は色帯と凡例を正とする。
- selector色帯も共通candidateは上記の色を再利用し、exp083にないcandidateはstable Tableau colorsを固定した。summaryに`plot_colors` contractを追加した。
- データ、selector score、top-1決定、RMSE、fold contract、生成物仕様は変更なし。実行variant 1、model config/fold/booster学習0、parent/control再学習なし、submissionなし。
- Jupytext convert/test、`py_compile`、`ruff --select F821`、strict experiment validationはpass。canonical/package notebookは同一SHA。
- 修正後SHA: source `08bfa8241c0b48c28c6320b2b4f6baf5ca26fd62c70a4003f055e6bb56e036e1`、canonical/package notebook `a20765c1b5b2477bdda34d6eebebf78396759cad2c96aaec52bbd57055cdada3`、metadata `145b382e4d46ec7ed379dcce731fe760dde0c1a2acbc4efe2805cc7a80ba9d17`。
- 同じcanonical kernel `kentookumura/exp238-oof-selector-confidence-probe`のv2として実行し、代表図の色を目視確認する。

### OOF selector-confidence plot v2 push

- 同じcanonical kernel `kentookumura/exp238-oof-selector-confidence-probe`にversion 2としてpush成功。id_noは`127444478`を維持し、別slugは作っていない。
- push前pullでv1 kernelの存在を確認し、push後pullでCPU、internet disabled、6 kernel sources、canonical id/titleと修正後16 cell sourceの一致を確認した。
- pulled sourceにexp083色`#1f77b4`、`#ff7f0e`、`#2ca02c`、`#e11d48`、`#a16207`、`#7c3aed`、`#64748b`が存在し、v1固有の`#1d4ed8`、`#f97316`がないことを確認した。
- 実行variant 1、model config/fold/booster学習0、parent/control再学習なし、submissionなし。push直後statusは`RUNNING`。

### OOF selector-confidence plot v2完了・配色監査

- v2は`COMPLETE`。CPU / internet disabled、ログ最大時刻938.856925秒（約15分39秒）。学習、候補再生、submission生成、competition submitはすべて0。
- 3,783,989行 / 773 wells、outer-valid 5 foldの合計3,783,989行、773 / 773 plots、global metrics、top-1 distributionはv1と一致した。
- v2 summaryの`plot_colors` contractを検証。ML OOF `#e11d48`、PF ANCC `#1f77b4`、Beam `#ff7f0e`、LikPF `#2ca02c`、exp226 `#a16207`、HMM `#7c3aed`、selector top-1 `#64748b`で参照元exp083と一致した。
- selector top-1色帯の共通candidateも同色であることをsummaryで検証した。manifest 773 unique wells/paths、distribution行数合計とshare合計はpass。
- 代表図`000d7d20.png`を取得して目視確認。exp238 OOFのローズ、PF ANCCの青、Beamの橙、LikPFの緑、exp226の茶、HMMの紫、selector top-1の灰色破線、候補色帯と凡例の表示は正常。
- local outputは`kaggle/output/oof_selector_confidence_probe_v2/`にsummary、manifest、distribution、代表PNG、logのみ取得した。
- SHA: manifest `0cef08141fe7cb2f2941064c3d5bba2b65f33fef49967b77660f1f30bd3c2f2a`、distribution `639e2fb475bfb5c8859c03622c47b2b56fa0aa10b565ca17c31567e1aef01dae`、plots zip `305b1c73e9627839fd3a74d18bef8095367d7aba4589ea1e9f1bf2550cca97ce`、summary `0aed162b50ccd0cdb0cc15d9ff78578ebbc1b057c7d7829b823709929986d8f7`、representative PNG `28ac781a6c07f57d3615dd7ed338043d6622c5c48703c7d39a368fa71df3c8fe`、log `25a34e3db00b7466b3def86053303b9ea7f2509dbdf313ade03d91b4b66d0207`。
- v1の図はv1履歴として維持し、今後の配色付き図はv2を正とする。selector worst-well guard不通過とdiagnostic-only判定は不変。

## 2026-07-18 exp238 likelihood-PF 128 paths plot notebook実装

ユーザー指定のKaggle notebook `kentookumura/exp238-oof-selector-confidence-probe`
（`scriptVersionId=335655690`）を元に、likelihood-PFの128 seed trajectoryを全て
表示する別バージョンを同じexp238内へ追加した。既存notebookと既存canonical kernelは
上書きしていない。

追加ファイル:

- Jupytext source: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218_oof_likpf_128_paths_probe.py`
- 正規notebook: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218_oof_likpf_128_paths_probe.ipynb`
- package: `kaggle/oof_likpf_128_paths_probe/`
- 新canonical metadata: `kentookumura/exp238-oof-likpf-128-paths-probe`

可視化・再現性contract:

- exp072 v2と同じraw-GR Gaussian likelihood-PF、500 particles × 128 seedsをraw trainからwell単位で再生する。
- seed baseは`stable_seed("likpf", "train", well_id)`、seed indexは0..127。thread並列単位はwellで、各wellのseed系列を固定する。
- 128本は全て青`#2563eb`、linewidth `0.55`、alpha `0.06`で描画する。
- true TVTは黒linewidth 2.5、exp238保存済み`lgb_mean_pred_tvt` OOFはrose `#e11d48` linewidth 2.1で、不透明・前面表示する。
- raw evaluation row IDとexp072 cache IDをwellごとに検証する。128 seed meanをfloat32 deltaへ戻し、保存済み`likpf_mean_d`とのexact parityをfail-fastする。
- true TVTはPF生成後のplot/RMSEだけに使用し、state update、seed選択、trajectory選択には使わない。
- 約1.94 GB相当になる全well seed bankは保存せず、batch処理中だけ保持する。出力は773 well PNG、manifest、plots zip、summary JSONの設計。
- diagnostic overlayのみ。model fit 0、LightGBM booster 0、prediction blend 0、submission生成0、competition submit 0。

Kaggle package contract:

- CPU、internet disabled、`run_on_push=false`。
- kernel sourcesはexp072 train feature cacheとexp238 final trainの2件。
- configはprepare scriptのzip bootstrapに含め、canonical notebook 16 cellsとpackageのbootstrap後16 cellsのsource一致を確認した。
- metadata id/title slugは`exp238-oof-likpf-128-paths-probe`で一致した。

静的検証:

- `py_compile`: pass
- Ruff full check / `--select F821`: pass
- Jupytext convert / `--test`: pass
- canonical/package notebook JSON parse、output 0、execution count 0: pass
- metadata CPU/internet-off/run-on-push-false、competition source 1、kernel sources 2: pass
- `make validate-exp EXP=exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`: strict pass
- `__file__`、model `.fit()`、submission生成/submit call: なし

file SHA:

- Jupytext source: `d99ffbc787fbd68175d9f37163ebdf37e0b55f861a0dac3b80fbc814657569c5`
- 正規notebook: `3ea17c66dd1146133680ef4a0873ca32dd9769f9e1db687eb2360562193428e2`
- package notebook: `5ed80d4baff10f577851e19e215106e15b693f1e762cd5688c8ca4898cf8d2dd`
- metadata: `22b97cda7d68f4017cc0232e181c7fdb4be4022bc2b403c9fc432d278f20f0fd`

Kaggle push / notebook実行は今回の依頼範囲に含めず、未実施。したがって128-path plot、
manifest、summary、runtime、saved-mean parityの実行結果はまだない。初回実行はユーザーの
明示依頼後、上記の新canonical kernelへversion 1としてpushする。

### Likelihood-PF 128 paths plot Kaggle実行依頼

- 2026-07-18、ユーザーがKaggle実行を明示依頼した。
- 実行対象はall-well likelihood-PF 128-path diagnostic plot 1 variant。
- CPU、internet disabled。PFは500 particles × 128 stable seeds × 773 wellsをraw trainから再生する。
- LightGBM config 0、fold学習0、booster 0、selector/final model再学習0、parent/control再学習なし。
- true TVTと保存済みexp238 `lgb_mean` OOFを描画するが、prediction blend、submission生成、competition submitは行わない。
- canonical kernelは`kentookumura/exp238-oof-likpf-128-paths-probe`。`run_on_push=true`でversion 1をpushし、同じslugで完了まで監視する。

### Likelihood-PF 128 paths plot v1 push

- pre-push canonical pullは`403 Forbidden`。初回作成前のため想定どおりで、別slugは作っていない。
- canonical kernel `kentookumura/exp238-oof-likpf-128-paths-probe`へversion 1をpushした。URL: `https://www.kaggle.com/code/kentookumura/exp238-oof-likpf-128-paths-probe`。
- Kaggle id_noは`127756003`。CPU、internet disabled、competition source 1、kernel sources 2、canonical id/title一致をpost-push pullで確認した。
- local packageとpull後notebookは17 cellsのsource textが全て一致した。Kaggle pullはsourceをlistではなくstringへ正規化するためJSON型だけが異なる。
- package notebook SHA `5ed80d4baff10f577851e19e215106e15b693f1e762cd5688c8ca4898cf8d2dd`、push時metadata SHA `e7b838b836c775d1707a1ca1c31a5460f7ff68b72e7459c8cb4cf2e54c37cdb0`。
- 実行variant 1、model config 0、fold学習0、booster 0、parent/control再学習なし、submissionなし。同じcanonical kernelを完了まで監視する。

### Likelihood-PF 128 paths plot v1実行中

- 起動ログでbootstrap 26 files、exp072 cache、exp238 OOF、competition raw dataの解決を確認した。
- Kaggle CPUは4 coreのため、config上限8に対してruntime `n_jobs=4`。PFは500 particles × 128 seedsのまま。
- base contractは3,783,989行 / 773 wells。exp238 `lgb_mean` OOF RMSEは`7.936690030870403`で既存値と一致した。
- Numba likelihood-PF helper compile後、最初の8 / 773 wellsを142.7秒でreplay・parity検証・plot保存した。単純外挿の全体見込みは約3.8時間。
- ローカル`kaggle kernels logs -f`だけを切り離した。Kaggle statusはその後も`RUNNING`で、Kaggle実行は継続中。

### Likelihood-PF 128 paths plot v1完了・監査

- canonical kernel `kentookumura/exp238-oof-likpf-128-paths-probe` v1は`COMPLETE`。summary runtimeは`14,067.881`秒（約3時間54分28秒）、CPU / internet disabledで完走した。
- 3,783,989行 / 773 wellsを処理し、773 / 773 wellsのPNG、manifest 773行、plots zip、summary JSONを生成した。manifestはwell/pathとも773 unique、各wellの`pf_seed_count=128`、`pf_particles_per_seed=500`を満たす。
- 128 seed平均とexp072保存済み`likpf_mean_d`は773 / 773 wellsでexact parity。global max abs差とweighted mean abs差はいずれも`0.0`だった。
- global RMSEはexp238 `lgb_mean` OOFが`7.936690030870403`、128-seed PF平均が`11.594897672217703`。manifestのfull-row squared-error sumから再計算してsummaryと一致した。
- 描画は128本すべて青`#2563eb`、alpha `0.06`、linewidth `0.55`。true TVTは黒、exp238 LGB OOFはrose `#e11d48`の不透明線である。代表図`000d7d20.png`を取得し、128-path分布、truth、LGB OOF、凡例、反転TVD軸を目視確認した。
- 142 wellsは描画時だけ6,000点上限へ間引き、全773 wells合計の描画点は3,672,435。parityとRMSEは間引き前の全3,783,989行で評価している。
- model fit、LightGBM booster、prediction blend、submission生成、competition submitはいずれも0。これはdiagnostic outputであり、exp238 predictionやML route anchorを変更しない。
- local outputは`kaggle/output/oof_likpf_128_paths_probe_v1/`にsummary、manifest、代表PNG、kernel logのみ取得した。全plotはKaggle outputのplots zipを正とする。
- SHA: manifest `5dbb76afcb70f6bbc94df138d2d5232050dd6c14fa80a52e237134cf71abaf13`、plots zip `ab58067cd70fd73e921bdf0bab84659379a610c0f0461ffa1e70f617e6c16797`、summary `543bb733d93d2eddc0085de9b6aa5e65a905dd76390cf971a7245f36ccfff262`、representative PNG `951ddd13ce15703e330e1cf829a32a65ff3dcb585d8052740ee72aaf4df9d17a`、kernel log `0ec8137b59ed0eb9c28319ec29eb6c7bb9f69b65773907ce1d22e6495cff0c78`。

## 2026-07-18 OOF selector-confidence PNG共通typewell順

ユーザー指定のcanonical notebook `kentookumura/exp238-oof-selector-confidence-probe`
（`scriptVersionId=335655690`、local上のv2実装と一致）について、well ID辞書順だった
PNG出力をexp065共通typewell順へ変更した。

実装contract:

- 共通typewell対応表はexp065 `common_typewell_cluster_assignments.csv` SHA
  `dcda8588cc1dd9261bafae7de00c890393e38b8a0ca0eb86fbba18a2cffc4a50`を読む。
- `method=native_overlap` / `threshold=0.999`を固定し、773 wells / 54 groups / well重複0 / coverage欠損0をfail-fastする。
- exp065のdeterministic `cluster_id`順、同一group内`well_id`順へstable sortする。cluster IDはgroup size降順、同サイズでは代表well ID順に作られている。
- PNG名を`typewell_{typewell_order:04d}_{well}.png`へ変更する。先頭は
  `typewell_0001_09441b8d.png`、末尾は`typewell_0054_f5859199.png`。
- manifestへ`plot_order`、`typewell_order`、group内well順、method/threshold、cluster ID/size/代表well、`plot_filename`を追加する。
- plots zipはglob再sortではなくmanifestの`plot_filename`順に格納し、Kaggle output UI、manifest、zip member順を一致させる。
- summaryへordering contract、対応表path/SHA、54 groupsを保存する。
- kernel sourceへ`kentookumura/exp065-typewell-supertype-cluster-cv-audit-train`を追加し、合計7 sourcesとした。

実行contract:

- diagnostic variant 1、model config 0、fold学習0、booster 0、parent/control再学習なし。
- 保存済みOOF/candidate/selector scoreの値、plot内容、全RMSE、配色は変更しない。
- CPU、internet disabled、candidate再生成0、submission生成0、competition submit 0。

静的検証:

- `py_compile`: pass
- `ruff --select F821`: pass
- Jupytext convert / `--test`: pass
- canonical/package notebook 16 cellsのsource一致、output 0、execution count 0、byte SHA一致: pass
- exp065 mapping 773 wells / 54 groups、filename uniqueかつ辞書順=typewell順: pass
- config/metadata kernel source 7件、重複0、順序一致: pass
- `make validate-exp EXP=exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`: strict pass
- `__file__`、model `.fit()`、submission生成/submit call: なし

file SHA:

- Jupytext source: `43235d8e97ae8d34c284f60a7e729d8f3d38c873139734b359a0b124420ffd70`
- 正規/package notebook: `3e7dd62d1e6051f944e626ad27f6e06d0384e1402b3fdc55937c62e363e7020d`
- metadata: `2e7f1b65bbdb5e14dd5917ca043e7267ea36332f12aa51cb8c4000b36bce01de`

同じcanonical kernelへ次versionとしてpushし、773 PNGの実ファイル名、manifest順、zip順、
global metrics不変を確認する。別slugは作らない。

### OOF selector-confidence plot v3 push・監視停止

- 同じcanonical kernel `kentookumura/exp238-oof-selector-confidence-probe`へversion 3としてpush成功。id_noは`127444478`を維持し、別slugは作っていない。
- post-push pullでlocal packageとKaggle側16 cellsのsource一致、CPU、internet disabled、competition source 1、kernel sources 7、exp065 source追加を確認した。
- 実行variant 1、model config 0、fold学習0、booster 0、parent/control再学習なし、candidate再生成0、submissionなし。
- 最後に確認したstatusは`RUNNING`。CLI logsは実行中のため空だったが、既知挙動なので再pushしていない。
- 2026-07-18、ユーザー指示により監視を停止した。以後はstatus/logs/outputをpollせず、ユーザーから完了連絡を受けた後にv3 outputの順序とmetricsを監査する。

### OOF selector-confidence plot v3完了・共通typewell順監査

- ユーザーの完了連絡後に確認し、canonical kernel
  `kentookumura/exp238-oof-selector-confidence-probe` v3は`COMPLETE`だった。CPU / internet
  disabled、ログ最大時刻は`925.582679251`秒（約15分26秒）。benignな`tight_layout`と
  nbconvert warning以外の異常はない。
- baseは3,783,989行 / 773 wells、共通typewell対応はexp065
  `native_overlap` / threshold `0.999`の54 groups。773 / 773 plotsを生成し、filenameは全件
  uniqueでpatternに一致した。先頭は`typewell_0001_09441b8d.png`、末尾は
  `typewell_0054_f5859199.png`。
- Kaggle files APIを全4 pages監査し、outputは773 PNG + manifest / plots zip / distribution /
  summaryの計777 files。PNGのAPI表示順は辞書順であり、typewell順と一致した。
- manifest 773行の`plot_order`、typewell order、group内順、cluster ID/size/代表well、well、
  filenameをexp065対応表と全件照合して一致した。coverage欠損、重複、順序違反は0。
- plots zip全体は取得せず、signed outputのtailとcentral directory約121 KBだけをrange取得した。
  zipは343,358,561 bytes / 773 membersで、member順はmanifest順と完全一致した。
- 先頭・末尾PNGだけを取得し、いずれも2093×1485。先頭図を目視し、v2と同じ配色、灰色破線の
  selector top-1、候補色帯、凡例、反転TVD軸が正常であることを確認した。
- global metricsはv2から不変: exp238 LGB mean OOF `7.936690030870403`、selector top-1
  `8.512264240223669`、LikPF mean `11.594897672217703`、exp226 K16
  `9.42710967407494`、confidence mean / p50 / p90
  `0.3192525804042816 / 0.15628385543823242 / 0.7142080307006837`。
- 学習、candidate再生成、prediction変更、submission生成、competition submitは0。この実行は
  diagnostic PNGの並びだけを変え、exp238評価、ML route anchor、`experiment_summary.md`、
  `backlog/KAGGLE_DIRECTION.md`を変更しない。
- SHA: exp065 assignments
  `dcda8588cc1dd9261bafae7de00c890393e38b8a0ca0eb86fbba18a2cffc4a50`、manifest
  `c93ba17662279dfbcfb557599c86a64308a2ffb52da3f6e69595e24354176614`、distribution
  `639e2fb475bfb5c8859c03622c47b2b56fa0aa10b565ca17c31567e1aef01dae`、plots zip
  `56d2d578f85c127e2ff26b8780ba8f756251d9bf77a8fb2bcbc5f77be8509819`、summary
  `66981c2994b195395e338dda776d851fd64928de4c9ec8a745f28f0f3ca323e8`、kernel log
  `466942384a47358357a3cecaae409fc6541de1c341586f0a761c9020a52c335c`。
