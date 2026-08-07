# exp231_same_typewell_horizontal_gr_atlas_gated_hmm_emission セッションノート

## 目的

`same_typewell_horizontal_gr_atlas_gated_hmm_emission` を exp209 exact HMM の正式な train-side 実験として実装する。これはraw-test提出候補ではなく、fold-safe atlas emission が成立するかを判定する第1段階である。

## 現在の状態

- 状態: train-side完了・不採用（inference / submitなし）
- Route: `pf_beam`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 参照: exp065 typewell groups、exp201 readout、exp115 hidden-like、exp223/225/230
- 推論 / 提出: なし

## 実装内容

- exp065 `native_overlap / threshold=1` group assignmentを入力にする。
- seed 42の5 well foldsで、validation wellと同fold validation wellsをatlas sourceから除外する。
- training-fold peerのtrue TVTで `(group, 2ft TVT bin)` を索引し、`64/128/256` rowsのrobust normalized GR patch distributionを構築する。
- query/state scoreはstate方向にcenter/clipする。`alpha=0.01/0.025/0.05` の3 HMM variantsで、target-free confidenceのみをgateとして加算する。
- `peer_atlas_confidence`、support、novelty、uniqueness、base ambiguity、innovation、change pointをcacheに保存する。
- direct comparisonでcandidate true-state rankと、saved exp072 likPF errorを診断用labelにしたpersistent-offset onset AUC/q90 liftを追加した。

## GPU / 実行コストガード

- Kaggle runtime: CPU、internet off、`outer_workers=2`、`numba_num_threads=2`。
- active variants: `hmm_peer_atlas_a010` / `a025` / `a050` の3本。
- LightGBM config数: 0、fold数: well 5 folds、合計booster数: 0。
- exp072 / exp209 controlの再学習・再生成: なし。saved exp072 cacheとexp209記録を比較基準として使う。
- 実行時間は3 HMM variantsとfold atlas構築により重い。Kaggle train push前にgenerated package/metadataを検証し、結果をlogs中心に記録する。

## 再現性

- atlas construction、patch stride、TVT binning、HMMは乱数なし。fold assignmentだけをlocal RNG seed 42で固定する。
- well HMMをthread並列してもatlasは先に固定構築されるため、thread schedulingはsource membershipを変えない。
- cluster assignment SHA、atlas fold summary SHA、schema SHA、raw gzip SHA、decompressed feature content SHAを保存する。
- deterministic submission anchorではない。raw-test atlasのsource policyとfeature parityは未監査。

## 実施した検証

- `py_compile`（atlas/HMM、cache、comparison、joint generation、train/inference notebook source）: pass。
- `ruff check --select F821`: pass。
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb` と `--test`（train/inference）: pass。
- `make validate-exp EXP=exp231_same_typewell_horizontal_gr_atlas_gated_hmm_emission`: pass。
- `make validate-template`: pass。
- 親 exp209 / exp230 には compact self-contained notebook source がないため、通常のJupytext sourceを章立ての参照元にした。
- ローカル notebook 実行は行わない。初回実行はKaggle Notebookとする。

## Kaggle train 準備

- canonical kernel id: `kentookumura/exp231-same-typewell-horizontal-gr-atlas-gated-hmm-emission-train`
- title: `exp231 same typewell horizontal gr atlas gated hmm emission train`
- `prepare-kaggle-notebooks --strict`: pass。generated metadataは `enable_gpu=false`、`enable_internet=false`、competition sourceは ROGII、kernel sourceは exp072 / exp065 / exp115 の3件。
- generated support file の `feature_cache.py`、`exact_hmm_smoother.py`、`direct_hmm_comparison.py`、`joint_cache_generation.py` の構文チェック: pass。

## Kaggle train push 記録

- 2026-07-11: 長い canonical id `kentookumura/exp231-same-typewell-horizontal-gr-atlas-gated-hmm-emission-train` を push したが、Kaggle `SaveKernel` が詳細なしの HTTP 400 を返した。
- 同IDを `kaggle kernels pull -m` で確認し、HTTP 403（存在しない private kernel）だった。別versionの存在やslug衝突ではない。
- Kaggleのkernel slug長制約を回避するため、意味を保つ短縮canonical id/title `kentookumura/exp231-same-typewell-gr-atlas-hmm-train` / `exp231 same typewell gr atlas hmm train` に再生成して同一exp231内で再pushする。モデル、fold、variant、入力は変えない。
- 短縮canonical idで Kaggle train v1 のpushに成功。URL: `https://www.kaggle.com/code/kentookumura/exp231-same-typewell-gr-atlas-hmm-train`。CPU / internet off / run-on-push。結果は同IDのlogsで確認する。
- push直後の `kaggle kernels status` は `KernelWorkerStatus.RUNNING`。CLI logsは実行中は空であり、空ログだけでは失敗と判定しない。
- Kaggle train v1 は timeout で未完了。最終ログは 43,206.108 sec で `[589/773]` を開始した時点で途切れた。atlas構築完了からの進捗は2 well並列で約147 sec/well相当のため、全773 wells・3 variantsの予測wall timeは約15.8hとなりruntime上限を超える。
- 開始済み588 wellsの `peer_atlas_available_rate` は約0.96-0.98だが、575 wellsの `peer_atlas_confidence_mean` は `0.0`、残る13 wellsも `0.000046–0.014480` に留まった。先頭を含む大多数のwellで `hmm_peer_atlas_a010/a025/a050` のRMSE、std、loglikは同一で、alpha gridはほぼbase HMMの重複再実行になった。
- v1のtimeoutはOOMやinput path errorではない。v2前にmatch-confidenceをrank/relative scoreへ再scaleしてnonzero gateを確認し、full runのactive alphaを1本に制限する必要がある。この変更は実験実行計画に影響するため、ユーザー承認待ち。

## v2 gate-rescale preflight 実装

- ユーザー承認により、同一 exp231 内に v2 preflight を実装した。新規expは作らない。
- v1の主要原因は、2ft TVT bin に0.35ft HMM statesが複数入るため、同じbinの重複stateをtop1/top2として比較し、uniqueness marginが0になったこと。v2はunique TVT bin単位でscore順位・marginを計算し、そのscoreをstateへ展開する。
- absolute patch distanceの指数減衰はunderflowしやすかったため、candidate-bin内のmedian/IQRに対するrelative fitと、soft absolute-novelty guardの積へ変更した。true TVT/error/LBは使わない。
- preflightはalpha `0.025` の1 variant、12 fixed target wellsだけでHMMを実行する。一方、5foldのatlasは全773 wellsから従来どおり構築するため、source policyはfull runと同一。
- preflightではdirect comparison / CV採否 / inference / submitを行わず、gate各成分、nonzero rate、variant差分、runtimeだけを確認する。
- v2変更後の `py_compile`、F821 lint、Jupytext train conversion/test、strict `validate-exp`: pass。

## Kaggle v2 preflight 準備

- canonical kernel id/title は v1 と同一の `kentookumura/exp231-same-typewell-gr-atlas-hmm-train` / `exp231 same typewell gr atlas hmm train` を使う。v1のkernel metadata pullに成功し、別slugを作らずv2としてpushする。
- `prepare-kaggle-notebooks --strict` は pass。generated configは alpha `0.025` 1 variant、12 fixed target wells、`run_direct_comparison=false`、CPU / internet off、full 5fold atlas を確認した。
- 実行予定: LightGBM config 0、booster 0、親/control再生成なし。目的はgate component / nonzero rate / runtimeの確認だけであり、CV/LB判断はしない。
- Kaggle train v2 preflight push: success。URLは v1 と同一。run-on-pushで実行開始済み。

## Kaggle v2 preflight 結果

- Kaggle kernel `kentookumura/exp231-same-typewell-gr-atlas-hmm-train` version 2 は完了。実行はCPU / internet offのまま、LightGBM config 0、booster 0、親/control再生成なしである。
- atlas sourceは5 fold・773 wellsを維持し、HMM対象のみ固定12 wellsにした。12/12 wellsが `status=ok`、skip 0、出力55,801 rowsとなった。
- total wall timeは485.437秒、HMM generationは296.884秒、成功well当たりHMM elapsed平均は48.293083秒だった。v1の3 variants全well見積もりとは異なり、alpha 1本の全773 wellは約5.3 CPU時間と見積もられ、Kaggle runtime枠内に収まる見込みである。
- `peer_atlas_confidence_mean` は全12 wellで非zero。well平均は `0.022708–0.143981`、対象全体の平均は `0.077299`。match confidenceは `0.294159–0.646224`、uniquenessは `0.058919–0.227758` で、relative-fitとunique-bin marginが実際にgateを駆動した。
- raw gzip feature SHA256: `9819bcbd37f49dac063d1109eb31a2ff83748a588987a923d0cf993ab1958627`。
- direct comparisonは無効のため、ログ中の12-well mean HMM RMSE `6.863589540975078` は選択subsetの記述値に過ぎず、CV、baseline比、採否判断として使用しない。
- 結論: v2 preflightは「gateが実際に発火し、alpha 1本のfull runが時間内に成立する」ことを確認した。本実験のCV/readoutはまだ未実行である。

## Kaggle v3 正式full run

- ユーザー承認により、同一 exp231・同一kernelで正式full runへ進む。v3では `preflight_target_wells=null` とし、全773 target wellsを評価する。
- active variantは新規 `hmm_peer_atlas_a025` 1本、LightGBM config 0、well 5 folds、booster 0。CPU / internet off、`outer_workers=2`、`numba_num_threads=2`を維持する。
- `run_exp072_full_cache=false` のため親/controlを再生成しない。Kaggle入力のsaved exp072 full replay cacheを読み、`run_direct_comparison=true` でglobal、distance bucket、hidden-like、worst-well、true-state rank、persistent-offset-onsetのreadoutを出力する。
- preflight実測から全体のHMM部分は約5.3時間、comparisonを含めてもKaggle runtime枠内の見込み。raw-test generation、inference、submissionは引き続き無効。
- 2026-07-11: strict packageを生成し、generated configで `preflight_target_wells=null`、`hmm_peer_atlas_a025` 1本、`run_direct_comparison=true`、CPU / internet offを再確認した。静的`validate-exp`、`py_compile`、F821 lintもpass。
- 同一kernelの既存v2 metadata（id/title、3 kernel sources、CPU / internet off）をpullして確認後、Kaggle train v3 pushに成功。URL: `https://www.kaggle.com/code/kentookumura/exp231-same-typewell-gr-atlas-hmm-train`。run-on-pushで正式full runを開始した。

## Kaggle v3 正式full run 結果

- Kaggle kernel version 3 はCPU / internet offで正常完走。`hmm_peer_atlas_a025` 1 variant、LightGBM config 0、booster 0、5 well folds、773 target wells、3,783,989 rowsで、全773 wellsが `status=ok`、skip 0だった。HMM generationは17,435.185秒、全体は17,817.683秒。
- fold atlasは各foldで `validation_in_source_count=0`。cluster assignmentはnative-overlap threshold 1、41 groups / 760 wells、SHA256 `dcda8588cc1dd9261bafae7de00c890393e38b8a0ca0eb86fbba18a2cffc4a50`。decompressed HMM feature SHA256は `4d15cd4d115d6ad138a199110f7745fb28fc4e8defb74e3ad1e3f0a46691a91b`。
- gateはfull runでも有効で、mean `peer_atlas_confidence=0.086781`。v1のgate=0問題とruntime問題は解消した。
- saved exp072 `likpf_mean` 比のglobalはRMSE `11.594897668 → 11.569950236`（`-0.024947433`）、MAE `-0.460251178`、within10 `+0.014836988`。ただし`1000_plus`はRMSE `12.702990212 → 12.719560239`（`+0.016570028`）と悪化した。
- by-wellは457改善 / 316悪化、最大悪化は `b19b0395` の `+48.316177856 RMSE`。persistent-offset onsetはAUC `0.507653828`、q90 lift `1.111110817`で、gateをoffset recovery detectorとして支持しない。hidden-like outputは空で未評価のため、そのguardも満たせない。
- 判定: globalの小改善ではworst-well / longtailリスクを相殺できない。exp209保存済みbest blend RMSE `10.269696147`も更新しないため、train-side不採用。alpha再grid、raw-test port、inference、submitは行わない。

## 次のアクション

1. same-typewell GR atlasのemission直接加算は追加tuningせずclosedとする。
2. 次のPF/Beam候補は、peer TVTをemissionへ入れない既存backlogからユーザーと選ぶ。
