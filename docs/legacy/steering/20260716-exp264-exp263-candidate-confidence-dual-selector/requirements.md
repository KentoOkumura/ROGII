# 要件

> **監査結果:** 旧実装はraw-test-safe要件を満たしていない。training-only formation raw/delta
> 12特徴をouter-validで使用したため、旧Stage A/B/CとStage D add-only結果を無効化した。
> 修正版Stage A version 4は12特徴を削除し、actual train/current-test availability gateをPASSした。

## 依頼

`exp263_last_anchor_better_candidate_confidence_pair_cache`が生成する候補をselector入力の正とし、候補数を増やし、各候補の信頼度を加え、selector特徴量を整理する。selectorは候補別scoreを学習した後、漏洩なくdownstream TVT LightGBM用compact meta-featureへ変換できる設計にする。

## 確定した前提

- 候補集合のsource of truthはexp263であり、旧exp237/251の候補集合ではない。
- exp263 Stage 1でcurrent-test生成・parity確認済みのsurfaceは6 primitive、5 fixed 50/50 pair、固定`exp226_w500_50_50`の合計12本である。
- Stage 0 OOFにあるが現行Stage 1出力に未収録のsurfaceは、原理的な生成可否を否定せず、`stage0_oof_only_not_in_current_stage1_*`として別inventoryに記録する。
- exp263で除外した`sc_ens`、`hyb`、`tvt_dense`、`tvt_densew`、`tvt_dense50`は復活させない。
- `blend_likpf_hmm_w500`は`likpf_mean__exact_hmm`のaliasであり、別候補として重複登録しない。
- HMM+LGB exp221/234/240はscope外とする。raw exact/self-GR HMM candidateをLightGBM selectorでscoreすることはscope内である。
- exp251から継承するのはraw-test-safeなcandidate-long context、dual objective、nested stackingの方法であり、旧11候補や最終Viterbi予測ではない。
- exp238から継承するのは候補scoreをcompact meta-featureへ変換してTVTモデルへadd-onlyする考え方であり、35列という固定列数ではない。

## 機能要件

1. 12候補すべてに`pred_abs_error`と`p_within10`を出せること。
2. candidate IDは文字列としてartifactへ保持し、model入力では12 one-hotにすること。ordinal `candidate_index`は禁止する。
3. source-nativeなsigma等がない候補も、availability、shape、anchor距離、bank disagreement、formula parent disagreement等のtarget-free proxyを必ず持つこと。
4. formula候補は親confidenceをnamespacedで保持し、異familyのsigmaを単純平均しないこと。
5. exp251 v4の295列はfeature seedとして監査し、共有contextを残し、旧候補固有列を除去し、exp263由来featureへ再計算すること。
6. feature catalogに全特徴の説明、provenance、raw-test status、欠損率、重複・相関監査、objective×fold重要度を保存すること。
7. candidate-long scoreを同じfold/process内でcompact meta-featureへ変換し、推論時にscore CSVの保存・再読込を要求しないこと。
8. 12本を一つのhard-selectable domainにせず、exp263 lineage guardに従って11本primitive+pair domainと7本primitive+fixed domainを分けること。
9. selector-only OOFと、downstream用nested stackingを別stageにすること。
10. Stage A採用schemaが依存するcurrent-test confidence 21列をexp263 Stage 1の
    `confidence__<primitive_id>__<field>`から読み、欠損・非finite・invalidを推論前にfail-closedにすること。

## 制約

- Route: `ml_model`。PF/HMM/Beam候補はselector用meta featureの補助入力に限定し、direct blend、
  hard-path、Viterbi、softmax TVT平均を使わず、最終予測はdownstream LightGBMが生成する。
- feature監査は0 booster、selector-only初回は10 CPU boostersに固定する。
- nested selector 40 CPU boostersとmatched TVT 30 GPU boostersはstageを分け、Kaggle push前に別途ユーザー承認を得る。2026-07-17の旧Stage C/D承認runはfeature availability leakageで無効であり、修正版へ承認scopeを持ち越さない。
- 既存control、parent model、PF/HMM/Beam candidateを再学習・再生成しない。
- true TVT/error/oracle、exp263 catalog RMSE、pair readout、outer eligibilityはrow featureへ入れない。
- outer-valid wellのlabelをselector fit、calibration、threshold選択に使わない。
- Viterbi、hard-path提出、softmax TVT平均、HMMへのLGB score feedbackはscope外とする。
- 再現性は`docs/06_reproducibility.md`に従い、input/schema/content/model/prediction SHAとKaggle versionを記録する。

## 2026-07-18 推論例外の確定要件

- Stage Dの事前worst-well guard FAIL（+17.446742）は変更・緩和せず記録へ残す。
- ユーザーの明示指示により、例外scopeを保存済みmodelによるhidden-safe推論成果物生成だけに限定して進める。
- 学習は行わない。Stage Cの保存済み40 selector modelとStage Dの`selector_compact_addonly`保存済み15 TVT modelだけを使う。
- exp263 current-test Parquetを推論入力にせず、raw competition testから6 primitive、5 pair、fixed 1、21 confidence列を同じnotebook内で再生成する。
- 各downstream outer foldは対応するinner 4 model × 2 objectivesから74 compact列を生成し、対応するStage D 3 modelへ渡す。
- `submission.csv`は提出形式の推論成果物として生成してよいが、Kaggle competitionへのsubmit操作は行わない。
- hard selector、Viterbi、candidate TVTのsoftmax平均は引き続き禁止する。
- candidate-long selector入力はStage A feature catalogの欠損率を契約とする。学習時から疎な
  `conf__`/`formula__`のNaNはLightGBM missing semanticsとして保持し、0補完しない。`±inf`、
  学習時missing率0の列への新規NaN、構造的欠損率のずれ、current-test全欠損化はfail-closedにする。
- current-test ID/行順がsample submissionと一致し、候補値・必須native confidence・compact 74列・
  exp218 base 380列・最終454 model feature・最終予測がfinite、40/40 selector SHAと15/15 TVT model SHAが
  一致しなければfail-closedにする。raw selector 100列の契約内NaNはfinite要件の例外とする。

> 上記は旧100列・380+74列モデルに対する履歴であり、availability audit後は実行禁止とする。

## 2026-07-19 修正版推論・提出override

- ユーザーの明示指示により、修正版Stage C v6とStage D v3を使うhidden-safe推論、および
  submit-check PASS後のcompetition submit 1件を承認scopeとする。
- Stage D worst-well `+14.482873`のguard FAILは保持し、PASSへ変更しない。
- 学習は0 booster。Stage C v6の88特徴・40 selectorと、Stage D v3のclean 273 + compact 74 =
  347特徴add-only 15 TVT modelだけを使う。matched control 15 modelは推論しない。
- Stage C private Datasetはv6の40 model、88列schema、74列compact schema、manifestを含むbundleへ更新し、
  bundleと内部modelをSHA検証する。
- clean 273 allowlistのSHA、件数、一意性、モデル先頭273列との列順一致を推論前に検証する。
- notebook自身はsubmit APIを呼ばない。Kaggle outputの`submission.csv`をsample互換、ID順、重複、
  NaN/Inf、SHAまで検証し、PASSした場合だけ外部CLIで1件提出する。

## 受け入れ基準

- `candidate_contract.yaml`に12候補、alias、2 legal domain、confidence契約が列挙されている。
- `feature_contract.yaml`にfeature group、exp251 295列からのretain/recompute/add/remove規則、重複・相関監査が列挙されている。
- `output_contract.md`にcandidate-long dual score、compact変換、nested stacking、禁止事項が記載されている。
- config、README、SESSION_NOTES、result、metrics、notebook scaffoldが同じ候補数・booster数・stage gateを参照する。
- backlogと調査docsが7候補の旧案ではなくexp264の12 score surface設計を参照する。
- strict experiment validationとYAML/JSON/notebook構文検証が通る。
- candidate-long confidence/formula列は一括構築し、DataFrame fragmentation warningを回帰テストで防ぐ。
- deterministic anchorとする場合はfeature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel versionが記録されている。
- gzip生成物を比較する場合はraw `.csv.gz` SHAではなくdecompressed content SHAを主証拠として記録している。

## 2026-07-18 修正版要件

- selector raw horizontal allowlistはactual train/current-test共通の`MD/X/Y/Z/GR`だけにする。
- `ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA`と全派生をselectorから禁止する。
- Stage A/B/Cの各runは、全actual horizontal fileのheader availability gateをfit前に通す。
- 旧100列、旧score、旧compact、旧modelを修正版runへ混ぜない。修正版Stage Aでschema SHAを再発行する。
- exp218 380列matched controlは再利用しない。ユーザー選択により、非fold-safe 107列を削除した
  clean 273列をdownstreamの正式surfaceとし、380列のfold-safe再生成は行わない。
- clean 273 allowlistは`artifacts/feature_availability_audit/exp218_clean_273_allowlist.csv`、
  SHA256 `d01a73cc28485345dd86ed56ad6276f1727dca6b270d87685e1cf578afb677bf`で固定する。
- Stage Dはsource 380列・allowlist 273列・compact 74列を検証し、matched control 273列と
  selector compact add-only 347列以外の入力を拒否する。
- 修正版Stage Bは1 variant × 2 objectives × 5 folds = 10 CPU boosters、control/親再学習0本で
  2026-07-18に完走。score guardはPASS、hard top1 guardはFAIL。修正版Stage Cも40 CPU boostersで
  同日に完走し、score/leakage PASS、hard top1 FAIL。Stage Dも30 GPU boostersを承認しversion 3で完走済み。
- 修正版downstreamのvariant/config/fold/booster数とcontrol再学習理由はpush前に再承認済み。
  承認scopeはclean 273 control 15本 + clean 273/compact 74の347列add-only 15本だけとする。
- 修正版Stage A version 4の正は88列、logical schema SHA `aaef4ffdd...ddd3a4`。raw contextは
  train 773/773・current-test 3/3 fileで`MD/X/Y/Z/GR` availabilityを通過した。
- 修正版Stage B version 5のselector score OOFは意思決定に使用可能。candidate-long 45,407,868行、
  compact 3,783,989行×74特徴、10/10 model SHAを監査済み。downstream TVT CVとは扱わない。
- 修正版Stage C version 6のnested compactは後段add-only入力に使用可能。40/40 model SHA、25 partition
  manifest、18,919,945 compact rowsを監査済み。Stage D fit前に25 Parquet本体のbyte SHAを全件再検証する。
- 修正版Stage D version 3はclean 273 control 15 + 347列add-only 15の30/30 GPU boostersを完走した。
  pooled 10.476169 → 8.460811、5/5 folds、near/1000+/hidden-likeは改善したが、worst-well
  +14.482873で事前guard FAIL。corrected inference、hard selector、Viterbi、softmax TVT平均、
  competition submissionは禁止を維持する。

## 2026-07-19 OOF診断notebookとviewer CSV要件

- 指定されたexp238 selector-confidence probe `scriptVersionId=336248071`とLikPF 128-path probe
  `scriptVersionId=336196931`を構成参照元とし、同じexp264配下へ別名のJupytext sourceと正規notebookを追加する。
- 最終ML overlayとviewer出力は、feature availability leakageを除いたcorrected Stage D version 3の
  `selector_compact_addonly__lgb_mean__pred_tvt`だけを使う。旧Stage D version 2は入力候補に含めない。
- selector-confidence probeはcorrected Stage C version 6のstrict nested outer-valid candidate scoreを使う。
  selector結果はprimary `primitive_pair_bank`のpredicted-error top1とtop2-top1 marginだけを表示する。
  selectorの候補集合・top1結果はexp264へ更新してよいが、図の3段構成、主panelの比較パス種類、線色、
  線種、linewidth、alpha、HMM平均と±2sigma帯、margin panel、top1帯はexp238から変更しない。
  `primitive_fixed_bank`と`p_within10`はsummary監査には残してよいが、plotへ追加しない。
- Stage C outer-valid scoreは診断用、Stage D final OOFは最終予測用であり、Stage C top1をexp264最終予測や
  nested compactの採用候補値と誤記しない。hard selector guard FAILとStage D worst-well guard FAILを図とsummaryへ残す。
- LikPF probeはexp072と同じ500 particles、128 stable seedsをraw trainからwell単位で再生し、保存済み
  `likpf_mean_d`とのexact parityを確認する。true TVTは描画・RMSE以外へ使わない。
- viewer用CSVはKaggle形式の`id,tvt`、3,783,989 unique ID、NaN/Inf 0とし、corrected Stage D v3 OOF
  ParquetのSHA `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`を入力契約にする。
- 今回はローカルsource/notebook/CSVの作成と静的・schema検証までとする。Kaggle kernelの新規作成、push、
  773 well plot生成、competition submitは別の明示依頼があるまで実行しない。
