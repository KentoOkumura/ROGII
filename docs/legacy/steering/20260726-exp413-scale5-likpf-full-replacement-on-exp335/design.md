# 設計

## アプローチ

exp404で同じ128 seed trajectoryから得た`scale 5・gs×1.0`のdirect RMSEは
`10.914522073423171`で、exp072互換`likpf_mean`の`11.59489788373621`より
約`0.680376 ft`良かった。この差をPF direct endpointの提出候補とはせず、
現行Public-LB bestのexp335全体へ伝播させたときのML価値を評価する。

13候補目のadd-onlyではなく、固定12候補bankの`likpf_mean` slotを同一IDのまま
`likpf_scale_5_x1p0`へ置換する。これによりcandidate width、one-hot width、
legal domain、formula weight、モデルfamilyを固定し、PF seed aggregationの変更だけを
原因変数にする。

全面置換はprimitive列だけの上書きではない。primitiveを入力とするformula、
bank disagreement、candidate-long、clean273、strict-nested compact74、
signed-residual compact23を依存順に再生成し、最後に370列LightGBMを再学習する。
旧mean primitiveはparity監査にだけ読み込めるが、selectorまたはdownstream入力には
残さない。

## 実験範囲

- 対象実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- Route: `ml_model`
- 親実験: `exp335_signed_residual_meta_on_exp264`
- 科学的根拠: `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- PF kernel親: `exp072_exp063_full_replay_feature_cache`
- candidate/compact親: corrected `exp264_exp263_candidate_confidence_dual_selector`
- 変更する変数: `likpf_mean` slotのseed aggregationをarithmetic meanから
  likelihood temperature 5の`likpf_scale_5_x1p0`へ変更する。
- 固定する変数:
  - `gs×1.0`、base scale estimator、clip `[10,60]`
  - 500 particles、128 seeds、PF dynamics、resampling、GR補完、float32出力
  - candidate count 12、ID、順序、family、2 legal domain
  - pair/fixed formula weight
  - exp264 outer5/inner4 split、sample key、88 selector schema、2 objectives
  - exp335 outer5/inner4 split、sample key、signed-residual objective
  - clean273 allowlist、compact74 schema、signed23 schema、final370 schema
  - LightGBM 3 configs、seed、target residual、runtime mode

## 全面置換契約

### 変更する5 candidate slot

1. `likpf_mean`
2. `exp226_k16__likpf_mean`
3. `selfgr_hmm_a070__likpf_mean`
4. `likpf_mean__exact_hmm`
5. `exp226_w500_50_50`

pair/fixed slotは既存weightで再計算する。`blend_likpf_hmm_w500`は引き続き
`likpf_mean__exact_hmm`のaliasであり、独立slotを作らない。

### 固定する7 candidate slot

1. `exp226_k16`
2. `selfgr_hmm_a070`
3. `exact_hmm`
4. `pf_ancc`
5. `beam_mean`
6. `exp226_k16__selfgr_hmm_a070`
7. `exp226_k16__exact_hmm`

### feature graph

- `clean273`: 列名と順序を固定する。allowlist上で`likpf_mean`を名前に含む22列だけを
  patchするのではなく、raw/candidate sourceから273列全体を再構築する。
- `selector88 -> compact74`: 新candidate値とbank値からcandidate-longを再構築し、
  outer5 × inner4、2 objectivesの40 selectorを再学習する。
- `signed23`: 同じ12候補に対する`true_tvt - candidate_tvt`をstrict nestedで
  outer5 × inner4、1 objectiveの20 selectorとして再学習する。
- `final370`: rebuilt clean273 + rebuilt compact74 + rebuilt signed23。
  1 variant × 3 configs × 5 foldsの15 GPU boostersを学習する。
- 全段でsemantic slot名は親schemaを維持するが、manifestに
  `value_source=likpf_scale_5_x1p0`を必須記録する。
- old `likpf_mean_x1p0`はparity監査専用。candidate/model inputへの混入は
  fail-closedとする。

## 実行段階

### Stage 0: 0-booster replacement preflight

- exp404 frozen predictionのrow/well/schema/decompressed/logical SHAを検証する。
- `likpf_scale_5_x1p0`をexp335/exp264 row identityへstrict joinする。
- candidate 12本、変更5本、固定7本、formula parity、feature lineageを検証する。
- 全replacement後のclean273/selector88 schemaと、旧meanがmodel inputへ
  残っていないことを検証する。
- PF、selector、downstream modelは実行しない。

### Stage C: corrected exp264 strict-nested selector replacement

- 1 variant × 2 objectives × outer5 × inner4 = 40 CPU boosters。
- control selector再学習0。
- score qualityとhard top1は診断として記録するが、downstream価値を代理して
  Stage Dを自動棄却しない。technical/leakage gateだけを次段の必要条件にする。

### Stage S: exp335 signed selector replacement

- 1 variant × 1 objective × outer5 × inner4 = 20 CPU boosters。
- 親signed selector再学習0。
- new Stage C top1 annotationと同じreplacement candidate bankを使用する。

### Stage D: downstream full replacement

- rebuilt 370 features、1 variant × 3 LightGBM configs × 5 folds = 15 GPU boosters。
- saved exp335 OOFをcontrolとし、親control再学習0。
- 合計学習量は60 CPU + 15 GPU = 75 boosters。

各段階のtrain-side実装候補は2026-07-26のユーザー依頼で承認・実装済み。
正規Notebook採用、Kaggle package/push/runは引き続き別承認を必要とする。
Stage DのGPU実行前には、1 variant / 3 configs / 5 folds / 15 GPU boosters /
control再学習0を再確認する。

Stage D version 2は15/15 modelsを完了し、saved exp335 RMSE
`8.146107755881022`から`7.884802794404715`へ`0.26130496147630744 ft`
改善、5/5 folds nonworse、全固定scope改善でprimary gateをPASSした。

### Current-test inference

- exp335 compact self-contained CPU inferenceを構成参照元とし、raw competition
  testから同じ12候補、21 native-confidence列、clean273を再生成する。
- exp073互換likelihood-PFは500 particles ×128 seedsをwellごとの
  `SHA256("likpf::test::<well>")`由来seedで一度だけ再生する。同じtrajectory bankに
  既存temperature-5集約が含まれるため、`likpf_mean` semantic slotへ
  `likpf_scale_5`を入れ、arithmetic meanはparity監査にだけ保持する。
- exp413 Stage C version 3の40 selector、Stage S version 1の20 signed selector、
  Stage D version 2の15 TVT modelをmanifest/model SHA検証後にCPU適用する。
  学習boosterは0。
- 各downstream outer foldへ対応するnested74 / signed23を作り、
  clean273と結合したfinal370を同foldの3 TVT modelsへ入力する。
- 予測監査ファイル、feature schema、missingness、model audit、
  reproducibility manifestを保存する。後続の明示承認を受けたKaggle version 3
  では、sample submissionのID/orderへstrict joinした`id,tvt`列を
  `/kaggle/working/submission.csv`にも保存する。
- version 2 output取得後にローカルで作ったCSVはschema/valueの事前検証だけに
  使用し、Kaggle Notebook outputやCode Competition提出物とは扱わない。
- Kaggle submit APIは呼ばず、competition submit authorizationはfalseを維持する。
- hidden code rerunではsample submissionとraw testからrow / ID / well集合を動的に
  決める。公開commit runの14,151 rows / 3 wellsは観測値として記録するだけで、
  runtime gateに使わない。PF生成直後のgateは、非empty、sampleと同じrow数・
  ID集合、ID一意、well数1以上とする。
- 既存正規inference placeholderは上書きせず、別名Jupytext
  `*_current_test_inference.py/.ipynb`と専用packageを使う。

## 評価と判定

primary comparisonはnew Stage D `lgb_mean`とsaved exp335 OOFの同一row比較。

- pooled RMSE改善: `parent - replacement >= 0.03 ft`
- fold: 5 folds中3以上でnonworse
- near `0--250`、mid `250--1000`、`1000+`、hidden-like spatial、
  hidden-like typewell-purged: 各`replacement - parent <= +0.02 ft`
- technical: row/well/fold、finite、candidate/formula、schema、SHA、model countを
  全PASS
- by-well p95、worst-well、`+1/+3/+5 ft`悪化well数は必須report-only

このgateはPublic-LB改善を狙うlate-stage ML replacement用であり、
train-side robust promotionとは分ける。primary gate PASS時だけ同じexp413内の
current-test実装候補へ進む資格を得る。2026-07-29に推論実装・実行は承認済み。
Kaggle Notebookでのsubmission生成は2026-07-29に承認済み、外部提出は引き続き
別承認とする。FAIL時はscale/multiplier/feature/candidate/weightの
same-OOF rescueを行わずbranchを閉じる。

## 再現性設計

- seed policy: train PFはexp404の`stable_sha256_per_well(split, family, well)`
  生成物を再利用する。selector/LightGBMは親と同じseed 42とfold/sample keyを使う。
- stochastic 処理の有無: train feature生成では新規PF乱数消費0。selectorとGPU
  LightGBMは学習を伴う。
- PF/Beam / likelihood-PF / seed baggingの有無: likelihood-PF scale5を使う。
  HMM/Beam/K16値は保存済み親値を固定し再実行しない。
- 並列処理と乱数の関係: current-testではwellごとのstable seedを先に決め、
  thread schedulingと乱数系列を分離する。global RNGをthread内で使わない。
- CPU/GPU runtimeとdeterministic flags: CPU selectorは
  `deterministic=true`, `force_col_wise=true`, `n_jobs=8`。GPU downstreamは
  `gpu_use_dp=true`, `deterministic=true`, `force_col_wise=true`, threads 8。
  GPU bitwise deterministic anchorとは呼ばない。
- train cache / test feature regenerationのSHA: exp404入力のraw/decompressed/
  logical SHA、replacement primitive、candidate bank、clean273、compact74、
  signed23、final370のschema/content SHAを段階別に記録する。
- model manifest / prediction / submission SHA: 40/20/15 model slotと各model SHA、
  OOF prediction SHA、current-test prediction content SHA、submission SHAを記録する。
- Kaggle package bootstrap: package作成時にbootstrap内configを展開し、
  experiment名、stage、run flags、parent sources、replacement source、model count、
  GPU/internet metadataを正規configと照合する。

## リスク

- リークリスク: candidate生成後にtruthをjoinし、outer-valid target/errorを
  selector fit、feature選択、threshold選択に使わない。exp404 predictionは
  truth/fold/hidden-like読込前にfreeze済みのものだけを使う。
- CV/LB不一致リスク: exp335はCV `8.146108` / LB `7.517`、exp372はより良い
  CV `8.071564`でもLB `7.587`だった。CV gateは必要条件であり、LB改善保証ではない。
- ランタイム/メモリリスク: 60 CPU + 15 GPU boosters。candidate-longは約
  45.4M rows、compactは3.78M rowsのためpartition streamingとmanifest SHAを使う。
- 再現性リスク: GPU LightGBMはbitwise固定を主張しない。current-test PFはrawから
  再生成されるためstable per-well seedとfeature content SHAが必須。
- semantic aliasリスク: 列名`likpf_mean`を維持しつつ内容がscale5になる。
  replacement manifestとstale-source guardがない実装は拒否する。
- scope拡大リスク: 旧meanとscale5のadd-only、13候補化、x1.3、scale grid、
  threshold/router、blendはこの実験に含めない。
