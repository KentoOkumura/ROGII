# 設計

## アプローチ

corrected exp264の12候補candidate-long dual selectorへ、exp486のabsolute
geometry unary predictionを13本目として追加する。候補を単独採用する実験でも、
likelihood-PFを再実行する実験でもない。保存済みtarget-free候補とmechanism
confidenceを、既存と同じnested selectorへ渡したときに安全に利用できるかだけを
paired OOFで検証する。

Residual版はpooled `-0.225289948 ft`、2/5 foldsであり、追加候補数の変更と
科学的問いを1つに保つため除外する。absolute + HMM 50:50はexp486でreport-only
改善したが、新規blend rescueになるため候補化しない。

## 実験範囲

- 対象実験:
  `exp496_exp486_absolute_geometry_fixed13_selector_on_exp264`
- Route: `ensemble`
- selector parent: `exp264_exp263_candidate_confidence_dual_selector`
- candidate parent: `exp486_exp226_geometry_residual_likelihood_pf`
- 構成参照: `exp392_exp389_fixed13_dual_selector_on_exp264`
- 変更する変数: primary hard-select候補を12本から13本へ増やし、
  `exp486_absolute_geometry_likpf`を末尾へ追加する。
- 固定する変数:
  既存12候補のID、順序、値、式、domain、exp264 outer folds、2 objectives、
  nested inner folds、LightGBM設定、sampling、fixed fallback 7候補、科学gate。

## 候補契約

### 追加候補

- ID: `exp486_absolute_geometry_likpf`
- source column: `likpf_scale5_absolute_geometry_unary`
- source: exp486 Stage 1 target-free freeze
- rows / wells: `3,783,989 / 773`
- generation:
  exp226 group-safe OOF `tvt_geop` + known-prefix GR / trajectoryだけを使う
  500 particles ×128 seeds、temperature 5のabsolute geometry likelihood-PF
- source fold:
  exp226 OOF safety auditにだけ存在する。selector featureには入れず、global
  `(well_id,row_idx)` join後にexp263 outer foldへpartitionする。

### Native confidence

absolute mechanism ledgerから次だけを許可する。

- `geometry_residual_mean`
- `geometry_residual_std`
- `geometry_log_factor_mean`
- `effective_sample_size`
- `resampled_seed_fraction`

candidate predictionとmechanism confidenceにはfinite / coverage / global-key parity
を要求する。`tvt_geop`、Residual prediction/ledger、truth、control、fold、role、
scope、gate、by-well結果は追加candidate featureとして使わない。

### Domain

- primary meta / hard-select domain:
  exp264の11候補 + `exp486_absolute_geometry_likpf` = 12候補
- fixed fallback comparison domain:
  exp264の7候補をbyte-for-byte維持
- score candidate inventory:
  12 + 1 = 13候補
- candidate ID encoding:
  宣言順one-hot 13、ordinal indexは特徴にしない

## Fold・leakage設計

1. exp486 predictionとabsolute ledgerのpayload SHA、schema、coverage、finiteを
   truthなしで検証する。
2. exp263 base rowsへglobal key joinし、exp263 selector outer foldを付与する。
3. candidate value、native confidence、既存raw-test-safe context、候補間
   disagreementからStage A schemaをfreezeする。
4. Stage A freeze前のtruth/error/oracle/control/gate read countが0であることを
   ledgerに残す。
5. outer-trainのcompact値はinner OOF scoreだけ、outer-validはouter-train内
   4 inner modelsのensemble scoreだけから作る。
6. 全40 model scoreとhard choiceをfreezeした後だけtruth、scope、hidden-like、
   parent fixed12 scoreをattachする。
7. parent比較、candidate usage、incumbent reranking、by-well tail、非gating oracleを
   truth-lateで出力する。

## Model・実行量

- active variant: 1 (`absolute_geometry_fixed13_addone`)
- objectives:
  - candidate absolute errorのL1回帰
  - `abs_error <= 10 ft`のbinary分類
- outer / inner folds: `5 / 4`
- planned CPU selector boosters: `1 × 2 × 5 × 4 = 40`
- parent/control retraining: 0
- PF / HMM / Beam再実行: 0 / 0 / 0
- GPU / downstream TVT boosters: 0 / 0
- inference / submission: 0 / 0

Stage Aの最終feature数とcompact列数は、事前固定した機械的drop
（all-missing、constant、exact duplicate）後にtruthを見る前にSHA freezeする。
exp392の77列を結果に合わせて強制しない。

## Gate

### Technical / leakage

- 3,783,989 rows / 773 wells / 13 candidates
- exp486 prediction・absolute ledger SHA一致
- candidate / mechanism finite coverage 100%
- exp486とexp263のglobal key parity
- upstream exp226 OOF geometry contractとexp486 scientific contract SHA一致
- source foldをmodel featureに使わない
- feature freeze前truth/error/oracle/control/gate read 0
- 40/40 models、25 compact partitions、18,919,945 compact rows、
  49,191,857 outer-valid candidate-score rows
- fixed fallback hard prediction / error parity max abs `0.0 ft`

### Selector score

expected-error MAE、within10 logloss、within10 Brierの全3指標について、
outer-train priorよりpooled改善し、各指標4/5 folds以上改善する。

### Scientific AND gate

- exp486 absolute top1 fraction `>= 0.5%`
- positive usage folds `>= 4/5`
- fixed13 hard pooled RMSEがparent fixed12以下
- parent fixed12より改善fold `>= 4/5`
- raw GR observed / missing、高missing、near 0--250、1000+、hidden-like spatial、
  hidden-like typewell-purgedのdelta RMSEが各`<= +0.02 ft`
- by-well delta RMSE p95 `<= +0.25 ft`
- worst-well delta RMSE `<= +0.25 ft`

全項目AND。1つでもFAILなら
`FAIL_CLOSE_EXP486_ABSOLUTE_FIXED13_SELECTOR`で閉じる。利用率やpooled改善だけで
救済しない。

### 診断専用

- H512 / whole-well add-one oracle headroom
- exp486利用率とwell deltaのPearson / Spearman
- exp486非top1行におけるincumbent choice変更率
- exp486利用0 wellの改善 / 悪化数
- score margin / entropy別のincumbent reranking

これらは全score/gate freeze後に計算し、科学gateや閾値選択には使わない。

## 再現性設計

- seed policy:
  LightGBM seed 42 + fold/objective/candidateを含むstable SHA256 sampling key
- stochastic処理:
  LightGBM row/column sampling、candidate-long row sampling
- PF/Beam:
  exp486保存predictionを再利用し、新規PF/Beam乱数は0
- 並列処理:
  parallel fit前にstable-key samplingを完了し、worker内global RNGを使わない
- runtime:
  将来実装時はKaggle private CPU、GPU/internet off、8 threads、deterministic / 
  force_col_wiseを固定
- input:
  exp486 gzip raw/decompressed、prediction logical、absolute ledger payload、
  exp264 parent score、exp263 cache/catalog、hidden-like assignment SHAをhard check
- output:
  input contract、feature catalog/schema/content、fold manifest、40-model manifest、
  candidate score、compact manifest、paired metrics、gate、summaryのSHAを記録
- deterministic anchor:
  deterministic flagsとSHAは記録するが、submission未生成かつstochastic selectorの
  ためdeterministic submission anchorとは呼ばない
- Kaggle bootstrap:
  loose / package / bootstrap config、contracts、metadata、Dataset / kernel sourcesを
  push前にbyte / SHA照合する

## リスク

- leakage:
  exp486 truth-late rows、Residual、control、role/fold、gate結果をselector featureへ
  混ぜる危険。予測・absolute ledgerのallowlistとread ledgerで防ぐ。
- fold:
  exp486はexp226 OOF geometryを使う。upstream group-safe contractは監査するが、
  upstream fold ID自体はselector特徴にしない。
- tail:
  absolute候補単体のp95 / worst悪化が非常に大きい。selector score改善だけでは
  安全性を示さないためfixed tail AND gateを維持する。
- reranking:
  exp392などfixed13実験では、追加候補を選ばないwellでも既存候補順位が変わり
  tailが悪化した。parent saved scoreとのpaired監査とpost-freeze診断を必須にする。
- CV/LB:
  train-side selector PASSでもraw-test exp486候補の生成可能性やLB改善を保証しない。
  inference設計・実装・提出は別承認とする。
- runtime/memory:
  49,191,857 candidate-long score rowsと40 CPU modelsを扱う。exp392と同じ
  base-row cap / chunkingを固定し、親controlを再学習しない。
- multiple testing:
  Residual追加、pair/blend、threshold、weight、domain、feature family、gateを同じ
  OOFで調整しない。

## 次のアクション

現在はdesign frozenで停止する。ユーザーが別途実装を承認した場合だけ、
Jupytext compact self-contained train候補、fail-closed inference placeholder、
candidate / feature contract validation、専用testを実装する。実装後もKaggle
Stage A/C runはさらに別承認とし、40 CPU boostersとcontrol再学習0を再確認する。
