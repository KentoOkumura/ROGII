# 要件

## 依頼

exp389のfixed Huber `delta=1.345` exact-HMMは単体平均RMSEを改善したため、
selectorの候補bankへ追加して安全に利用できるか検証する。

## 制約

- Route: `ensemble`
- 親selectorはcorrected
  `exp264_exp263_candidate_confidence_dual_selector`に固定する。
- 既存fixed12 bankへ`huber_exact_hmm`だけを13本目として追加する。
  exp388のStudent-t候補は科学ゲートFAIL済みなので同時追加せず、fixed14にしない。
- Huber候補はprimary hard-select domainだけへ追加し、7候補fixed fallback domainは
  変更しない。
- exp389入力は`id,well_id,row_idx,huber_*_hmm_tvt,huber_*_hmm_std,
  huber_*_hmm_loglik`だけをallowlistする。truth、error、Gaussian/LikPF control、
  scope、gate診断はfit前に開かない。
- exp389はknown-prefixだけからwellごとに生成されたtarget-free候補でsource foldを
  持たない。全行を`well_id,row_idx`でglobal key joinし、exp263 selector outer foldへ
  repartitionする。source foldを合成・特徴利用しない。
- selector objective、outer/inner fold、sampling、LightGBM設定、raw-test-safe context、
  candidate weight、thresholdはexp388/exp264から変更しない。
- 実行量は1 variant / 2 objectives / outer 5 × inner 4 =
  40 CPU selector boosters。親control再学習、GPU booster、downstream TVT学習は0。
- inference、submission、current-test candidate生成は対象外。
- 再現性は`docs/06_reproducibility.md`に従い、入力decompressed SHA、
  feature schema/content SHA、40 model manifest SHA、score SHAを記録する。
- ユーザー指示「平均で改善しているのなら次に進んでください。selectorの候補に
  入れるのが次です。」をimplementationと上記Kaggle CPU Stage A/C実行の承認とする。

## 受け入れ基準

- technical:
  3,783,989 rows / 773 wells / 13 candidates / compact 77 features /
  40 models / 25 compact partitions / 18,919,945 compact rows /
  49,191,857 candidate-score rows、SHA、global-key join、leakageをすべてPASSする。
- selector score:
  `pred_abs_error`と`p_within10`がouter-train priorをpooledで改善し、
  各4/5 folds以上改善する。
- candidate integration:
  Huber top1率`>=0.5%`、利用fold`>=4/5`。
- scientific:
  fixed13 hard RMSEがparent fixed12 `8.652531955610227`以下、改善fold`>=4/5`。
  near / 1000+ / hidden-like 2面のdelta上限は各`+0.02 ft`、
  by-well p95 / worst上限は各`+0.25 ft`。
- fixed fallback errorはparentと`1e-6 ft`以内で一致する。
- H512 / whole-well add-one oracleはscore freeze後のdiagnostic-onlyで、
  学習・科学ゲートへ戻さない。
- FAIL時はweight、threshold、domain、同一OOF救済を行わずbranchを閉じる。
- PASSしてもdownstream TVT、inference、submissionへ自動移行しない。
