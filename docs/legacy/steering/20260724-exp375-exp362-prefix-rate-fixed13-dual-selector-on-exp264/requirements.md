# 要件

## 依頼

exp362の保存済みOOFで実際に観測された`prefix_rate_exact_hmm`を、
corrected exp264のfixed12 candidate bankへ13本目として追加し、同じ
dual-objective selectorを直接再学習する実験を設計する。

別のadd-one novelty監査は設けず、候補価値の上限診断はselector実験の評価出力へ
参考値として同梱する。今回はsteering、実験ディレクトリ、バックログ、実験サマリーの
設計記録だけを作成し、実装、validation、Kaggle実行は行わない。

## 追加依頼

2026-07-24のユーザー指示「exp375を実装してください」により、上記design-only
停止条件をimplementation-onlyまで解除する。正規notebook採用、Kaggle package、
push、run、downstream TVT、inference、submissionは引き続き対象外とする。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 追加候補はexp362の保存済み`candidate_tvt` 1本だけとし、候補IDは
  `prefix_rate_exact_hmm`とする。意図していたlocal donor slope、K16 donor field、
  donor support、`mu_rate`、fallback列は候補の意味やselector featureへ持ち込まない。
- exp362の観測結果は全12,368 segmentで`mu_rate == prefix_rate`だったため、
  donor-slope仮説の成功証拠として扱わない。
- fixed12の候補値・順序、既存`exact_hmm`とその派生formula、fixed fallback 7本、
  exp264 selector fold、2目的、sampling、LightGBM設定は変更しない。
- `prefix_rate_exact_hmm`は既存`exact_hmm`を置換しない。既存formulaを新候補から
  再計算せず、primary domainだけを11候補から12候補へ増やす。
- exp362 OOFのsource foldはprovenanceとして保持し、selector featureには使わない。
  `well_id,row_idx`でglobal key join後、親と同じexp263 selector foldへrepartitionする。
- target-free allowlistは`well_id`、`row_idx`、source fold、`candidate_tvt`、
  `candidate_std`、`hmm_loglik`に限定する。`candidate_std`は`sigma_tvt`、
  `hmm_loglik`は`source_loglik`と`loglik_per_row`へ標準化してnative confidenceに使う。
- feature freeze前にtruth、error、oracle、exp362 post-run評価列を読み込まない。
- Stage A feature auditとnested Stage C selectorを同じrunで行う。別Stageの
  novelty監査、候補生成HMMの再実行、親fixed12 selectorの再学習は行わない。
- 実行予定量は1 variant、2 objectives、outer 5 × inner 4、
  合計40 CPU selector boosters。GPU booster、downstream TVT、inference、
  submissionは0とする。
- 実装とKaggle runには別のユーザー指示を必要とする。

## 受け入れ基準

- 設計時点:
  - `docs/legacy/steering/20260724-exp375-exp362-prefix-rate-fixed13-dual-selector-on-exp264/`
    にrequirements、design、tasklistが揃う。
  - `experiments/exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264/`
    にdesign-onlyのconfig、README、SESSION_NOTES、result、metricsが揃う。
  - `KAGGLE_DIRECTION.md`と`experiment_summary.md`に未実装実験として登録される。
  - notebook、helper、candidate/feature contractの実装変更は行わない。
- 将来のtechnical gate:
  - 3,783,989行 / 773 wells / 13候補のkey、finite、fold、SHA契約が成立する。
  - exp362 prediction logical SHA
    `bdf616e00bdebb496093d3d05526aebce01381281c4b1c46f7b77e72e57415cb`
    とdecompressed SHA
    `e1d672ff9743b92c33a40bec8d4cf3b0a8c29cdbbb37948992f0809522e3e7ef`
    を照合する。
  - outer-valid wellsをinner fit/early stoppingから除外し、40 modelと
    25 compact partitionsを生成する。
  - feature freeze前のtruth/error/oracle accessが0である。
- 将来のscientific gate:
  - 2目的のselector scoreがouter-train candidate priorをpooledと4/5 folds以上で改善する。
  - `prefix_rate_exact_hmm`のprimary top1利用率がpooled 0.5%以上かつ4/5 folds以上で正になる。
  - fixed13 hard selectorのpooled RMSEが保存済みfixed12
    `8.652531955610227`を悪化させず、4/5 folds以上で改善する。
  - near 0--250 ft、1000+、hidden-like 2面の親差を各`+0.02 ft`以内、
    by-well p95差とworst-well差を各`+0.25 ft`以内とする。
  - pooledが改善しても安全性gateを満たさない場合は、平均改善の診断値だけを残し、
    downstream TVT、inference、submissionへ自動移行しない。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
