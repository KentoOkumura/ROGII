# 要件

## 依頼

exp333 の保存済み Stage 1 OOF を exp264 の fixed12 candidate bank へ 1 本だけ追加し、
fixed13 candidate-long dual selector として再学習する。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親 selector は corrected `exp264` Stage C v6 とする。
- candidate 値は exp263 deployable12 と exp333 Stage 1 OOF を保存済み SHA で固定する。
- 変更は `exp333_segment_offset` 1 candidate の追加だけとする。selector の目的、
  LightGBM 設定、outer/inner split、sample 上限、early stopping、raw-test-safe context は
  exp264 corrected Stage C v6 から変えない。
- exp333 は candidate-long の13本目として score し、primary hard-select domain には追加する。
  fixed fallback domain は exp264 の7本を変更しない。
- exp333 saved-exp226 source foldはOOF provenanceとして保持するがselector特徴には使わない。
  exp333は全行を`well_id,row_idx`でglobal key joinし、親exp264と同じexp263 selector
  outer foldへ再partitionする。source foldとselector foldの一致は要求しない。
- outer-valid truth は inner assignment、fit、early stopping、feature freeze に使わない。
- fixed12 selector / exp264 Stage C は保存済み結果を比較基準とし、再学習しない。
- Stage A feature audit と Stage C nested selector だけを対象とし、downstream TVT、
  current-test inference、submission はこの実行に含めない。
- Kaggle run 前に 1 variant / 2 objectives / outer 5 × inner 4 /
  40 CPU boosters / control retraining 0 を明示し、実行承認を記録する。

## 受け入れ基準

- exp263 manifest/catalog、exp333 OOF file/decompressed SHA、13-candidate contract、
  global row-key identity、source/selector fold overlap、feature schema、compact schemaを
  fit 前に照合できる。
- `3,783,989 rows / 773 wells / 5 selector folds / 5 source folds / 13 candidates`が
  完全に揃い、exp333 predictionがfinite、key重複・欠損なし、5×5 fold overlapを保存する。
- Stage A は train/current-test 共通 raw context allowlist を保ち、target/error/oracle 列を
  selector feature に含めない。
- Stage C は 40/40 model、25 compact partition、`3,783,989 × 5` compact rows、
  `3,783,989 × 13` outer-valid candidate-long rowsを生成する。
- dual score は outer-train candidate prior より pooled と 4/5 folds 以上で改善する。
- exp333 primary top1 使用率を pooled / fold で報告し、少なくとも pooled 0.5%、
  4/5 folds で 0% 超とする。
- 13候補 hard-primary OOF は、保存済み exp264 fixed12 hard-primary OOF
  `8.652531955610227` を pooled で悪化させず、4/5 folds 以上で改善する。
- fixed fallback `exp226_w500_50_50`、near 0--250、1000+、hidden-like 2面、
  by-well p95 / worst を報告する。安全gateは保存済みparent fixed12 selector比で、
  by-well p95 / worst悪化を `+0.25 ft` 以下とする。fixed fallbackは値不変parityと
  絶対参考値として別に報告する。
- 上記 scientific gate を通過した場合だけ downstream TVT 15 GPU booster
  （保存済み exp264 OOF control、control retraining 0）を別承認候補にする。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 2026-07-24 Stage D 例外進行

- fixed13 selectorはpooledでparent fixed12より`-0.232534584 ft`改善したが、
  by-well p95 / worst安全gateは不合格だった。元のscientific gateはFAILのまま保持する。
- ユーザーの「平均で改善しているのなら次に進みましょう」を、既設計のStage Dに限る
  明示的な実行許可として扱う。安全gateのPASSへの再分類や閾値緩和とは扱わない。
- Stage Dはfixed13 compact 77列をclean273へadd-onlyする1 variantだけを学習する。
  3 LightGBM configs × 5 folds = 15 T4 GPU boosters、保存済みexp264 Stage D v3を
  比較対象とし、parent/control再学習は0とする。
- Stage D独自の判定は保存済みparent12 compact add-only比で、pooled改善、3/5 folds、
  near / 1000+ / hidden-like、by-well p95 / worstを事前固定したAND gateとする。
- Stage Dにはcandidate/selector/PF/HMM再生成、inference、submissionを含めない。
