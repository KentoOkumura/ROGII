# 検証方針

実験テンプレートが読む fold 数、seed、metric、共通 validation 方針は `project.yml` を正とします。このファイルは設計意図とリークチェックの運用メモです。

## CV 設計

- Fold 作成方法: `GroupKFold`
- グループキー: ファイル名から抽出する `well_id`
- 層化キー: 初期はなし。必要なら well 単位の target 範囲、evaluation zone 長、GR 欠損率で層化候補を作る。
- ランダムシード: 42
- Fold 数: 5
- メトリック: RMSE。小さいほど良い。
- 評価対象行: validation well のうち `TVT_input` が NaN の行だけ。
- 主な検証コマンド:
  ```bash
  task validate-exp EXP=exp001_baseline
  task prepare-kaggle-notebooks EXP=exp001_baseline EXTRA_ARGS="--strict"
  task push-kaggle-train EXP=exp001_baseline
  task kaggle-status KERNEL=<username>/<train-kernel-slug>
  task kaggle-logs KERNEL=<username>/<train-kernel-slug>
  ```

  `task kaggle-logs` は Kaggle CLI 2.2.3 の `kaggle kernels logs -f owner/slug` を実行し、Kaggle UI と同系統の live SSE から stdout/stderr を逐次取得する。`--interval` は使わない。

## リークチェックリスト

- [ ] 同一 `well_id` のデータが train/valid にまたがっていないか。
- [ ] `TVT_input` の NaN より後ろの真値 `TVT` を特徴量や補間に使っていないか。
- [ ] train-only formation columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) を推論前提の特徴量にしていないか。
- [ ] typewell や horizontal well の統計量を valid/test 全体で fit していないか。
- [ ] test 由来の統計量を train に使っていないか。
- [ ] target encoding が fold 外データを参照していないか。
- [ ] augmentation や前処理が validation に不適切に影響していないか。
- [ ] 学習時と推論時の前処理が一致しているか。
- [ ] CV と LB の乖離を `result.md` と `experiment_summary.md` に記録しているか。

## CV/LB 乖離ログ

| 実験 | CV | Public LB | 乖離 | メモ |
| --- | --- | --- | --- | --- |
| exp001_baseline | - | - | - | planned。well holdout RMSE を実装予定。 |

## 検証判断

- 採用候補: 5-fold の平均 RMSE と fold 間ばらつきを必ず記録する。1 fold だけの改善は provisional とする。
- Public LB は最終判断の参考にするが、CV で説明できない改善は過学習/分布合わせの疑いとして扱う。
- CV を切り直す場合は、過去実験との比較が崩れるため `experiment_summary.md` に理由を残す。
- 提出前には Kaggle output を取得した場合だけ `task submit-check EXP=<exp> SUBMISSION=<downloaded-submission.csv>` で sample submission と行数、列、`id` 順、NaN を確認する。
