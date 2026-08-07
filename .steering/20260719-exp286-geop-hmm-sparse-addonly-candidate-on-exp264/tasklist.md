# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。Stage 0の固定gate branchだけを閉じ、ユーザー明示指示によりfull-all-well Stage Bを再開した。

## 完了

- backlog契約、親exp264/exp279、再現性文書を確認した。
- `.steering/20260719-exp286-geop-hmm-sparse-addonly-candidate-on-exp264/`を作成した。
- Stage 0のgate feature、単調方向、oracle、guard、runtime manifest契約を固定した。
- compact self-contained train/inference、専用tests、正規notebookを実装する。
- 静的/Jupytext/strict experiment validationを実行する。
- `--no-src`のprivate CPU packageをprepareし、canonical metadataとbootstrap SHAを確認した（pushなし）。
- `KAGGLE_DIRECTION.md`から未着手backlog行を削除し、実装済み・Kaggle train待ちへ移した。
- 1 variant / 0 config / 0 trained fold / 0 boosterを再提示し、Stage 0 CPU push承認を得た。
- canonical private CPU kernel version 1（id_no `127856113`）をpushし、`COMPLETE`まで監視した。
- outputを取得し、input / truth-free gate / oracle / unique-best / by-well / summary SHAを全件照合した。
- full unionは3粒度・5/5 foldsで改善したが、固定gateのwhole-well SSE gain保持率が`27.71% < 50%`でguard FAIL。
- 固定gateの救済gridは行わず、sparse gate分岐だけを閉じた。
- ユーザーの明示指示により、full-all-well `geop_hmm`を他候補と同じcandidate-long情報付きで
  selectorへ追加するStage B（1 variant / 2 objectives / 5 folds / 10 CPU boosters）を承認済みとした。
- candidate contract、chunked exp279 loader、exp263 outer-fold alignment、Stage A/B、parent12比較を実装した。
- Jupytext、構文、ruff、対象30 tests、strict validation、Kaggle package/cost監査をPASSした。
- version 2 path error、version 3 source-fold errorはいずれもmodel 0のtechnical failureとして修正し、
  version 4で10 CPU modelsを完走した。
- hard RMSE `8.587004 -> 8.477740`、3/5 folds改善、geop selection 19.50%、score guard 5/5、
  ID/confidence coverage 1.0でselector-addition guardをPASSした。
- fixed fallback `8.238332`には未達のため、Stage B時点ではStage C/Dを自動実行しなかった。
  後続のユーザー別承認でStage C/Dだけを追加実行し、inference/submissionは実行しなかった。
- model実体10件とmanifest SHA、小さいmetrics/schema/comparison artifactを取得・照合し、全実験記録を更新した。

## Stage C/D追加実行（2026-07-19承認）

- [x] Stage C full13 nested compact（40 CPU models）を実装・静的検証する。
- [x] Kaggle CPUでStage Cを完走し、40 models / 25 partitions / score / leakage / SHAを監査する。
- [x] Stage D full13 compact add-only（15 GPU models、control再学習0）を実装・静的検証する。
- [x] Kaggle T4でStage Dを完走し、保存済みparent12 add-onlyとOOF比較する。
- [x] result / metrics / SESSION_NOTES / experiment_summary / KAGGLE_DIRECTIONを更新する。

### Stage D確定結果

- [x] 15/15 GPU boostersとmodel manifestを確認した。
- [x] pooled RMSE `8.460811 -> 8.403784`、delta `-0.057027 ft`を確認した。
- [x] near / mid / 1000+、hidden-like 2面の改善を確認した。
- [x] fold改善2/5、worst-well `+5.862833 ft`により総合guard FAILを確定した。
- [x] inference / submissionをdisabledのまま維持した。
