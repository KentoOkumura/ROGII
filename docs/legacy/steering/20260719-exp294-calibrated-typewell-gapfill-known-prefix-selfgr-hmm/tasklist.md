# タスクリスト

## 未着手 / 実行中

- なし。

## ブロック中

- Stage 1実装/実行: Stage 0 performance hard gate FAILにより閉鎖。
- inference/submission: Stage 1閉鎖により未実行・閉鎖。

## 完了

- [x] `kaggle-review-exp` / `kaggle-strategy` の手順と `docs/06_reproducibility.md` を確認した。
- [x] `docs/legacy/steering/20260719-exp294-calibrated-typewell-gapfill-known-prefix-selfgr-hmm/` を作成した。
- [x] `experiments/exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm/` をtemplateから作成し、親実装をコピーしなかった。
- [x] 変更対象をknown-prefix self-GR donorのraw missing cellだけに固定した。
- [x] Stage 0 / Stage 1のhard gate、リーク防止、再現性、runtime、禁止事項を確定した。
- [x] active variant / model countを確定した: Stage 0 audit 1、Stage 1 HMM 1、773 well-runs、LightGBM config / fold / booster `0/0/0`。
- [x] バックログ、実験記録、experiment summaryをdesign-only状態へ更新する。
- [x] Jupytext percent形式の別名 compact self-contained train script/notebookへStage 0だけを実装した。
- [x] Type Well duplicate median、range内挿、deterministic Huber IRLS、hybrid donor、raw-mask parityを実装した。
- [x] stable SHA256 5 folds、fold外missing-run quantile、非重複pseudo-mask、truth-late-joinを実装した。
- [x] Stage 0生成物、SHA manifest、hard-gate判定、専用contract tests 11件を実装した。
- [x] Jupytext、syntax、Ruff、`make validate-exp`、`make validate-template`、repository tests 298件を通した。
- [x] 実装結果と実行規模を記録し、Kaggle CPU実行は明示承認待ちとしてfail-closeした。
- [x] ユーザーから正規train notebook採用と1件のKaggle CPU Stage 0実行承認を得た。
- [x] canonical CPU kernel version 1（id_no `127890033`）をpushし、Kaggle metadataをpull監査した。
- [x] 773 wells / 2,319 blocks / 3,865 rowsのStage 0を160.32秒で完了した。
- [x] technical gate 6件PASS、performance gate 5件FAIL、Stage 1 unauthorizedを確認した。
- [x] Kaggle outputを取得し、artifact manifest 9 entriesのbyte数・raw/decompressed SHAを全照合した。
- [x] Stage 1、救済grid、inference、submissionを閉じ、実験記録と横断記録を更新した。
