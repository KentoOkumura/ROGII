# 設計

## アプローチ

既存の exp083 v12 plot notebook に exp226 OOF 入力を追加する。exp226 OOF は `well_id,row_idx,tvt_pred` 形式なので、`id = f"{well_id}_{row_idx}"`、`well = well_id`、`exp226_k16_oof_tvt = tvt_pred` へ変換して、既存の exp072 feature-cache plot frame に `id,well` で left join する。

可視化では exp148 OOF と同じ TVT パネルに exp226 OOF を線として重ねる。RMSE は plot frame の `true_tvt` と exp226 OOF 予測から計算し、global / per-well / manifest / summary に記録する。

## 実験範囲

- 対象実験: `experiments/exp083_pf_beam_true_tvt_2d_well_eda`
- Route: `pf_beam`
- 親実験: exp083 v12 visualization、比較入力として exp226 train OOF
- 変更する変数: plot overlay、title fields、manifest fields、summary source/coverage/sha 記録
- 固定する変数: exp072 PF/Beam feature cache、exp148 OOF、exp209 HMM overlay、raw train data、plot well scope

## 再現性設計

- seed policy: なし。既存 OOF を読み込む診断可視化のみ。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 既存 exp072 feature cache と exp226 OOF を読むのみ。再生成しない。
- 並列処理と乱数の関係: なし。
- CPU/GPU runtime と deterministic flags: notebook は描画処理のみ。GPU 不要。
- train cache / test feature regeneration の SHA 記録方針: 入力 gzip は raw gzip SHA と decompressed SHA を summary に記録する。
- model manifest / prediction / submission SHA 記録方針: 新規モデル、予測、提出は生成しない。exp226 OOF は exp226 側 metrics の decompressed SHA を参照し、plot summary にも計算値を保存する。
- Kaggle package bootstrap 確認方針: `.py` から `.ipynb` を Jupytext で再生成し、round-trip と構文チェックを行う。

## リスク

- リークリスク: train-side OOF 診断可視化のみで提出に使わない。exp226 OOF は exp226 の group-safe CV 生成物を読む。
- CV/LB 不一致リスク: exp226 は CV 9.427 / Public LB 9.837 の不採用結果であり、plot は採用判断ではなく形状診断に使う。
- ランタイム/メモリリスク: exp226 OOF は約 378 万行で exp148 と同程度に chunk 読み込みする。
- 再現性リスク: Kaggle input に exp226 train output が無い場合はファイル解決に失敗する。エラーメッセージに必要 input slug を含める。
