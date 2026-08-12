# 設計

## アプローチ

1. raw train directory から `*__horizontal_well.csv` を well 名順に列挙し、対応する
   `*__typewell.csv` の存在を確認する。
2. horizontal の `MD`, `GR`, `TVT_input`, `TVT` と typewell の `TVT`, `GR` だけを読む。
3. finite typewell pair を TVT 順に並べ、重複 TVT の GR median を取る。
4. alignment TVTを、known rowでは`TVT_input`、予測対象rowではtrain true `TVT`として作る。
   finite alignment TVTに`np.interp`を適用し、typewell TVT範囲外はNaNにする。
5. full-well共有 MD 軸の上下 2 段 plot を作る。上段は参照 GR、下段は horizontal GR。
   GR 軸は両段で共有し、`TVT_input.isna()`予測対象区間を着色する。
6. `artifacts/reference_vs_horizontal_gr/{well}.png`、manifest CSV、HTML index、summary JSON を保存する。

## 実験範囲

- 対象実験: `exp288_known_tvt_typewell_horizontal_gr_visualization`
- Route: `pf_beam`
- 親実験: `exp168_gr_matching_pair_visualization`（可視化構成のみ参照）。比較対象は `exp170` / `exp211` の known-prefix calibration 診断。
- 変更する変数: sampled shift-scan visualization から、全 train well の known + prediction-target direct interpolation comparison へ評価面を変更する。
- 固定する変数: raw GR、raw Type Well GR、known `TVT_input`、prediction-target true `TVT`、MD row order。smoothing、affine calibration、shift search、quality metric、model fitting は追加しない。

## 再現性設計

- seed policy: RNG なし。well 名の辞書順で逐次処理する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: すべてなし。
- 並列処理と乱数の関係: 初回は single process。Matplotlib の決定的な描画設定を固定する。
- CPU/GPU runtime と deterministic flags: CPU only、GPU off、internet off。
- train cache / test feature regeneration の SHA 記録方針: 入力ファイル相対パスとサイズを manifest に記録し、summary/manifest の SHA256 を保存する。PNG byte SHA は環境差があり得るため deterministic anchor の根拠にしない。
- model manifest / prediction / submission SHA 記録方針: model、prediction、submission は生成しない。
- Kaggle package bootstrap 確認方針: loose source と notebook source、埋め込み config、kernel metadata の route/CPU/internet/run flag を prepare 後に確認する。push は今回の依頼範囲外。

## リスク

- リークリスク: prediction-target train true `TVT`を参照GR図示に使う。train-only EDA限定であり、生成値・集計値を特徴量、model fit、候補選択、inference、submissionへ接続しない。
- CV/LB 不一致リスク: CV/LBを計算しないdiagnostic-only notebookである。
- ランタイム/メモリリスク: 773 PNG の I/O と Kaggle output サイズ。1 wellずつ読み、figureを毎回closeする。
- 再現性リスク: Matplotlib/Pillow versionでPNG byteが変わり得る。科学内容は補間契約とmanifestで固定する。
