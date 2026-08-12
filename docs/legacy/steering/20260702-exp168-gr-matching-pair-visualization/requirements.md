# 要件

## 依頼

GR マッチングで実際に比較された波形同士を、Kaggle Notebook 上で確認できる可視化 notebook を作成する。
対象は `exp167_fft_denoised_gr_matching_audit` と同じ train-side typewell GR shift-scan とし、
各評価 row について水平井 GR の local window と best shift 後の typewell GR window を pair として保存・描画する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行は行わない。
- モデル学習、PF/Beam 生成、予測置換、提出は行わない。
- true TVT は train-side の評価・図示 annotation のみに使い、match center や filter selection には使わない。

## 受け入れ基準

- `exp168_gr_matching_pair_visualization_train.ipynb` が Kaggle 上で raw train input から可視化を生成できる。
- `artifacts/exp168_gr_matching_pair_visualization_scored_pairs.csv.gz` に、well、row、region、filter、prior TVT、best shift、matched TVT、score、error が保存される。
- `artifacts/exp168_gr_matching_pair_visualization_selected_pairs.csv` に、描画対象 pair と PNG path が保存される。
- `artifacts/figures/` に、水平井 window、matched typewell window、shift score curve、typewell context を含む PNG が保存される。
- `artifacts/exp168_gr_matching_pair_visualization_index.html` から PNG と主要 metadata を一覧できる。
- gzip 生成物は raw `.csv.gz` SHA ではなく decompressed content SHA を summary に記録する。
