# exp168_gr_matching_pair_visualization

## 状態

- ルート: pf_beam
- 状態: completed_visualization_diagnostic
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-02
- 親実験: exp167_fft_denoised_gr_matching_audit

## 仮説

GR matching の score だけでは失敗モードが見えにくいため、実際に比較した水平井 GR window と
matched typewell GR window を重ね描きすると、良い match / decoy / prior 依存の外れ方を目視できる。

## 変更点

- exp167 と同じ known-prefix linear TVT prior + typewell GR shift-scan を軽量に再実行する。
- 評価 row ごとに match pair metadata を保存する。
- selected pair について waveform overlay、shift score curve、typewell context の PNG と HTML index を作る。
- 学習、PF/Beam 生成、推論、提出は行わない。

## 検証方針

- Fold: なし。train-side diagnostic。
- Group: well 単位で deterministic subsample。
- Stratification: hidden_tail と prefix_backtest を分けて描画。
- Leakage Check: true TVT は評価と plot marker のみに使用し、best shift selection には使わない。

## 実行入口

- 学習 notebook: `exp168_gr_matching_pair_visualization_train.ipynb`
- 推論 notebook: `exp168_gr_matching_pair_visualization_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp168_gr_matching_pair_visualization`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| Kaggle kernel | `kentookumura/exp168-gr-matching-pair-visualization-train` v3 COMPLETE |
| rows scored | 4096 |
| wells scored | 16 |
| figures | 32 four-panel + 32 simple overlay |
| OOF join | exp098 lgb1 OOF error matched 9 selected IDs |
| output | `kaggle/output/train_v3/artifacts/` |

## 所見

### 良かった点

- GR matching pair の scored CSV、selected CSV、HTML index、32 PNG を Kaggle output として生成できた。
- 代表 PNG で水平井 GR、matched typewell GR、shift cost、true TVT marker が確認できる。
- v3 で query vs matched の単純 overlay PNG と、exp098 lgb1 OOF error に紐づけた good/bad HTML を追加できた。

### 悪かった点

- v1 は notebook kernelspec metadata 不足で Papermill 起動前に失敗した。v2 では `python3` kernelspec を追加して解消。
- v3 の OOF join は selected pair 32 件中 9 unique IDs に限定。未一致行は GR match error のみで読む。

### リスク / 注意

- 可視化診断なので CV / LB 改善を直接主張しない。
- PNG 数は config の `audit.max_total_figures` で制限する。

## 次

- HTML index を見て、追加で見たい well / region があれば `well_include` や `max_wells` を調整して追加実行する。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
