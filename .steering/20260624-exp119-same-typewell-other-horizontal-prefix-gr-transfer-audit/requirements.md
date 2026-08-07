# 要件

## 依頼

`same_typewell_other_horizontal_prefix_gr_transfer_audit` を実装する。

同じ native typewell overlap group の他 horizontal well について、source 側は疑似 predict start 前の raw GR と `TVT_input` 相当だけを使い、query well の evaluation-zone raw GR に対応付けて TVT path prior として転用できるかを診断する。

## 制約

- Route: `ensemble`
- Kaggle Notebook 実行を正とする。
- source pool は train-fold wells のみ。validation well と同 fold validation true TVT を source に入れない。
- source 側は pseudo predict start 前の prefix raw GR と `TVT_input` に限定し、source tail true TVT を使わない。
- test batch 内の他 well `TVT_input` 利用や rules 解釈は本実験では扱わない。
- direct submit はしない。改善した場合も raw-test parity / rules audit を別実験で行う。
- 再現性: `docs/06_reproducibility.md` に従い、upstream cache SHA、cluster assignment SHA、OOF prediction SHA、gzip decompressed content SHA を記録できる構成にする。

## 受け入れ基準

- `.steering/`、`config.yaml`、train/inference notebook、補助 `.py`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json` が exp119 として整合している。
- `native_overlap_0p999` と `exact_hash` group を fold-safe source pool として使える。
- `same_typewell_gr_match`、`same_typewell_random_control`、`different_typewell_gr_match` を同一指標で比較できる。
- offset / slope / path delta の prior と、`likpf_mean` / `pf_ancc` / `beam_mean` への clipped correction を評価できる。
- candidate metrics、bucket metrics、by-well metrics、signal metrics、OOF prediction、feature schema、summary JSON を生成できる。
- `validate-exp` と Python 構文チェックが通る。
- gzip 生成物を比較する場合は raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
