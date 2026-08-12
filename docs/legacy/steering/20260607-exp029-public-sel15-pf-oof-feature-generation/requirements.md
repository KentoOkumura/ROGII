# 要件

## 依頼

`public_sel15_pf_oof_feature_generation` を実装する。公開 replay で Public LB 8.781 を再現した sel15 PF/Beam route を、後続の selector / meta-stack が学習に使える train-side OOF-like feature artifact に変換する。

## 制約

- Route: `pf_beam`
- train well ごとに cutoff 以降の `TVT_input` を NaN にしてから PF/Beam を実行する。
- PF/Beam の入力に使ってよい列は hidden test でも使える `MD`, `X`, `Y`, `Z`, `GR`, typewell `TVT/GR`, cutoff 以前の `TVT_input` のみ。
- `TVT`, cutoff 以降の `TVT_input`, train-only formation/geology columns は PF/Beam 入力に使わない。
- 初回は full generation ではなく小サンプル smoke で leakage と runtime を確認できる設定にする。
- 出力は後続実験が直接読める CSV schema に固定する。

## 受け入れ基準

- `experiments/exp029_public_sel15_pf_oof_feature_generation/` に feature generator module、train notebook、config、記録がある。
- train notebook から `features/public_sel15_pf_oof_features.csv` と `artifacts/public_sel15_pf_oof_well_summary.csv` を生成できる。
- 保存 feature に `pf_pred`, `pf_pred - last_anchor`, optional `pf_pred - exp026_oof`, `abs_diff`, beam spread, best-second gap, path score, selected scale, prefix length, distance bucket, GR availability が含まれる。
- `task validate-exp EXP=exp029_public_sel15_pf_oof_feature_generation` と静的チェックが通る。
