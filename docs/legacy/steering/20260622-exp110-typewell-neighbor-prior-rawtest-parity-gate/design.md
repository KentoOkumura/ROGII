# 設計

## アプローチ

exp109 の prior 生成を再利用し、source mode を 2 つに分ける。

- `oof`: GroupKFold の train-fold wells だけを neighbor source にする。gate CV の正。
- `full_train`: query well 自身を除く full train same-cluster wells を neighbor source にする。本番推論に近い parity 診断専用。

gate は exp109 best correction `likpf_mean + 0.20 * clip(prior_tvt - likpf_mean, -40, 40)` を固定し、補正を適用する row だけを条件で絞る。条件は `prior_std`、row prior count、neighbor wells、correction abs、cluster size の小 grid とする。

## 実験範囲

- 対象実験: `exp110_typewell_neighbor_prior_rawtest_parity_gate`
- Route: ensemble
- 親実験: `exp109_typewell_neighbor_prior_features`
- 変更する変数: gate 条件、full-train-source prior parity 診断
- 固定する変数: exp099 train feature cache、exp065 cluster assignments、native overlap 0.999、base `likpf_mean`、alpha 0.20、clip 40ft、GroupKFold seed 42

## 再現性設計

- seed policy: deterministic GroupKFold seed 42。新規 stochastic 処理なし。
- stochastic 処理の有無: なし。上流 PF/Beam cache は固定 artifact として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: この実験内では再実行しない。
- 並列処理と乱数の関係: 乱数なし。numpy deterministic aggregation のみ。
- CPU/GPU runtime と deterministic flags: CPU、GPU disabled。
- train cache / test feature regeneration の SHA 記録方針: exp099 cache raw/decompressed SHA、exp065 cluster assignment SHA、OOF prediction raw/decompressed SHA を summary に保存する。
- model manifest / prediction / submission SHA 記録方針: model なし。submission なし。prediction は OOF artifact SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks.py --strict` 後の Kaggle train notebook を正とする。

## リスク

- リークリスク: full-train-source prior を CV score として扱うと validation fold の peer labels を使うため、primary CV には使わない。
- CV/LB 不一致リスク: test-side candidate surface がまだないため、global OOF 改善だけで submit しない。
- ランタイム/メモリリスク: gate grid は全候補予測列を保存せず metrics のみを計算し、best gate だけを保存する。
- 再現性リスク: 上流 exp099 / exp065 artifacts の SHA を記録する。
