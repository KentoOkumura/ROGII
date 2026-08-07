# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある、typewell と horizontal well の GR シグネチャを合わせる local matcher / multi-scale NCC を実装する。

## 制約

- 既存 raw GR rolling / delta feature の置換ではなく、add-only の補助特徴として検証する。
- feature alignment に evaluation-zone true `TVT` を使わない。
- 使用してよい情報は horizontal well の `MD`/`GR`、paired typewell の `TVT`/`GR`、known `TVT_input` prefix、既存 trajectory/prefix features に限定する。
- `ANCC` などの train-only formation columns は使わない。
- 初回 full notebook 実行は Kaggle 上を正とし、ローカル notebook 実行はしない。

## 受け入れ基準

- `exp008_gr_ncc_matcher` が作成され、train / inference notebook で paired `__typewell.csv` を使える。
- `no_gr_signal_plus_gr_ncc` と `all_plus_gr_ncc` の feature set が validation 可能。
- `control_exp002_all`、`control_exp003_no_gr`、`gr_ncc_no_gr_multi`、`gr_ncc_all_multi` を同一 GroupKFold で比較できる。
- `task validate-exp`、lint、py_compile、pytest、Kaggle notebook prepare が通る。
