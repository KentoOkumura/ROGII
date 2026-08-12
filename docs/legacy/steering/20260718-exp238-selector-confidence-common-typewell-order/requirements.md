# 要件

## 依頼

Kaggle notebook `kentookumura/exp238-oof-selector-confidence-probe`
（指定 `scriptVersionId=335655690`）が出力する全 well PNG を、well ID の辞書順ではなく
リポジトリ共通の typewell group 順に並べる。

## 制約

- Route: `ml_model`。exp238 の予測・selector score・評価値は変更せず、診断 PNG の列挙順と命名だけを変更する。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 共通 typewell 対応表は exp065 の `common_typewell_cluster_assignments.csv` を正とし、後続 exp238 でも使用している `native_overlap=0.999` を採用する。
- typewell group は exp065 の deterministic `cluster_id` 順、同一 group 内は `well_id` 順にする。
- 773 wells の対応表 coverage を fail-fast し、未対応 well を末尾へ暗黙 fallback しない。
- Kaggle output UI、manifest、plots zip の順を一致させるため、PNG 名に zero-padded typewell order と well ID を含める。
- model fit、candidate 再生成、submission 生成、competition submit は行わない。

## 受け入れ基準

- PNG が `typewell_0001_<well>.png` 形式になり、ファイル名の辞書順が共通 typewell 順と一致する。
- manifest に `plot_order`、`typewell_order`、`typewell_cluster_id`、`typewell_cluster_size`、`typewell_representative_well_id`、`plot_filename` が残る。
- manifest と zip member の順が `(typewell_order, well)` と一致し、well/path は773件 uniqueである。
- exp065 対応表の `method=native_overlap` / `threshold=0.999` が773 wellsを1回ずつ覆い、54 groupsになることを検証する。
- summary に typewell ordering contract、対応表 path/SHA、group数を保存する。
- Jupytext source、正規 `.ipynb`、Kaggle package、metadata の kernel source が同期している。
- `py_compile`、`ruff --select F821`、Jupytext `--test`、strict experiment validation が通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
