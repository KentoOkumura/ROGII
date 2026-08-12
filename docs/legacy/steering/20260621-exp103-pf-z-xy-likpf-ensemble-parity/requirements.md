# 要件

## 依頼

exp100 の best variant `pf_z_xy_slope` を、exp072 `lik_pf` と同じ土俵で比較できるようにする。

- 128 seed 化する。
- likelihood-weighted scale ensemble を作る。
- `pf_mean` / `pf_scale_*` 相当の `xy_likpf_mean` / `xy_likpf_scale_*` を出す。
- exp072 保存済み `pf_z` / `likpf_*` と同じ candidate metrics で比較する。

## 制約

- Route: `pf_beam`
- 学習モデルは作らない。train-side pseudo-tail の PF candidate parity audit に限定する。
- exp072 の feature cache を固定 baseline として読み、baseline candidate の再生成はしない。
- XY rate prior は prefix の有限 `TVT_input` 行だけで推定し、評価区間の true TVT は scoring 以外に使わない。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行はしない。
- 再現性: stochastic PF なので stable seed、input SHA、exp072 cache SHA、gzip decompressed SHA を記録する。

## 受け入れ基準

- `config.yaml` に `pf_beam` route、exp100 parent、exp072 cache parent、Kaggle kernel source が明記されている。
- train notebook で設定、exp072 cache preview、raw input preview、audit 実行、metrics preview が確認できる。
- helper が `xy_likpf_mean` と `xy_likpf_scale_3/5/8/12` を生成する。
- `candidate_metrics.csv` に exp072 `pf_z`、`likpf_mean`、`likpf_scale_*`、exp103 `xy_likpf_*` が同じ行集合で並ぶ。
- gzip 生成物は decompressed content SHA を summary に記録する。
