# 設計

## アプローチ

exp099 v2 feature cache の OOF pseudo-tail row ids から、各 train well の pseudo predict start と visible anchor を復元する。raw train horizontal CSV から raw GR / MD / TVT / TVT_input を読み、source wells では pseudo predict start 前だけを利用する。

validation well の evaluation-zone raw GR window を、train-fold source well の prefix raw GR windows と normalized dot product で照合する。best source-prefix window から次の 3 種類の delta を作る。

- `offset`: source prefix match center の `TVT_input - source_anchor_TVT`
- `slope`: query `md_since` に source local TVT/MD slope を掛けた値
- `path`: source offset から query `md_since` まで source local slope で延長した値

各 delta を query anchor `last_known_tvt` に足して prior TVT とし、prior 単体と既存 `likpf_mean` / `pf_ancc` / `beam_mean` への clipped correction を評価する。

## 実験範囲

- 対象実験: `exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit`
- Route: `ensemble`
- 親実験: `exp109_typewell_neighbor_prior_features`
- 参照: `exp065_typewell_supertype_cluster_cv_audit`、`exp099_pf_multi_observation_likelihood_probe`
- 変更する変数: source prefix raw GR matching と source prefix TVT_input transfer prior
- 固定する変数: exp099 PF/Beam/likPF candidate cache、exp065 typewell cluster assignment、well-grouped 5 folds、submit なし

## 再現性設計

- seed policy: fixed seed 42 の well GroupKFold。same-typewell random control は global RNG を使わず、query/source/row key の SHA256 から source center index を決める。
- stochastic 処理の有無: 新規 stochastic 処理なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行なし。exp099 upstream cache を固定入力として読む。
- 並列処理と乱数の関係: 実装は single-process。global RNG と thread scheduling に依存しない。
- CPU/GPU runtime と deterministic flags: CPU only、GPU disabled。
- train cache / test feature regeneration の SHA 記録方針: exp099 gzip raw SHA と decompressed content SHA、schema SHA、exp065 cluster assignment SHA を summary JSON に記録する。
- model manifest / prediction / submission SHA 記録方針: model と submission はない。OOF prediction gzip は raw SHA と decompressed content SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に generated package の config と bootstrap に exp118 の config / helper が入ることを確認する。

## リスク

- リークリスク: source prefix 以外の source true TVT を使うと leakage になる。実装では source center を `eval_start_row_index` より前に限定し、valid well を source pool から除外する。
- CV/LB 不一致リスク: train pseudo-tail の same-typewell relation が hidden test と異なる可能性がある。改善しても raw-test parity / rules audit 前に submit しない。
- ランタイム/メモリリスク: GR window matching は query/source/window の積で重くなる。`max_source_wells`、`query_stride_rows`、`source_stride_rows`、`chunk_size` で制限する。
- 再現性リスク: upstream exp099 PF/Beam cache は別実験由来。summary に input SHA を残し、exp118 自体は deterministic posthoc audit として扱う。
