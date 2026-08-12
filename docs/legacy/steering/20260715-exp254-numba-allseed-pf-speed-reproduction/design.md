# 設計

## アプローチ

1. exp072 cache の SHA と raw horizontal/typewell 入力から exp243 と同じ masked evaluation zone を構築する。
2. eval length の固定分位に最も近い well を、距離・well id の順で決定的に選ぶ。
3. exp243 v3 の all-seed Numba core を reference として固定する。
4. 同じ single-seed body を Numba 関数化し、Python 側 seed loop で呼ぶ legacy 実装を用意する。
5. reference all-seed、legacy loop、benchmark all-seed の trajectory / log-likelihood / aggregate を比較する。
6. seed bank を `.npz` に保存・再読込し、temperature / mean / prefix subset の candidate spec を最大300本まで deterministic に再合成する。
7. `perf_counter` と `resource.getrusage` で runtime / peak RSS を記録し、実測値と 773-well 外挿を分離する。
8. parity・repeated SHA・cache round-trip guard がすべて通過した場合だけ `full_workload` mode を許可する。

## 実験範囲

- 対象実験: `exp254_numba_allseed_pf_speed_reproduction`
- Route: `pf_beam`
- 親実験: `exp243_pf_seed_medoids`
- 参照実験: `exp072_exp063_full_replay_feature_cache`、`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: PF kernel の seed-loop placement、seed count、warm candidate spec count。
- 固定する変数: raw input、exp072 evaluation zone、particles=500、seed base/order、PF dynamics、RNG call order、resampling、float64、single process、Numba thread count。
- 既定 mode: `probe`。full mode は同じ固定設定の全773 wellsを対象とするが、probe summary path と SHA を明示し guard 通過を確認しない限り停止する。

## benchmark 出力

- `probe_wells.csv`: 分位、well、eval rows、入力 SHA。
- `parity.csv`: implementation / seed count ごとの exact parity、max abs diff、SHA。
- `timings.csv`: compile、legacy PF、all-seed PF、cache write/read、warm aggregation。
- `candidate_specs.csv`: 事前生成した最大300 spec の temperature / subset / aggregation contract。
- `cache_manifest.csv`: seed bank shape/dtype/content SHA、file SHA、round-trip 判定。
- `full_runtime_projection.csv`: probe 実測からの773-well外挿。`measurement_kind=projection` を必須とする。
- `summary.json` / `metrics.json`: guard、環境、thread、peak RAM、生成物 SHA。

## 再現性設計

- seed policy: `sha256("likpf::train::<well>")[:16] % 2147483647 + 1 + seed_index`。
- stochastic 処理の有無: PF 初期化、propagation、conditional systematic resampling に限定する。
- PF/Beam / likelihood-PF / seed bagging の有無: likelihood-PF 500 particles × 最大128 seeds。Beam/model学習なし。
- 並列処理と乱数の関係: process 1、Numba thread固定、seed body内で `np.random.seed(seed_base + seed_index)`。thread schedulingに依存しない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU off、internet off。Numba `njit(cache=True, nogil=True)`、parallel=False。
- train cache / test feature regeneration の SHA 記録方針: exp072 gzip は decompressed SHA、raw input filesはraw SHA、seed bankはarray dtype/shape/bytesを順序付きSHAへ入れる。
- model manifest / prediction / submission SHA 記録方針: model / submissionなし。per-seed trajectory、log-likelihood、aggregate candidate SHAを記録する。
- Kaggle package bootstrap 確認方針: prepare後にsource/configとbootstrap内のbyte一致、CPU/internet/thread/mode/particle/seed gridを確認する。

## リスク

- リークリスク: true TVTは読み込まず、cacheのtarget列もPF入力・candidate spec・計時・guardに使わない。
- CV/LB不一致リスク: 精度評価・提出を行わないため対象外。runtime基盤としてのみ採否を判断する。
- ランタイム/メモリリスク: 128×long-well trajectoryは大きい。1 wellずつ処理・保存し、legacy referenceはwall-time budgetで停止可能にする。
- 再現性リスク: single-seed関数への分割でRNG call orderが変わる可能性がある。reference all-seedとのexact parityを採用guardにする。
- cacheリスク: `.npz` container metadataではなく、array content SHAを主証拠にする。
