# 設計

## 仮説

128 seed平均で潰れる複数の長期trajectory modeが存在し、実在seed medoidとして保持すれば
exp237 base8 unionへblock / whole-well candidate headroomを追加できる。

## アプローチ

各wellのexp072 pseudo-tailをraw trainから再構成し、exp072互換likelihood-PFを1回だけ実行する。
得られた`128 × eval_rows` trajectory matrixに、tail前半1.0 / 後半1.5のweighted RMSEを
適用して128×128距離行列を作る。決定的BUILD初期化とbest-improvement PAMでK=3/5/8を
別々にcluster化し、各clusterのmedoid seed trajectoryをcandidateとして保存する。

候補生成後にだけtrue TVTを使い、medoid bank単独、`likpf_mean` fallback付きbank、exp237
base8 union付きbankについてrow、128/256/512-row block、whole-well oracleを計算する。

## 実験範囲

- 対象実験: `exp243_pf_seed_medoids`
- Route: `pf_beam`
- 親実験: `exp072_exp063_full_replay_feature_cache`
- 実装参照: `exp241_adaptive_likelihood_pf_trajectory_containment_audit`
- 変更する変数: seed平均前のtrajectoryを保持し、固定K-medoidsで実在medoid候補へ展開する。
- 固定する変数: raw/typewell GR、prefix、particle、transition、likelihood、resampling、seed contract。

## 再現性設計

- seed policy: exp072 SHA256 modulo + 1 per-well seed base + seed index。
- dtype policy: raw PF入力、GR grid、PF trajectory、replay mean、K-medoids入力までfloat64。保存列だけfloat32。`astype(float64)`前のfloat32丸めも禁止する。
- stochastic処理: particle初期化、伝播、conditional resampling。
- PF: 500 particles × 128 seeds、1 replay/well。
- clustering: numpy CPU、決定的BUILD+PAM、seedやthread RNGなし。
- 並列処理: Numba single worker。well間thread並列を使わない。
- runtime: Kaggle CPU、GPU false、internet false。
- artifact方針: 全128 trajectoryを永続化せず、medoid row candidate、cluster membership、診断へ縮約する。
- SHA: canonical exp072 v2 input cache/schemaとexp209 enriched parity controlの期待SHAをconfigに固定し、不一致ならPF開始前に停止する。gzip decompressed row candidates/cluster manifest、各CSVも記録する。
- model/submission: model、manifest、submissionなし。
- bootstrap: push前にgenerated notebook内config、kernel sources、CPU/internet、active shardを確認する。
- parity probe: full 4 shard再実行前に固定1 wellでsaved exp072 `likpf_mean`とのexact parityを確認する。

## 候補集合

- fallback: `pf_base_likpf_mean`
- medoids: `pf_seed_medoid_k{3,5,8}_m{slot}`
- exp237 base8: PF ANCC、Beam mean、likPF mean、SC ensemble、hybrid、dense/densew/dense50。
- bank: medoid-only、fallback+medoid、base8、base8+各K、base8+全固定K。

## リスク

- リークリスク: clustering/K/medoid/orderにはtargetを渡さず、oracleは候補固定後の診断列に限定する。
- CV/LBリスク: pseudo-tail train-side auditであり、raw-test/LB採用根拠にしない。
- ランタイム: full replayは長いためstable hash 4 well shard。各wellのtrajectory matrixは処理後解放する。
- メモリ: global 128×3.78M matrixを作らず、well単位の128×eval_rowsだけ保持する。
- Monte Carlo noise: row oracleだけ改善しblock/whole-well改善がない、cluster massがsingleton、medoid間が近重複なら閉じる。
- selector risk: exp237のnear/worst-well guard失敗を踏まえ、selector/safety guard前にinference/submitしない。

## 次のアクション

one-well exact parity確認後、ユーザー承認によりKaggle CPU 1 notebookで全773 wellsを実行する。
単一source / 単一configの生成物を監査し、modeかMonte Carlo noiseかを判定する。
