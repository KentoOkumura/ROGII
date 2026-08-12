# 設計

## アプローチ

exp273 aggregate の plane diagnostics / by-well metrics と shard 0/1 candidate を SHA 固定入力にする。
shard candidate は `well,true_tvt,hmm_grad_*` だけを chunk 読みし、5候補の well-level RMSE を再集約して
aggregate by-well metrics と照合する。scalar control は保存済み by-well metrics を正とし、HMM は再実行しない。

raw horizontal well は `MD/X/Y/Z/TVT_input` だけを読み、`TVT_input.notna()` の known prefix を
full / tail 512 / tail 256へ切る。各 window に exp273 と同じ SVD geometry guard と deterministic Huber IRLSを
適用する。exp273 generation guardのvalidity/fallback/zero-gradientは別列に保存してfull再計算値を
保存済み plane diagnostics と parity guardする。一方、tail windowはguard不通過が多く角度・大きさ・
fit RMSEが全欠損になるため、min-points/rank-2を満たすwindowでは同じHuber IRLSのdiagnostic fitを計算する。
diagnostic fitはHMM generation、guard緩和、gradient scale調整には使わない。

full-valid 111 wells を primary cohort とする。各 window pairについて次を計算し、pair 最大値を取る。

- gradient angle disagreement: `angle_deg / 180`
- gradient magnitude disagreement: `abs(log((norm_a+eps)/(norm_b+eps))) / 5` を `[0,1]` clip
- plane RMSE disagreement: 同じ log-ratio clip
- rank-ratio disagreement: absolute gap を `[0,1]` clip
- condition disagreement: log-ratio clip
- validity flip: 3 window のいずれかが invalid なら 1

6成分を等重み平均した `stability_risk_score` を primary risk とする。scale、clip、等重み、window、
pair 集約は outcome 接続前に固定し、outer-train fitも行わない。outer fold は
`sha256("exp278::outer_fold::<well>") % 5` だけで固定する。

5 gradient candidates の well-level `delta_rmse_vs_scalar` 平均を primary outcome、max と candidate別を
secondary/report-only outcomeにする。risk と outcome の pooled / fold別 Spearman、固定 quintile readoutを保存する。

## 実験範囲

- 対象実験: `exp278_formation_gradient_prefix_stability_risk_readout_on_exp273`
- Route: `pf_beam`
- 親実験: `exp273_two_dimensional_formation_gradient_transition`
- 変更する変数: known-prefix window 別 plane diagnostics と事前固定 stability risk readout。
- 固定する変数: exp273 plane config、gradient candidate 5本、shard predictions、scalar/by-well outcome、
  full-valid cohort、outer fold hash、risk formula、quintile、guard。
- 対象外: HMM、GR emission、candidate path、gradient scale、guard threshold、selector、gate、inference、submission。

## 再現性設計

- seed policy: 数値処理は no RNG。outer fold は well id から stable SHA256。bootstrap なし。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 保存済み exp273 path を読むだけで新規生成なし。
- 並列処理と乱数の関係: `num_workers=1`、乱数なし。well は文字列順で処理する。
- CPU/GPU runtime: Kaggle CPU、GPU off、internet off。
- input SHA: aggregate CSV byte SHA、shard gzip raw/decompressed SHA、raw horizontal file SHA manifestを保存する。
- feature SHA: plane diagnostics / stability feature の logical content SHA と出力 file SHAを保存する。
- model / prediction / submission SHA: 新規 model/prediction/submission がないため対象外。
- Kaggle bootstrap: prepare 後に source/package/bootstrap の config/source SHA と metadata を照合する。
- deterministic anchor: readout は route anchor や submission anchor としない。Kaggle rerun前は
  deterministic diagnostic anchorとも呼ばない。

## リスク

- リークリスク: outcome を stability formula、fold、quantileへ使うと gate を事後最適化できる。
  コード上で feature frame と outcome frame を別に作り、risk 凍結後だけ one-to-one joinする。
- CV/LB 不一致リスク: これは新規 CV ではなく exp273 の train-side diagnostic。CV/LB anchorを更新しない。
- cohort リスク: full-valid は111 wellsだけ。fold最小15 wellsをtechnical guardとし、fallback 662 wellsは
  coverage/parity reportには残すが primary correlationに混ぜない。
- outcome リスク: candidate同士が近いため bank mean が実質同じ失敗を重複計上する可能性がある。
  candidate別とbank maxも保存するが primary guardは変更しない。
- ランタイム/メモリ: shard 3,783,989 rowsはchunk readし、raw wellは5列だけ逐次読む。
- 数値再現性: LAPACK/IRLSの微小差を許容する parity toleranceを事前固定し、差が超えればfail-closedする。
