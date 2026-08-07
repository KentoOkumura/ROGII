# 設計

## 1. 結論

`exp491_exp226_final_tvt_rate_direct_hmm` は、exp437 の TVT-only HMM に対して、遷移 schedule の入力だけを exp226 geometry-only `tvt_geop` から exp226 最終 `tvt_pred` へ交換する単一要因実験とする。

rate の履歴状態や補正器を導入する実験ではない。exp226 最終予測から各行の TVT 増分を計算し、それを HMM の遷移中心へ無加工で与える。HMM は絶対 TVT の posterior のみを保持する。

## 2. 仮説

exp408 では HMM の message-derived rate に追従遅れがあり、exp410 では PF 側にも rate basin の問題が確認された。一方、exp226 の最終 `tvt_pred` は geometry-only 中間値より良い OOF を持つ。

したがって、exp226 最終予測が含む局所 TVT-rate signal を遷移 schedule として固定し、HMM は絶対位置の posterior と GR emission による補正だけを担当させると、exp226 の累積 offset を補正しながら rate 追従遅れを減らせる可能性がある。

## 3. 遷移 schedule の厳密な定義

well ごとに既知 prefix の最終行を境界とし、未知 suffix の exp226 最終予測を `p_t`、Measured Depth を `MD_t`、垂直座標を `Z_t` とする。

- 最初の未知行:
  - `delta_tvt_0 = p_0 - last_known_TVT_input`
- 2 行目以降:
  - `delta_tvt_t = p_t - p_(t-1)`
- 監査用 TVT rate:
  - `tvt_rate_t = delta_tvt_t / delta_MD_t`
- 監査用 U rate:
  - `u_rate_t = (delta_tvt_t + delta_Z_t) / delta_MD_t`
- HMM 遷移中心:
  - `transition_center_t = delta_tvt_t`

`u_rate_t * delta_MD_t - delta_Z_t = delta_tvt_t` の恒等式を数値監査する。HMM に渡すのは `u_rate` そのものではなく、等価な TVT 増分 `delta_tvt_t` である。

禁止事項:

- exp226 schedule に対する smoothing、clipping、scaling、segment 集約
- rate state、rate momentum、rate noise model
- 残差 offset/rate state、branch state
- exp226 との blend、selector、oracle 選択
- emission、grid、kernel、開始事前分布の同時変更
- PF 分岐

## 4. 固定する HMM

exp437 の以下を固定する。

- 状態: absolute TVT grid の確率分布のみ
- grid step: 0.35 ft
- transition `sig_p`: 0.02
- start sigma: 0.75 ft
- band padding: 100 ft
- position kernel: 5 cells
- emission: Gaussian typewell-GR、lambda 1.0
- emission sigma: 既知 prefix の population standard deviation、`[10, 60]` に clip
- missing GR: 両方向補間後に typewell mean
- inference: forward-backward
- output: smoothed posterior mean

比較の主 control は保存済み exp226 最終 `tvt_pred` とする。exp437、exp209、exp281、exp357、exp263 は機構理解または report-only comparison とし、再実行しない。

## 5. 入力と leakage guard

exp226 OOF は decompressed content SHA
`709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
と一致する保存物だけを使う。

予測凍結前に exp226 OOF から読む列は次だけとする。

- `well_id`
- `row_idx`
- `suffix_offset`
- `fold`
- `tvt_pred`

`TVT`、`tvt_true`、`tvt_geop`、`gr_delta`、`error`、`abs_error` を含め、ファイル全列を読んで後から落とすことは禁止する。候補予測、diagnostic、logical prediction SHA を凍結後にだけ真値・episode role・評価 scope を join する。

固定 32 well の ID は予測前に使えるが、persistent/control role、episode label、fold outcome は凍結後に付与する。

exp226 最終予測は suffix GR を既に使っており、HMM emission でも raw suffix GR を再利用する。これは未知 TVT の target leakage ではない。ただし同じ観測証拠を二度使うため、raw-GR observed/missing 別の非劣化を Stage 1 gate に含める。

## 6. Stage 0: 固定 32 well の機構確認

目的は「最終 `tvt_pred` 差分を直接入れると persistent episode が改善し、matched control を壊さないか」を安価に確認すること。CV や本昇格判定ではない。

- manifest: exp411 `stage0_fixed32_manifest.csv`
- 16 persistent + 16 matched control
- expected suffix rows: 156,088
- candidate: 1
- HMM well-run: 32
- control rerun: 0

技術 gate:

- well、row、fold、SHA、coverage が期待値と一致
- duplicate/missing row が 0
- schedule prediction と正の `delta_MD` coverage が 100%
- 予測凍結前の forbidden column、role、episode 読み込みが 0
- first-difference parity と rate identity の最大誤差が `1e-10 ft` 以下
- transition row sum と posterior normalization の最大誤差が `1e-6` 以下
- full 773 well への projected runtime が 3,600 秒以下
- peak RSS が 8 GB 以下

機構 gate:

- all32 RMSE が exp226 最終予測より 0.10 ft 以上改善
- matched control の悪化が 0.02 ft 以下
- persistent well RMSE が 0.10 ft 以上改善
- 5 fold 中 4 fold 以上で改善
- persistent episode SSE を 5% 以上削減
- paired by-well delta の p95 が 0.25 ft 以下
- worst-well delta が 2.0 ft 以下

1 つでも不合格なら exp491 を終了し、同じ実験内で smoothing、clipping、emission、grid、blend、selector、PF による救済をしない。

## 7. Stage 1: full group-safe OOF

Stage 0 の全 gate 合格後、結果をユーザーへ提示し、別承認を得た場合だけ実行する。

- 773 wells / 3,783,989 unknown suffix rows / 5 folds
- candidate: 1
- HMM well-run: 773
- control rerun: 0

主 promotion gate:

- full OOF RMSE `<= 9.377109596582213 ft`
  - 保存済み exp226 `9.427109596582213 ft` に対し 0.05 ft 以上改善
- 5 fold 中 4 fold 以上で改善
- MD 1000+ で 0.05 ft 以上改善
- hidden-like spatial と typewell-purged を悪化させない
- near 0–250 ft の悪化が 0.02 ft 以下
- raw-GR observed と missing の両方を悪化させない
- persistent episode SSE を 5% 以上削減
- paired by-well delta p95 `<= 0.25 ft`
- worst-well delta `<= 2.0 ft`

不合格なら inference へ進めず、exp491 は negative result として終了する。

## 8. PF への進行条件

PF は exp491 の内部 variant にしない。exp491 の Stage 0/1 結果をレビューし、rate signal の改善と安全性が確認できた場合にのみ、ユーザー承認後に別 steering・別 exp を作る。

PF 後続で採用する具体的な注入方法、noise model、particle 数、resampling 設計は現時点では決めない。HMM の結果を見る前に PF 設計を固定しないことで、失敗した rate signal をそのまま PF へ持ち込むことを防ぐ。

## 9. 再現性設計

- stochastic processing: なし
- seed: 設定互換のため 42 を記録するが候補生成には使わない
- stable ordering: `fold, well_id, row_idx`
- parallelism: HMM well 間並列を将来使う場合も各 well は独立で、乱数・実行順に依存させない
- runtime: CPU、single deterministic numeric path を基準に記録
- inputs: exp226 decompressed content SHA、fixed32 manifest SHA、episode asset SHA、hidden-like asset SHA を照合
- outputs: candidate prediction の decompressed/logical content SHA、metrics JSON、runtime、RSS を記録
- deterministic anchor: 初回実行だけでは主張しない
- GPU / package bootstrap: 使用しない。将来 Kaggle notebook 化する場合は metadata と実行環境を記録する

## 10. 今回の作業範囲

今回作るもの:

- steering 3 文書
- 実験 scaffold と設計済み config/docs
- `KAGGLE_DIRECTION.md` の HMM 現行案と条件付き PF 後続案
- `experiment_summary.md` の設計段階エントリ

今回作らないもの:

- HMM 実装
- Jupytext source、notebook 実装、テスト
- Kaggle package、kernel push、Stage 0/1 実行
- inference、submission
- PF 実験
