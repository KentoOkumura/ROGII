# 設計

## アプローチ

raw train well の known prefix から slope prior を作り、typewell GR と horizontal GR local window の shift-scan cost surface を計算する。各評価 row で top1 mode と、6-30ft 離れた top2 local minimum を抽出し、固定温度 posterior から `p * mode1 + (1 - p) * mode2` の候補を作る。

exp133 の midpoint/proxy 失敗を踏まえ、midpoint は比較対象に留める。positive でも immediate replacement には進まず、posterior candidate / `p` / entropy / gap を後続 selector または ML confidence feature の材料にする。

## 実験範囲

- 対象実験: `exp171_bimodal_posterior_pfbeam_candidate_audit`
- Route: `pf_beam`
- 親実験: `bimodal_posterior_pfbeam_candidate_audit` backlog、`exp133`、`exp167`、`exp170`、`exp072`
- 変更する変数: GR shift-scan top2 mode 抽出、posterior temperature、filter surface
- 固定する変数: raw train data、fixed exp072 candidate cache、sampled row policy、no ML training、no inference

## 再現性設計

- seed policy: no RNG。sampled rows は deterministic linspace。
- stochastic 処理の有無: 新規なし。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。upstream exp072 cache は stochastic component として SHA を記録する。
- 並列処理と乱数の関係: 並列処理なし、global RNG なし。
- CPU/GPU runtime と deterministic flags: Kaggle CPU。GPU 無効。
- train cache / test feature regeneration の SHA 記録方針: row context gzip は raw gzip SHA と decompressed content SHA を記録する。candidate cache SHA も summary に残す。
- model manifest / prediction / submission SHA 記録方針: 対象外。ML model、prediction、submission は作らない。
- Kaggle package bootstrap 確認方針: prepare 時に config と kernel source を確認し、Kaggle train result を SESSION_NOTES に記録する。

## リスク

- リークリスク: mode 抽出や温度選択に true TVT を使うと leak になる。true TVT は scoring のみに限定する。
- CV/LB 不一致リスク: train-side diagnostic であり LB 直接候補ではない。positive でも raw-test parity / hidden-like stress が必要。
- ランタイム/メモリリスク: all wells x sampled rows x filters x shifts の診断。row context gzip は大きくなり得るため、集計 CSV を主要証拠にする。
- 再現性リスク: exp072 cache は upstream stochastic component。exp171 自体は RNG を使わない。
