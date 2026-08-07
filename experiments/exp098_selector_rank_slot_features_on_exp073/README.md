# exp098_selector_rank_slot_features_on_exp073

## 状態

- ルート: ml_model
- 状態: submitted_complete_public_lb_8_441
- CV: best `lgb1` 9.358151052 / `lgb_mean` 9.427447987
- Public LB: 8.441
- Private LB: -
- Submit ID: 53927490
- 作成日: 2026-06-21
- 親実験: exp073_gpu_reproducibility_guard_for_exp063_full_replay

## 仮説

exp093 で PF/Beam/likelihood-PF 候補集合には oracle headroom がある一方、現行 rank score は `pf_ancc` を拾えていない。候補を直接選択するのではなく、rank1/rank2/rank3 の差分、source identity、score gap、U-space residual/disagreement として exp073 LightGBM に渡せば、selector bias を抑えつつ候補集合の情報を使える可能性がある。

## 変更点

- exp072 deterministic full replay cache の 196 base features と exp073 target を固定。
- 候補は `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb` の 5 つ。
- rank slot は target-free score で作る。
- `rank_slot_u_disagreement` のみを学習する。この pattern は delta、identity/score、U-space projection、U-space disagreement の全 feature group を含む。
- ユーザー依頼により `lgb1` 単体で inference と competition submit まで実行した。

## 検証方針

- Fold: 5-fold GroupKFold
- Group: `well`
- Stratification: なし
- Leakage Check: rank slot と score は true TVT を参照しない。true TVT は LightGBM target と OOF metric のみに使う。

## 実行入口

- 学習 notebook: `exp098_selector_rank_slot_features_on_exp073_train.ipynb`
- 推論 notebook: `exp098_selector_rank_slot_features_on_exp073_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp098_selector_rank_slot_features_on_exp073 EXTRA_ARGS="--notebook train --run-on-push --strict"`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | best `lgb1` 9.358151052 / `lgb_mean` 9.427447987 |
| Inference | 14,151 rows / fallback 0 / submit-check PASS |
| Public LB | 8.441 |
| Private LB | - |

## 所見

### 良かった点

- exp092 の LightGBM fullrun runner をベースに、model manifest、OOF prediction、feature importance、by-well / bucket metrics を保存できる。
- fold/model ごとの特徴量重要度、平均特徴量重要度、上位特徴量プロットを保存する。
- rank-slot source distribution を `rank_slot_feature_summary.csv` に保存するため、`pf_ancc` を拾えているか確認できる。
- `lgb1` は exp073 raw anchor 9.526374749 から -0.168223697、exp077 policy 9.470514801 から -0.112363749 改善した。

### 悪かった点

- exp092 best `lgb1` 9.322479896 には届かない。
- `lgb_mean` は `lgb0` に引っ張られ、`lgb1` より弱い。

### リスク / 注意

- 全体 CV が改善しても by-well regression が出る可能性が高い。
- direct selector / soft average / postprocess replacement として扱わない。
- Kaggle inference v1 は完了し submit-check は通過した。
- Public LB 8.441 は exp077 ML route anchor 8.611 を改善したが、exp092 8.350 と ensemble route anchor exp082 7.601 には届かない。`ref=53927479` / 8.350 は exp092 に再帰属した。
- exp098 は exp092 未満の standalone 候補ではあるが、exp077 を明確に上回ったため rank-slot idea 自体は有用と判断する。次は exp092 に compact / top-n rank-slot signals を add-only でマージし、追加ゲインが残るかを見る価値がある。

## 次

1. exp098 は exp077 を上回る有用な rank-slot 比較基準として保持する。
2. `compact_rank_slot_features_on_exp098` と `selector_topn_candidate_only_features` で rank-slot feature のノイズ削減を試す。
3. exp092 feature surface に compact / top-n rank-slot signals を add-only でマージし、exp092 `lgb1` 9.322479896 / Public LB 8.350 を更新できるか検証する。
