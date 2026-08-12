# 要件

## 依頼

`exp215_mtp_full_tail_heatmap_path_generator_probe` backlog を実験化する。exp212 の fallback-heavy / endpoint hold 直線 tail 問題を、heuristic stitch ではなく learned `path_logit` を持つ MTP 型 full-tail heatmap path generator で改善できるか train-side diagnostic として確認できる状態にする。

## 制約

- Route: `pf_beam`
- 親は exp202 heatmap MTP candidate generator、exp208 dense window generation、exp212 full-grid contract、比較は exp099 PF/Beam candidate cache。
- hengck23 `cnn-mtp-example` の `path_pred [K,L]` + `path_logit [K]` + closest-mode loss の発想を使う。
- `run_train_sdf.py` 型の SDF output head、SDF target、`sdf_loss` は使わない。
- ただし exp202 由来の 5ch 入力のうち `TVT_input` prefix history SDF channel は入力表現として使う。
- Kaggle GPU train は 1 active spec x 5 folds = 5 CNN models。LightGBM は 0 configs / 0 boosters。parent/control retraining はなし。
- direct TVT replacement、softmax average submission、PF weight replacement、postprocess blend、hidden-test inference、submit はしない。
- valid true TVT は fold-safe loss/eval にのみ使い、candidate score、aggregation weight、path_prob、path_logit、coverage/fallback flag、selector-facing features に oracle best / abs error / within10 / true-error rank を漏らさない。
- 再現性: `docs/06_reproducibility.md` に従い、GPU 学習の非 bitwise 性、stable loader seed、SHA 記録方針、Kaggle bootstrap 確認方針を記録する。

## 受け入れ基準

- `docs/legacy/steering/20260707-exp215-mtp-full-tail-heatmap-path-generator-probe/` に要件、設計、tasklist がある。
- `experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/` に config、train/inference notebook source、notebook、README、SESSION_NOTES、result、metrics がある。
- train notebook source は `path_pred [B,K,L]` と `path_logit [B,K]` を出す continuous MTP model、closest-mode path loss、dense full-tail generation、full-grid aggregation、candidate-union readout を含む。
- `config.yaml` に `experiment.route: pf_beam`、MTP loss/architecture/path_generation/full_grid/candidate_union/reproducibility が明記されている。
- full-grid output は少なくとも `well,id,row_index,md_since,path_rank,tvt_pred,path_logit,path_prob,weighted_tvt_pred,source_window_count,coverage_flag,fallback_flag,candidate_cost` を含む。
- success gate として source coverage >= 0.95、fallback unique row rate <= 0.05、rank1/weighted path が exp212 stitched-only top5 RMSE 50.085237573 を明確に下回ること、existing+learned topK union が existing union を改善することを記録する。
- `py_compile`、`ruff --select F821`、Jupytext conversion/test、`make validate-exp` が通る。
- Kaggle train push 前の予定コストとして 1 active variant、5 folds、5 CNN models、0 LightGBM configs、0 boosters、control retraining なしを `SESSION_NOTES.md` に記録している。
