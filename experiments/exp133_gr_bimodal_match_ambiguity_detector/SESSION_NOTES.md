# exp133_gr_bimodal_match_ambiguity_detector セッションノート

## 目的

`gr_bimodal_match_ambiguity_detector` を train-side diagnostic として実装する。GR score curve の二峰性、+/-15-25ft decoy、flat match を target-free に検出し、exp073 / exp092 / likPF / PF/Beam 候補の error bucket として評価する。

## 現在の状態

- Route: pf_beam
- 状態: completed_train_side_diagnostic_no_submit
- CV: reference best `pred_exp092_lgb1` RMSE 9.322479895503927
- LB: まだなし
- LightGBM 学習: なし
- 提出: なし

## コマンドログ

```bash
make new-steering EXP=exp133_gr_bimodal_match_ambiguity_detector
make new-exp EXP=exp133_gr_bimodal_match_ambiguity_detector
make validate-exp EXP=exp133_gr_bimodal_match_ambiguity_detector
make prepare-kaggle-notebooks EXP=exp133_gr_bimodal_match_ambiguity_detector EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp133-gr-bimodal-ambiguity-train --title 'exp133 gr bimodal ambiguity train' --run-on-push --strict"
make update-summary
make push-kaggle-train EXP=exp133_gr_bimodal_match_ambiguity_detector
```

予定:

```bash
make push-kaggle-train EXP=exp133_gr_bimodal_match_ambiguity_detector
```

### Kaggle train v1

- Kernel: `kentookumura/exp133-gr-bimodal-ambiguity-train`
- Version: 1
- 状態: failed
- 失敗箇所: `_build_well_ambiguity` の `shifted_topzero = score_cube[rows, best_zero[:, None], idx]`
- Error: `IndexError: shape mismatch: indexing arrays could not be broadcast together`
- 対応: row index / candidate index / shift index を `rows[:, None]`, `best_zero[:, None]`, `idx[None, :]` に揃えて修正。

予定:

```bash
make prepare-kaggle-notebooks EXP=exp133_gr_bimodal_match_ambiguity_detector EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp133-gr-bimodal-ambiguity-train --title 'exp133 gr bimodal ambiguity train' --run-on-push --strict"
make push-kaggle-train EXP=exp133_gr_bimodal_match_ambiguity_detector
```

### Kaggle train v2

- Kernel: `kentookumura/exp133-gr-bimodal-ambiguity-train`
- Version: 2
- 状態: completed
- 変更: v1 の NumPy advanced indexing shape mismatch を修正。
- rows / wells: 3,783,989 / 773
- runtime: 1783.534 sec
- best candidate: `pred_exp092_lgb1` RMSE 9.322479895503927 / MAE 5.980980396 / within10 0.822047051
- ambiguity: ambiguous rate 0.566856563、flat rate 0.432845324、mean ambiguity score 0.488638610
- feature cache: 40 columns
- feature cache decompressed SHA: `acdb77139e7f73efda4d8fe5dc29799823c4e46a687af5030e70a9dbe0b8d50a`
- output: `experiments/exp133_gr_bimodal_match_ambiguity_detector/kaggle/output/train_v2`
- conclusion: direct mode commit / midpoint / likPF midpoint blend は不採用。feature cache は低優先の exp092 add-only feature 候補としてだけ残す。

## 変更点

- `config.yaml` を PF/Beam diagnostic route に更新。
- `gr_bimodal_match_ambiguity_detector.py` を追加。
- train notebook を実装済み diagnostic entry に更新。
- inference notebook を no-submission summary に更新。
- steering docs、README、result、metrics を未実行状態で整理。
- train 用 Kaggle package を `experiments/exp133_gr_bimodal_match_ambiguity_detector/kaggle/train` に生成。
- `experiment_summary.md` に exp133 行を追加。

## 再現性メモ

- seed policy: `no_new_rng_gr_ambiguity_diagnostic`
- stochastic components: upstream exp072/073/092 artifacts のみ
- CPU/GPU runtime: CPU only
- Kaggle kernel id / version: `kentookumura/exp133-gr-bimodal-ambiguity-train` v2
- input / feature schema SHA: output summary に記録済み
- feature content SHA: raw gzip `cb0e9af9b55ba941b79c78eb3480f2d78207414edd7e62786e8485e81ed70f26`
- feature content SHA decompressed: `acdb77139e7f73efda4d8fe5dc29799823c4e46a687af5030e70a9dbe0b8d50a`
- model manifest / model SHA: なし
- prediction SHA: なし
- submission SHA: なし
- rerun check: 未実行

## 次のアクション

1. `KAGGLE_DIRECTION.md` と `experiment_summary.md` を結果で更新する。
