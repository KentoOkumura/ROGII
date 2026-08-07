# exp098_selector_rank_slot_features_on_exp073 セッションノート

## 目的

exp073 deterministic ML anchor の full replay LightGBM surface に、公開 selector / PF/Beam 候補の rank1/rank2/rank3 を structured features として追加する。候補値を直接選択または平均せず、delta、source identity、score gap、U-space projection residual/disagreement として監査する。

## 現在の状態

- Route: ml_model
- 状態: submitted_complete_public_lb_8_441
- CV: best `lgb1` pooled RMSE 9.358151052
- LB: Public LB 8.441
- inference: Kaggle inference v1 complete / submit-check PASS / submitted

## コマンドログ

### 実行済み

```bash
make new-steering EXP=exp098_selector_rank_slot_features_on_exp073
make new-exp EXP=exp098_selector_rank_slot_features_on_exp073
.venv/bin/python -m py_compile experiments/exp098_selector_rank_slot_features_on_exp073/selector_rank_slot_features_on_exp073.py
.venv/bin/python -m json.tool experiments/exp098_selector_rank_slot_features_on_exp073/exp098_selector_rank_slot_features_on_exp073_train.ipynb
.venv/bin/python -m json.tool experiments/exp098_selector_rank_slot_features_on_exp073/exp098_selector_rank_slot_features_on_exp073_inference.ipynb
make validate-exp EXP=exp098_selector_rank_slot_features_on_exp073
make prepare-kaggle-notebooks EXP=exp098_selector_rank_slot_features_on_exp073 EXTRA_ARGS="--notebook train --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp098_selector_rank_slot_features_on_exp073 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp098-selector-rank-slot-features-on-exp073-train --title 'exp098 selector rank slot features on exp073 train' --run-on-push --strict"
make push-kaggle-train EXP=exp098_selector_rank_slot_features_on_exp073
kaggle kernels pull kentookumura/exp098-selector-rank-slot-features-on-exp073-train -p /tmp/kaggle-pull/exp098-selector-rank-slot-features-on-exp073-train -m
kaggle kernels logs kentookumura/exp098-selector-rank-slot-features-on-exp073-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp098-selector-rank-slot-features-on-exp073-train
kaggle kernels output kentookumura/exp098-selector-rank-slot-features-on-exp073-train -p experiments/exp098_selector_rank_slot_features_on_exp073/kaggle/output/train_v1
kaggle kernels status kentookumura/exp098-selector-rank-slot-features-on-exp073-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp098-selector-rank-slot-features-on-exp073-train
make prepare-kaggle-notebooks EXP=exp098_selector_rank_slot_features_on_exp073 EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp098-selector-rank-slot-features-on-exp073-infer --title 'exp098 selector rank slot features on exp073 infer' --run-on-push --strict"
make push-kaggle-infer EXP=exp098_selector_rank_slot_features_on_exp073
kaggle kernels pull kentookumura/exp098-selector-rank-slot-features-on-exp073-infer -p /tmp/kaggle-pull/exp098-infer -m
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp098-selector-rank-slot-features-on-exp073-infer
kaggle kernels output kentookumura/exp098-selector-rank-slot-features-on-exp073-infer -p experiments/exp098_selector_rank_slot_features_on_exp073/kaggle/output/inference_v1
make submit-check EXP=exp098_selector_rank_slot_features_on_exp073 SUBMISSION=experiments/exp098_selector_rank_slot_features_on_exp073/kaggle/output/inference_v1/submission.csv
kaggle competitions submissions rogii-wellbore-geology-prediction
```

実装内容:

- `.steering/20260621-exp098-selector-rank-slot-features-on-exp073/` を作成し、requirements / design / tasklist を記入した。
- `config.yaml` を exp073 派生の rank-slot feature ablation 用に更新した。
- `selector_rank_slot_features_on_exp073.py` を追加し、exp072 cache 読み込み、rank slot feature generation、LightGBM GroupKFold ablation、metrics / feature importance / model manifest 保存を実装した。
- train notebook を Kaggle train 用の読みやすいセル構成に更新した。
- inference notebook は未選択として明示的に停止するよう更新した。
- `validate-exp` は strict で通過した。
- Kaggle train package は `experiments/exp098_selector_rank_slot_features_on_exp073/kaggle/train` に生成済み。
- `kernel-metadata.json` は GPU 有効、internet 無効、kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`、run_on_push true。

2026-06-21 追記:

- ユーザー確認により、`control_base`、`rank_slot_delta_only`、`rank_slot_identity_score` の個別学習を外し、`rank_slot_u_disagreement` のみを学習する設定に変更した。
- `rank_slot_u_disagreement` は smaller pattern を完全に包含する。
- 変更後に `py_compile`、`make validate-exp`、`make prepare-kaggle-notebooks ... --strict` を再実行し、通過した。
- 生成済み Kaggle train package の `config.yaml` も単一 variant になっていることを確認した。
- 初回 push は Kaggle SaveKernel 400 `Your kernel title does not resolve to the specified id` で失敗したため、kernel id に解決される短い title `exp098 selector rank slot features on exp073 train` で package を再生成した。同じ canonical kernel id を維持した。
- 再 push は成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp098-selector-rank-slot-features-on-exp073-train
- `kaggle kernels pull ... -m` は成功し、metadata / source の存在確認済み。
- 通常 logs、3分 follow、5分 follow はいずれも空。Kaggle CLI の session log がまだ返っていない状態として扱い、別 slug への再 push はしていない。
- `kaggle kernels status kentookumura/exp098-selector-rank-slot-features-on-exp073-train` は `KernelWorkerStatus.RUNNING`。output はまだ空。
- ユーザーから完了連絡後に再確認し、status は `KernelWorkerStatus.COMPLETE`。
- logs と output を取得。output は `experiments/exp098_selector_rank_slot_features_on_exp073/kaggle/output/train_v1`。
- Kaggle train v1 は `train_completed`。runtime は 14,959.909 sec。
- pooled OOF: `lgb1` 9.358151052、`lgb2` 9.366698537、`lgb_mean` 9.427447987、`lgb0` 9.732275226。
- best `lgb1` は exp073 raw anchor 9.526374749 から -0.168223697、exp077 policy 9.470514801 から -0.112363749 改善。ただし exp092 best `lgb1` 9.322479896 より +0.035671157 悪い。
- `lgb_mean` は exp073 / exp077 より改善するが、exp092 `lgb_mean` 9.343064066 より悪い。
- Rank1 source distribution は `pf_ancc` 33.65%、`beam_mean` 24.55%、`likpf_mean` 41.80%。`sc_ens` / `hyb` はほぼ選ばれない。
- 特徴量重要度 CSV と上位特徴量 plot は保存済み。上位に rank-slot U-space shape 系が入った。

2026-06-21 inference 追記:

- ユーザー依頼により、exp098 の inference port を実装して Kaggle で実行した。
- `public_notebook_replay_audit.py` を同実験に同梱し、raw test full replay base features と likelihood-PF replay features を Kaggle 上で再生成する構成にした。
- inference kernel: `kentookumura/exp098-selector-rank-slot-features-on-exp073-infer` v1。
- status: `KernelWorkerStatus.COMPLETE`。
- output: `experiments/exp098_selector_rank_slot_features_on_exp073/kaggle/output/inference_v1`。
- 使用 model: `rank_slot_u_disagreement` / `gpu_repro_guard_dp_threads8` / `lgb1`、fold boosters 5 個。
- test rows / submission rows: 14,151 / 14,151。
- feature count: 260。
- fallback rows: 0。
- prediction min / max / mean / std: 11590.388671875 / 12240.130859375 / 11905.629384697306 / 279.315775048125。
- prediction SHA: `b39dbb2c98db1416c99e71f37dc9558283de83a58be6e6c9dbf10cba59e16c8b`。
- submission SHA: `1d32582f3f5984eeb9dd0bc5798b12cdc2e7aa863e0334691028901f0325125f`。
- `make submit-check ...` は PASS。
- 2026-06-22 にユーザーから提出完了の連絡を受け、Kaggle submissions を確認した。
- 最新の blank-description submission は `ref=53927479` Public LB 8.350、直後の `ref=53927490` Public LB 8.441。
- ユーザー訂正により、`ref=53927479` / 8.350 は exp092 に再帰属した。
- exp098 は `ref=53927490` / Public LB 8.441 として記録する。
- Public LB 8.441 は exp077 ML route anchor 8.611 を -0.170 改善し、exp073 raw 8.780 を -0.339 改善した。一方で exp092 8.350 と exp082 ensemble route anchor 7.601 には届かない。

### 未実行

なし。competition submit は完了済み。

## 変更点

- 追加候補は `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`。
- rank score は target-free な候補不確実性 / disagreement / last anchor からの距離で作る。
- variant:
  - `rank_slot_u_disagreement`
- `rank_slot_u_disagreement` は delta、identity/score、U-space projection、U-space disagreement の全 feature group を含む単一 pattern。
- fold/model ごとの特徴量重要度、平均特徴量重要度、上位特徴量プロットを保存する。
- direct selector、soft average は行わない。
- ユーザー依頼により inference 用 `submission.csv` は Kaggle output として生成し、後続で competition submit まで完了した。

## 再現性メモ

- seed policy: fixed GroupKFold seed; rank-slot feature generation has no RNG.
- stochastic components: upstream exp072 PF/Beam cache、GPU LightGBM training。
- CPU/GPU runtime: primary `gpu_repro_guard_dp_threads8`。
- Kaggle kernel id / version: `kentookumura/exp098-selector-rank-slot-features-on-exp073-train` v1 complete。
- input / feature schema SHA: source cache SHA `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`。
- feature content SHA: exp072 cache source SHA を summary に記録。
- model manifest / model SHA: manifest SHA `dbc19bd4844e187335e3c0806883ede994eaf7e119ed0be483cbbc05e8dcb33e`。
- prediction SHA: `lgb1` `6a2aaf8a085a2dccfe5c7a013f371bac66860c1300dc25900ae93325daae19ca`; `lgb_mean` `12657ee7d87dc8e1d31b3d5ab7e3818abf7bfe851b1d65cec94a3fb9538a0088`。
- inference kernel id / version: `kentookumura/exp098-selector-rank-slot-features-on-exp073-infer` v1 complete。
- inference prediction SHA: `b39dbb2c98db1416c99e71f37dc9558283de83a58be6e6c9dbf10cba59e16c8b`。
- submission SHA: `1d32582f3f5984eeb9dd0bc5798b12cdc2e7aa863e0334691028901f0325125f`。
- submission ref: `53927490` Public LB 8.441。`53927479` Public LB 8.350 は exp092。
- rerun check: 未実施。

## 次のアクション

1. exp098 は exp077 を上回る有用な rank-slot 比較基準として保持する。ただし ML route submitted anchor は user correction により exp092 8.350 に更新済み。
2. exp092 は Public LB 8.350 で提出済みだが、raw-test parity / worst-well guard は残タスクとして確認する。
3. follow-up は `compact_rank_slot_features_on_exp098` と `selector_topn_candidate_only_features` を候補にし、不要な rank-slot noise を減らす方向を優先する。
4. exp092 と exp098 はどちらも exp073/exp072 surface に依存するが、exp092 は U-projection correction/disagreement、exp098 は rank-slot candidate structure なので完全重複ではない。compact / top-n rank-slot signals を exp092 に add-only でマージすれば追加ゲインが出る可能性がある。
