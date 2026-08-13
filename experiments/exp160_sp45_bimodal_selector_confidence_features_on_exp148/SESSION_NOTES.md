# exp160_sp45_bimodal_selector_confidence_features_on_exp148 セッションノート

## 目的

`backlog/KAGGLE_DIRECTION.md` の `sp45_bimodal_selector_confidence_features_on_exp148` を実験化する。公開上位 notebook の SP45 / PF / Beam / bimodal selector 系 signal を、直接置換・blend・hard gate ではなく、target-free な confidence / disagreement / shape feature として exp148 に add-only する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle train v2 / inference v1 / scoring 完了。train-side OOF は exp148 baseline から小幅改善し、submission.csv は submit-check pass。Public LB は 8.061。
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- base surface: `exp092_u_projection_correction_disagreement_fullrun`
- 比較基準: exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- 提出: あり。Public LB 8.061、ref `54183128`。

## 実装メモ

- exp148 の train / inference flow をコピーし、`sp45_bimodal_selector_confidence_features_on_exp148.py` に SP45/Bimodal feature builder を追加した。
- 既存の exp148 feature surface は維持する。
  - exp072/exp092 full replay 196 features
  - U-projection correction / disagreement
  - exp145 learned likelihood confidence features
- 新規 feature group は `sp45_bimodal_selector_confidence`。
  - `sc8/sc15/sc25` score posterior / margin / entropy
  - PF / Beam / likelihood-PF / SC / hybrid / dense candidate の delta / abs delta / normalized gap
  - candidate spread、bimodal midpoint、extreme gap、closest-candidate gap
  - prefix trust / `pfx_rmse` / known/eval length / near-row / longtail indicators
  - PF-Beam、Beam-likPF、SC15-likPF、dense-likPF gap interactions
  - U-projection shape columns
- Candidate TVT raw path は保存・派生に使うが、variant では confidence feature としてのみ使う。直接 TVT replacement、late blend、postprocess hard gate は実装しない。
- inference 側も current-test replay frame から learned likelihood features と SP45/Bimodal features を生成し、train manifest の feature group と一致しない場合は fail する。

## Kaggle train push 前ガード

- active variants: 1
  - `sp45_bimodal_selector_confidence_addonly`
- disabled variants:
  - `exp148_fulltrain_control`。control 再学習はしない。exp148 の既存 CV / LB を historical baseline として参照する。
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- active modes: 1 (`gpu_repro_guard_dp_threads8`)
- 合計 booster: 15
- control 再学習: なし

## 実装確認

- 2026-06-29: `docs/legacy/steering/20260629-exp160-sp45-bimodal-selector-confidence-features-on-exp148/` を作成。
- 2026-06-29: exp148 から `experiments/exp160_sp45_bimodal_selector_confidence_features_on_exp148/` を作成。
- 2026-06-29: `sp45_bimodal_selector_confidence_features_on_exp148.py` に feature builder と train / inference wiring を実装。
- 2026-06-29: train / inference notebook を exp160 用に更新。
- 2026-06-29: `.venv/bin/python -m py_compile ...` は PASS。
- 2026-06-29: `.venv/bin/python -m json.tool ...train.ipynb` / `...inference.ipynb` は PASS。
- 2026-06-29: `.venv/bin/ruff check ...` は PASS。
- 2026-06-29: `make validate-exp EXP=exp160_sp45_bimodal_selector_confidence_features_on_exp148` は PASS。
- 2026-06-29: train package を `kentookumura/exp160-sp45-bimodal-selector-confidence-features-on-exp148-train` / title `exp160 sp45 bimodal selector confidence features on exp148 train` で prepare 済み。
- 2026-06-29: inference package を `kentookumura/exp160-sp45-bimodal-selector-confidence-features-on-exp148-inference` / title `exp160 sp45 bimodal selector confidence features on exp148 inference` で prepare 済み。
- 2026-06-29: `make push-kaggle-train EXP=exp160_sp45_bimodal_selector_confidence_features_on_exp148` は long canonical kernel id `kentookumura/exp160-sp45-bimodal-selector-confidence-features-on-exp148-train` で `SaveKernel` 400 により失敗。Kaggle API は詳細 message を返さなかった。
- 2026-06-29: `kaggle kernels pull kentookumura/exp160-sp45-bimodal-selector-confidence-features-on-exp148-train -p /tmp/kaggle-pull/exp160-sp45-bimodal-selector-confidence-features-on-exp148-train -m` は 403 で、失敗 kernel id の作成は確認できず。
- 2026-06-29: slug 長リスクを避けるため、同じ exp160 のまま train kernel id/title を `kentookumura/exp160-sp45-bimodal-exp148-train` / `exp160 sp45 bimodal exp148 train` に短縮して再 prepare/push する。
- 2026-06-29: shorter id/title で train package を再 prepare し、`make validate-exp EXP=exp160_sp45_bimodal_selector_confidence_features_on_exp148` は PASS。
- 2026-06-29: `make push-kaggle-train EXP=exp160_sp45_bimodal_selector_confidence_features_on_exp148` は成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp160-sp45-bimodal-exp148-train
- 2026-06-29: `kaggle kernels pull kentookumura/exp160-sp45-bimodal-exp148-train -p /tmp/kaggle-pull/exp160-sp45-bimodal-exp148-train -m` は成功。`id_no=125290303`、`enable_gpu=true`、`enable_internet=false`、kernel sources は exp072 / exp145。
- 2026-06-29: `timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp160-sp45-bimodal-exp148-train` は 5 分間 log 空のまま timeout。再 push はしない。
- 2026-06-29: 通常 `kaggle kernels logs kentookumura/exp160-sp45-bimodal-exp148-train` も log 空。`kaggle kernels status kentookumura/exp160-sp45-bimodal-exp148-train` は `KernelWorkerStatus.RUNNING`。
- 2026-06-29: ユーザーより v1 失敗連絡。`kaggle kernels logs kentookumura/exp160-sp45-bimodal-exp148-train` を取得し、status は `KernelWorkerStatus.ERROR`。ログでは SP45/Bimodal feature generation 中に `DataFrame is highly fragmented` warning が多数出た後、`nbclient.exceptions.DeadKernelError: Kernel died`。明示的な Python exception ではなく、メモリ断片化または OOM と判断。
- 2026-06-29: v2 修正として `build_sp45_bimodal_selector_features` を逐次 DataFrame insert から dict-of-array 一括 DataFrame 化へ変更。train/inference 側の SP45 feature join も key merge ではなく row-order guard + numeric concat に変更。config の SP45 feature set も候補数、direct columns、shape columns を削減。
- 2026-06-29: v2 修正後 `.venv/bin/ruff check ...`、`.venv/bin/python -m py_compile ...`、`make validate-exp EXP=exp160_sp45_bimodal_selector_confidence_features_on_exp148` は PASS。train package は同じ kernel id/title `kentookumura/exp160-sp45-bimodal-exp148-train` / `exp160 sp45 bimodal exp148 train` で再 prepare 済み。
- 2026-06-29: `make push-kaggle-train EXP=exp160_sp45_bimodal_selector_confidence_features_on_exp148` により同じ kernel id に version 2 を push。URL: https://www.kaggle.com/code/kentookumura/exp160-sp45-bimodal-exp148-train
- 2026-06-29: `kaggle kernels pull kentookumura/exp160-sp45-bimodal-exp148-train -p /tmp/kaggle-pull/exp160-sp45-bimodal-exp148-train-v2 -m` は成功。`id_no=125290303`、GPU、internet disabled、kernel sources exp072 / exp145 を確認。
- 2026-06-29: v2 push 後、`kaggle kernels status kentookumura/exp160-sp45-bimodal-exp148-train` は `KernelWorkerStatus.RUNNING`。`timeout 120 kaggle kernels logs -f --interval 20 ...` と通常 `kaggle kernels logs ...` は log 空。
- 2026-06-30: ユーザー完了連絡後に `kaggle kernels status kentookumura/exp160-sp45-bimodal-exp148-train` を確認し、`KernelWorkerStatus.COMPLETE`。version 2 が正式完了。
- 2026-06-30: Kaggle logs から train result を確認。3,783,989 rows / 773 wells / 372 features、feature join coverage pass、dropped rows 0、dropped wells 0、elapsed 14,573.844 sec。
- 2026-06-30: pooled RMSE は `lgb0` 8.582750400、`lgb1` 8.458535254、`lgb2` 8.502983731、`lgb_mean` 8.463718774。exp148 historical baseline `lgb_mean` 8.501281182 から -0.037562408 改善。
- 2026-06-30: 最終 summary の `active_variants` に disabled control 名も混ざる表示不整合を確認。学習ログは `sp45_bimodal_selector_confidence_addonly` のみで、実行 booster 数も予定通り 15。将来再実行用に summary 出力を enabled variant のみに修正した。
- 2026-06-30: inference package を短縮 slug `kentookumura/exp160-sp45-bimodal-exp148-inference` / title `exp160 sp45 bimodal exp148 inference` で再 prepare。生成済み metadata は train source `kentookumura/exp160-sp45-bimodal-exp148-train`、exp072 / exp099 / exp111 / exp112 sources、GPU、internet disabled。
- 2026-06-30: `make push-kaggle-infer EXP=exp160_sp45_bimodal_selector_confidence_features_on_exp148` は成功。Kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp160-sp45-bimodal-exp148-inference
- 2026-06-30: `kaggle kernels pull kentookumura/exp160-sp45-bimodal-exp148-inference -p /tmp/kaggle-pull/exp160-sp45-bimodal-exp148-inference-v1 -m` は成功。`id_no=125343487`、`machine_shape=Gpu`、`enable_internet=false`、kernel sources に短縮 train slug `kentookumura/exp160-sp45-bimodal-exp148-train` を確認。
- 2026-06-30: push 後 `kaggle kernels status kentookumura/exp160-sp45-bimodal-exp148-inference` は `KernelWorkerStatus.RUNNING`。
- 2026-06-30: ユーザー完了連絡後に `kaggle kernels status kentookumura/exp160-sp45-bimodal-exp148-inference` を確認し、`KernelWorkerStatus.COMPLETE`。
- 2026-06-30: Kaggle output を `/tmp/kaggle-output/exp160_sp45_bimodal_selector_confidence_features_on_exp148/inference_v1/` に取得。summary は `status=inference_completed`。
- 2026-06-30: inference v1 は `sp45_bimodal_selector_confidence_addonly` / `gpu_repro_guard_dp_threads8` / `lgb_mean` を選択し、15 boosters、372 features、test rows 14,151、fallback rows 0。feature generation elapsed 102.769 sec、total elapsed 141.319 sec。
- 2026-06-30: 予測統計は min 11590.324219、max 12240.247070、mean 11905.439880、std 278.695602。prediction sha256 `cb85e56ed032e3f5c0577c1928272e9f1621da9e9b356932caac20fc0d1c03d2`、submission sha256 `366543ab052b98afec8c61f020c6eccc84c751fd734262dd9913bbb53fab354b`。
- 2026-06-30: rawtest learned likelihood features は 14,151 rows / 3 wells / 51 columns、sha256 `4568f323d336bc93e9822e3ee933f9597ae423a6611ecd7c73a34a9a7387e96b`。long likelihood は 70,755 rows、sha256 `4c424769db59f133538ba94f0e8e397f15bf97c8283fe3464420a471e02041c5`。
- 2026-06-30: `make submit-check EXP=exp160_sp45_bimodal_selector_confidence_features_on_exp148 SUBMISSION=/tmp/kaggle-output/exp160_sp45_bimodal_selector_confidence_features_on_exp148/inference_v1/submission.csv` は PASS。
- 2026-06-30: `.agents/skills/kaggle-submit-check/scripts/check_submission.py ... --sample data/raw/sample_submission.csv` は PASS。重複 ID なし、empty/NaN/Inf-like なし、rows 14,151 / columns 2、header と row count は `sample_submission.csv` と一致。
- 2026-06-30: `kaggle competitions submissions rogii-wellbore-geology-prediction -v -q` で scoring 完了を確認。最新 2 件はどちらも description 空で、Kaggle CLI は kernel id を返さない。
- 2026-06-30: ref `54183128` は submitted `2026-06-29 23:36:23.280000`、status COMPLETE、Public LB 8.061。`monitor_submission.py --once` でも最新 complete を確認した。
- 2026-06-30: ユーザー確認により exp160 の Public LB は 8.061 と確定。以前 exp160 近傍として仮記録した ref `54183122` / 7.921 は exp160 から除外した。
- 2026-06-30: exp148 baseline Public LB 7.960 に対し、exp160 Public LB 8.061 は +0.101 悪化。train-side CV は positive だが LB は negative のため採用しない。

## 次アクション

1. 追加監査する場合は Kaggle output から by-well / bucket / feature importance を取得し、near `000_050`、`1000_plus`、worst-well、common PF+ML worst wells を exp148 と比較する。
