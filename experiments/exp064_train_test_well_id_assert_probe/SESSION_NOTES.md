# exp064_train_test_well_id_assert_probe セッションノート

## 目的

train と hidden test に同じ `well_id` が含まれるかを、Kaggle code submission の run status だけで確認する assert probe を実装する。

## 現在の状態

- Route: pf_beam
- 状態: usable、Kaggle public sample run 完了
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
python3 -m json.tool experiments/exp064_train_test_well_id_assert_probe/exp064_train_test_well_id_assert_probe_train.ipynb >/tmp/exp064_train.json
python3 -m json.tool experiments/exp064_train_test_well_id_assert_probe/exp064_train_test_well_id_assert_probe_inference.ipynb >/tmp/exp064_infer.json
python3 -m py_compile experiments/exp064_train_test_well_id_assert_probe/well_id_assert_probe.py
python3 -c "from pathlib import Path; import sys,json; sys.path.insert(0, 'experiments/exp064_train_test_well_id_assert_probe'); from well_id_assert_probe import run_assert_probe; print(json.dumps(run_assert_probe(train_dir=Path('data/raw/train'), test_dir=Path('data/raw/test'), expected_public_test_wells=['000d7d20','00bbac68','00e12e8b']), indent=2, sort_keys=True))"
make validate-exp EXP=exp064_train_test_well_id_assert_probe
make prepare-kaggle-notebooks EXP=exp064_train_test_well_id_assert_probe EXTRA_ARGS="--notebook inference --run-on-push --strict"
make record-exp EXP=exp064_train_test_well_id_assert_probe STATUS=scaffold_completed METRIC=status KEY_IDEA="train/test well_id assert status probe" NOTES="Implemented assert/status probe only; no Kaggle run yet; hidden overlap counts and ids are not observable."
kaggle kernels pull kentookumura/exp064-train-test-well-id-assert-probe-inference -p /tmp/kaggle-pull/exp064-train-test-well-id-assert-probe-inference -m
kaggle kernels push -p experiments/exp064_train_test_well_id_assert_probe/kaggle/inference
make prepare-kaggle-notebooks EXP=exp064_train_test_well_id_assert_probe EXTRA_ARGS="--notebook inference --run-on-push --strict --title 'exp064 train test well id assert probe inference'"
kaggle kernels push -p experiments/exp064_train_test_well_id_assert_probe/kaggle/inference
kaggle kernels pull kentookumura/exp064-train-test-well-id-assert-probe-inference -p /tmp/kaggle-pull/exp064-train-test-well-id-assert-probe-inference -m
kaggle kernels logs kentookumura/exp064-train-test-well-id-assert-probe-inference
timeout 120 kaggle kernels logs -f --interval 5 kentookumura/exp064-train-test-well-id-assert-probe-inference
kaggle kernels output kentookumura/exp064-train-test-well-id-assert-probe-inference -p /tmp/kaggle-output/exp064_train_test_well_id_assert_probe/inference_v1
kaggle kernels status kentookumura/exp064-train-test-well-id-assert-probe-inference
timeout 360 kaggle kernels logs -f --interval 15 kentookumura/exp064-train-test-well-id-assert-probe-inference
kaggle kernels output kentookumura/exp064-train-test-well-id-assert-probe-inference -p /tmp/kaggle-output/exp064_train_test_well_id_assert_probe/inference_v1
kaggle kernels logs kentookumura/exp064-train-test-well-id-assert-probe-inference
make submit-check SUBMISSION=/tmp/kaggle-output/exp064_train_test_well_id_assert_probe/inference_v1/submission.csv
make record-exp EXP=exp064_train_test_well_id_assert_probe STATUS=usable METRIC=status KEY_IDEA="train/test well_id assert status probe" NOTES="Kaggle inference kernel v1 completed on public sample; known public overlap allowed; placeholder submission passed submit-check; hidden code submission probe not run."
```

`make record-exp` の前に `STATUS=implemented_not_run` を試したが、record script の allowed status ではないため失敗した。未実行実装は `scaffold_completed` として記録した。

初回 push は Kaggle から `Your kernel title does not resolve to the specified id` の 400 を返された。`kernel-metadata.json` の id は `kentookumura/exp064-train-test-well-id-assert-probe-inference` だったため、title を `exp064 train test well id assert probe inference` に短縮して slug を一致させ、同じ kernel id に version 1 として push した。

push 後の pull は成功し、metadata に `id_no=122845215` が返った。通常 logs は最初空、`logs -f` polling 中に実行ログが取得できた。途中でユーザーから監視停止依頼があったため停止処理を行ったが、その直後に public sample run 完了ログが返った。

### Kaggle inference v1 結果

- Kernel: `kentookumura/exp064-train-test-well-id-assert-probe-inference`
- Version: 1
- URL: https://www.kaggle.com/code/kentookumura/exp064-train-test-well-id-assert-probe-inference
- Output: `/tmp/kaggle-output/exp064_train_test_well_id_assert_probe/inference_v1`
- Probe phase: `public_sample`
- Probe status: `public_sample_overlap_allowed`
- Public sample overlap wells: `000d7d20`, `00bbac68`, `00e12e8b`
- Train well count: 773
- Test well count: 3
- Submission: `/tmp/kaggle-output/exp064_train_test_well_id_assert_probe/inference_v1/submission.csv`
- Submit-check: PASS
- Code submission ref: `53627058`
- Code submission status: complete
- Public LB: 11551.955
- Hidden / scoring test assertion: not triggered
- Interpretation: scoring test では公開された horizontal-well ファイル名 prefix としての exact train/test `well_id` overlap は検出されなかった。placeholder zero submission のため Public LB は診断用で、モデル性能として扱わない。Kaggle が同じ物理 well を別 filename / anonymized id で公開している場合は、この probe では検出できない。

## 変更点

- `well_id_assert_probe.py` を追加し、horizontal well filename から `well_id` を抽出する。
- 公開 sample の既知 3 wells (`000d7d20`, `00bbac68`, `00e12e8b`) は overlap を許可する。
- hidden / private test と判定した場合は `train_wells & test_wells` が非空なら `AssertionError("HIDDEN_TRAIN_TEST_WELL_ID_OVERLAP_DETECTED")` を出す。
- hidden test の overlap 件数、対象行数、id、予測差分は観測不能なので記録しない。
- notebook が成功した場合だけ `submission.csv` を sample submission copy として作る。

## 次のアクション

1. train/test same exposed-`well_id` 前提の static replay / visible override は優先度を下げる。
2. 見えない新規 well 用の hidden branch、public replay integrity audit、PF confidence residual clip を優先する。
