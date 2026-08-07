# exp148_learned_likelihood_fulltrain_addonly_on_exp092 セッションノート

## 目的

exp145 の full-train/raw-test learned likelihood feature cache を使い、exp127 で支持された learned likelihood add-only feature を exp092 full-row surface で再評価する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle train v1 / inference v7 完了、CPU runtime inference v1 ref `54183122` を現行 ML route submitted anchor として記録済み
- CV: `lgb_mean` pooled RMSE 8.50128118189582
- Public LB: 7.921（exp148 CPU runtime inference v1; GPU inference v7 は 7.960）
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- feature 親: `exp145_learned_likelihood_rawtest_feature_generator_parity`

## 実装メモ

- exp127 をベースに、learned likelihood feature source を exp112 155-well subset から exp145 full-train cache に変更した。
- train notebook は exp072/exp092 train surface と exp145 full-train `ml_features` の coverage を確認し、add-only variant だけを学習する。比較は保存済み exp092 metrics を historical baseline として扱う。
- inference notebook は exp148 saved booster manifest を使い、current test frame から learned likelihood features を生成して `learned_likelihood_confidence_addonly` の `submission.csv` を生成できる。
- ただし direct submit 判断は full-row OOF、worst-well regression、feature parity、予測範囲の確認後に行う。

## Kaggle train push 前ガード

- active variants: 1
  - `learned_likelihood_confidence_addonly`
- disabled variants:
  - `exp092_fulltrain_control`。2026-06-27 ユーザー指示により再学習しない。
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- active modes: 1 (`gpu_repro_guard_dp_threads8`)
- 合計 booster: 15
- control 再学習: なし。exp092 historical metrics を比較基準として参照する。

## Kaggle 実行記録

- Train: `kentookumura/exp148-train` v1、`COMPLETE`
- Inference: `kentookumura/exp148-inference` v7、`COMPLETE`
- v1 inference は `kentookumura/exp148-train` source がまだ有効でなく失敗した。
- v2/v3 inference は `public_notebook_replay_audit` import 不足で失敗した。
- v4 inference は import 修正後に model manifest source 不足で失敗した。
- v5 inference は `public_notebook_replay_audit.py` 同梱と `exp148-train` source 復帰後に完了した。
- Code submission rerun は hidden test が public test と異なるため、`exp145-inference` の public raw-test cache 依存で `Notebook Threw Exception` になった。
- v6 は module 側を dynamic generation 対応にしたが、notebook の入力確認セルが public raw-test cache を必須にしていたため失敗した。
- v7 は notebook 側も current-test learned likelihood feature generation に変更し、public run は完了した。

## Train 結果

- rows: 3,783,989
- wells: 773
- features: 294
- variant: `learned_likelihood_confidence_addonly`
- mode: `gpu_repro_guard_dp_threads8`
- trained boosters: 15
- feature join coverage: pass
- dropped base rows: 0
- dropped base wells: 0

| model | pooled RMSE |
|---|---:|
| `lgb0` | 8.59978585937889 |
| `lgb1` | 8.563971121229669 |
| `lgb2` | 8.509819718794075 |
| `lgb_mean` | 8.50128118189582 |

Historical exp092 baseline は control 再学習なしで参照する。記録済み exp092 `lgb1` CV 9.322479895503927 に対して、exp148 `lgb_mean` は -0.821198713608107 改善した。ただし同一 notebook / 同一 runtime の control ablation ではない。

## CPU runtime 評価

- Kernel: `kentookumura/exp148-cpu-runtime-train` v1
- Status: `COMPLETE`
- 実行日: 2026-06-30
- 目的: GPU 版 exp148 train と同じ add-only variant / 3 LightGBM config / 5 fold / 15 booster を CPU deterministic mode で実行し、実行時間を評価する。
- 既存 notebook は上書きせず、`/tmp/exp148-cpu-runtime-train.VeJlM8` に作成した Kaggle push 用コピーだけを変更した。
- Kaggle metadata: `enable_gpu=false`, `machine_shape=None`
- active variants: 1 (`learned_likelihood_confidence_addonly`)
- active modes: 1 (`cpu_deterministic_threads8`)
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- trained boosters: 15
- feature join coverage: pass
- dropped base rows: 0
- dropped base wells: 0
- elapsed_seconds: 35,259.658（約 9 時間 47 分 40 秒）

| model | pooled RMSE |
|---|---:|
| `lgb0` | 8.599517400738064 |
| `lgb1` | 8.595981522994983 |
| `lgb2` | 8.557131743373821 |
| `lgb_mean` | 8.52869811419221 |

Fold 完了ログは各 booster/fold の完了時に JSON で出力された。例: `{"best_iteration": 4357, "fold": 0, "mode": "cpu_deterministic_threads8", "model": "lgb0", "rmse_tvt": 9.134171177743546, "variant": "learned_likelihood_confidence_addonly"}`。

GPU 版 train v1 の `lgb_mean` pooled RMSE 8.50128118189582 に対して、CPU 版は 8.52869811419221（+0.02741693229639）。CPU は runtime 評価用であり、提出 anchor は GPU 版 v1 / inference v7 のままとする。

## CPU inference 作成

- Kernel: `kentookumura/exp148-cpu-runtime-inference` v1
- URL: https://www.kaggle.com/code/kentookumura/exp148-cpu-runtime-inference
- 作成日: 2026-06-30
- 目的: CPU runtime train v1 の saved booster manifest を使い、CPU inference 用 notebook を別 kernel として作成する。
- 既存 inference notebook は上書きせず、`/tmp/exp148-cpu-runtime-inference.MMHgb1` に作成した Kaggle push 用コピーだけを変更した。
- Kaggle metadata: `enable_gpu=false`, `machine_shape=None`
- train source: `kentookumura/exp148-cpu-runtime-train`
- selected mode: `cpu_deterministic_threads8`
- selected variant/model: `learned_likelihood_confidence_addonly` / `lgb_mean`
- Status: `COMPLETE`
- elapsed_seconds: 154.953（約 2 分 35 秒）
- loaded models: 15
- test rows: 14,151
- submission rows: 14,151
- fallback rows: 0
- prediction min: 11590.5625
- prediction max: 12240.1279296875
- prediction mean: 11905.38429442376
- prediction std: 278.7962785700215
- prediction SHA256: `681412ec547dfdccc555045170b59f51221440086b4109d10f9a56e980e0a8db`
- submission SHA256: `3cc51eab422b9e7b83864e601733136523247a1198e67caba3264dd10ea64fa5`
- current-test learned likelihood feature decompressed SHA256: `61a21bb1b52eb8ae2d242c758732fe3cb10682d9d8b147ebe4a40f75419704c8`
- submit-check: PASS against `data/raw/sample_submission.csv` for `/tmp/kaggle-output/exp148_cpu_runtime_inference_v1/submission.csv`
- CPU inference は当初 runtime / parity 確認用として扱っていたが、後日ユーザー確認により code submission ref `54183122` が exp148 CPU runtime inference に紐づくと判断した。Kaggle submissions table でも ref `54183122` は 2026-06-29 23:35:57.090000 UTC、`SubmissionStatus.COMPLETE`、Public LB 7.921。これは GPU inference v7 ref `54124882` の Public LB 7.960 より -0.039 良く、exp193 ref `54347471` の 7.946 より -0.025 良い。したがって現行の ML route submitted anchor は exp148 CPU runtime inference v1 / Public LB 7.921 とする。

## Inference 結果

- selected model: `lgb_mean`
- loaded models: 15
- test rows: 14,151
- submission rows: 14,151
- fallback rows: 0
- prediction min: 11590.2021484375
- prediction max: 12240.267578125
- prediction mean: 11905.555289061396
- prediction std: 278.8927386018049
- prediction SHA256: `9a5f5d1030c357d8059c3c9ee2ba3a0578563ce11b9d02fe07906aa8b235d50b`
- submission SHA256: `45a8b1787fd80213c158d9af04fb596750d8025802d1328ab9d075432bcb6e4b`
- current-test learned likelihood feature decompressed SHA256: `8d1146ac1e68da67a2c8d2d00788c1593fc99654b949e0a5ac065cf781344e13`
- submit-check: PASS against `data/raw/sample_submission.csv` for `/tmp/kaggle-output/exp148_inference_v7/submission.csv`

## 解釈と次アクション

- Full-row CV は大きく改善し、exp127 subset / exp144 hidden-like stress で見えた learned likelihood signal は exp092 full train surface でも強く残った。
- v7 は hidden rerun 用に public raw-test cache 依存を外したため、code submission rerun の `Notebook Threw Exception` 対策済み。
- 提出 ref `54124882` は `SubmissionStatus.COMPLETE`、Public LB 7.960。exp092 Public LB 8.350 から -0.390 改善したため、当時の ML route submitted anchor を exp148 に更新した。
- 後日、ユーザー確認により CPU runtime inference v1 ref `54183122` を exp148 CPU runtime として扱う。Public LB 7.921 で GPU inference v7 7.960 と exp193 7.946 を上回るため、現行の ML route submitted anchor は exp148 CPU runtime inference v1 とする。
- Control 再学習なしの historical 比較だが、ユーザー判断により追加 trust audit は不要とし、anchor 更新を優先する。

## 提出記録

- Submitted at: 2026-06-28 01:39:08.267000
- Ref: `54124882`
- Status: `SubmissionStatus.COMPLETE`
- Public LB: 7.960
- Private LB: 未公開
- Kernel: `kentookumura/exp148-inference` v7
- Submission SHA256: `45a8b1787fd80213c158d9af04fb596750d8025802d1328ab9d075432bcb6e4b`

CPU runtime inference v1:

- Submitted at: 2026-06-29 23:35:57.090000
- Ref: `54183122`
- Status: `SubmissionStatus.COMPLETE`
- Public LB: 7.921
- Private LB: 未公開
- Kernel: `kentookumura/exp148-cpu-runtime-inference` v1
- Submission SHA256: `3cc51eab422b9e7b83864e601733136523247a1198e67caba3264dd10ea64fa5`

## Jupytext workflow 試行

- 実行日: 2026-06-30
- 目的: notebook を直接編集する代わりに、Jupytext percent 形式の `.py` を正の編集対象にして `.ipynb` を生成できるかを exp148 で確認する。
- 追加した paired source:
  - `exp148_learned_likelihood_fulltrain_addonly_on_exp092_train.py`
  - `exp148_learned_likelihood_fulltrain_addonly_on_exp092_inference.py`
- 追加した self-contained 試行版:
  - `exp148_learned_likelihood_fulltrain_addonly_on_exp092_selfcontained_train.py`
  - `exp148_learned_likelihood_fulltrain_addonly_on_exp092_selfcontained_train.ipynb`
  - `exp148_learned_likelihood_fulltrain_addonly_on_exp092_selfcontained_inference.py`
  - `exp148_learned_likelihood_fulltrain_addonly_on_exp092_selfcontained_inference.ipynb`
- 生成・同期コマンド:
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to py:percent experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/exp148_learned_likelihood_fulltrain_addonly_on_exp092_train.ipynb`
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to py:percent experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/exp148_learned_likelihood_fulltrain_addonly_on_exp092_inference.ipynb`
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --set-formats ipynb,py:percent experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/exp148_learned_likelihood_fulltrain_addonly_on_exp092_train.ipynb`
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --set-formats ipynb,py:percent experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/exp148_learned_likelihood_fulltrain_addonly_on_exp092_inference.ipynb`
- 確認結果:
  - train: 10 cells、inference: 5 cells の cell type / source は `.py -> .ipynb` 往復で一致。
  - 既存 notebook には cell id がなかったが、Jupytext 生成後の `.ipynb` には cell id が付与された。
  - `metadata.jupytext.formats = "ipynb,py:percent"` を付け、以後は `jupytext --sync` の対象にできる。
  - `scripts/validate_experiment.py --experiment exp148_learned_likelihood_fulltrain_addonly_on_exp092` は strict pass。
  - `jupytext --to ipynb --test` は train / inference とも pass。
- self-contained 版の確認結果:
  - 既存の同名 train / inference `.ipynb` は元の薄い notebook として維持し、上書きしない。
  - self-contained 版は別名 `_selfcontained_train` / `_selfcontained_inference` として作成した。
  - `.py` 側は Jupytext percent 形式の `# %%` / `# %% [markdown]` でセル分割している。
  - self-contained 版は `settings.py`、`learned_likelihood_fulltrain_addonly_on_exp092.py`、`learned_likelihood_rawtest_feature_generator_parity.py`、`pf_multi_observation_likelihood_probe.py`、`public_notebook_replay_audit.py` のローカル import を使わず、必要コードを notebook 内へ展開した。
  - `ast.parse` による構文チェックと `jupytext --to ipynb --test` は self-contained train / inference とも pass。
- 2026-06-30 追加構造化:
  - self-contained 版の大きな helper cell を、`Runtime and configuration helpers`、`Feature engineering helpers`、`Candidate observation features`、`Learned likelihood feature engineering`、`Model and inference utilities`、実行 orchestration に分割した。
  - train self-contained notebook は 24 cells（code 13 / markdown 11）、inference self-contained notebook は 20 cells（code 11 / markdown 9）。
  - inlined CLI helper の `if __name__ == "__main__": main()` は notebook 実行時に誤実行されるため、self-contained 版ではコメントアウトした。
  - AST import scan で self-contained train / inference ともローカル helper import がないことを確認した。
  - `jupytext --to ipynb --test` は構造化後も train / inference とも pass。
- 2026-06-30 compact self-contained 版:
  - モジュール丸ごと展開ではなく、AST で実際に呼ばれる定義を追跡して必要な関数・定数だけを抽出した。
  - 追加ファイル:
    - `exp148_learned_likelihood_fulltrain_addonly_on_exp092_compact_selfcontained_train.py`
    - `exp148_learned_likelihood_fulltrain_addonly_on_exp092_compact_selfcontained_train.ipynb`
    - `exp148_learned_likelihood_fulltrain_addonly_on_exp092_compact_selfcontained_inference.py`
    - `exp148_learned_likelihood_fulltrain_addonly_on_exp092_compact_selfcontained_inference.ipynb`
  - train compact は 1,702 lines / 18 cells（code 8 / markdown 10）、inference compact は 2,733 lines / 18 cells（code 8 / markdown 10）。
  - 旧 self-contained 版は train 5,651 lines / inference 5,568 lines だったため、train は約 70% 削減、inference は約 51% 削減。
  - train compact は exp148 train path に必要な `settings.py` 相当と `learned_likelihood_fulltrain_addonly_on_exp092.py` の train-side reachable definitions だけを含む。
  - inference compact は saved-booster inference と current-test learned likelihood feature generation に必要な reachable definitions だけを含む。raw-test replay で必要な public replay / multi-observation helpers は部分抽出した。
  - AST import scan で compact train / inference ともローカル helper import がないことを確認した。
  - `ruff --select F821`、`jupytext --to ipynb --test`、`scripts/validate_experiment.py --experiment exp148_learned_likelihood_fulltrain_addonly_on_exp092` は pass。
  - 正規の `exp148_learned_likelihood_fulltrain_addonly_on_exp092_train.ipynb` / `..._inference.ipynb` は上書きせず維持した。
- 運用メモ:
  - `.venv/bin/jupytext` はそのままだと Jupyter signature store を `~/.local/share/jupyter` に作ろうとして sandbox に当たるため、ローカルでは `JUPYTER_DATA_DIR=/tmp/jupyter-data` を付ける。
  - 現行 template validation は `<exp>_train.ipynb` と `<exp>_inference.ipynb` の 2 つを必須にしている。最終成果物を 1 notebook に寄せる場合は、train/inference 分離を保つ現行 Kaggle push flow との整合を別途決める必要がある。
  - 今後の試行では既存の同名 `.ipynb` は直接上書きせず、まず `_selfcontained_*.py` から別名 `.ipynb` を生成して確認する。
