# exp082_public_artifact_replay_followup セッションノート

## 目的

`exp079_public_artifact_replay_integrity_audit` 後の follow-up として、SP45 / fle3n / Koolbox / SP45-Fleongg blend 系公開 notebook の exact source slug を固定し、外部生成物依存、static visible CSV、code competition rerun 互換、branch output、既存 anchor との差分を監査する。

## 現在の状態

- Route: ensemble
- 状態: audit_completed
- CV: なし
- LB: なし
- Submit: なし

## コマンドログ

```bash
uv run python scripts/validate_project.py
uv run python scripts/new_steering.py --experiment exp082_public_artifact_replay_followup
uv run python scripts/new_experiment.py --name exp082_public_artifact_replay_followup --source experiments/exp079_public_artifact_replay_integrity_audit
```

実装内容:

- `config.yaml` を `exp082_public_artifact_replay_followup` 用に更新。
- `runtime.kaggle.kernel_sources` に次を追加。
  - `lightningv08/lb-7-776-rogii-ridge-sp`
  - `fleongg/fle3n-rogii-v4`
  - `rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction`
  - `jaemin3404/rogii-sp45-fleongg-blend-v2`
  - `debatreyabiswas/wellboregeology-prediction-with-koolbox-best-8-188`
  - `packagemanager/pm-121774751-at-06-05-2026-09-29-28`
- `runtime.kaggle.dataset_sources` に次を追加。
  - `phongnguyn23021656/koolbox-offline`
  - `fleongg/rogii-claude-models-pub`
  - `ravaghi/wellbore-geology-prediction-artifacts`
- `audit.source_specs` に fle3n / SP45 / SP45-Fleongg blend / Koolbox best 8.188 の required source、branch files、expected checks を追加。
- `public_artifact_integrity_audit.py` の source inspection を `.ipynb` / `.py` 両対応にした。
- notebook ファイルを `exp082_public_artifact_replay_followup_train.ipynb` / `exp082_public_artifact_replay_followup_inference.ipynb` にリネームした。

## 再現性メモ

- seed policy: `no_rng_used`
- stochastic components: なし
- CPU/GPU runtime: CPU only。GPU なし。
- input SHA: Kaggle audit output の summary JSON に保存予定
- feature schema SHA: 対象外
- feature content SHA: 対象外
- model manifest / model SHA: 対象外
- prediction SHA: candidate CSV の SHA として submission summary CSV に保存予定
- submission SHA: candidate CSV の SHA として submission summary CSV に保存予定

## 実装後の検証

```bash
uv run python scripts/validate_experiment.py --experiment exp082_public_artifact_replay_followup
uv run ruff check experiments/exp082_public_artifact_replay_followup/public_artifact_integrity_audit.py experiments/exp082_public_artifact_replay_followup/settings.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp082_public_artifact_replay_followup --notebook train --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp082_public_artifact_replay_followup --notebook train --kernel-id kentookumura/exp082-artifact-followup-train --title "exp082 artifact followup train" --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp082_public_artifact_replay_followup --notebook inference --run-on-push --strict
uv run python -m py_compile experiments/exp082_public_artifact_replay_followup/public_artifact_integrity_audit.py experiments/exp082_public_artifact_replay_followup/settings.py
uv run python -c "import sys; from pathlib import Path; sys.path.insert(0, 'experiments/exp082_public_artifact_replay_followup'); from settings import load_config; from public_artifact_integrity_audit import run_integrity_audit; summary = run_integrity_audit(config=load_config(), root=Path('.').resolve(), artifacts_dir=Path('/tmp/exp082-smoke-artifacts')); print(summary['status'], len(summary['missing_required_sources']), len(summary['notebook_inspections']), len(summary['submission_summaries']))"
```

- `validate_experiment`: pass.
- `ruff check`: pass.
- `prepare_kaggle_notebooks` train / inference: pass。
- generated train metadata: kernel sources 6 件、dataset sources 3 件、GPU false、internet false。長い default slug を避けるため、正の push 候補は `kentookumura/exp082-artifact-followup-train`。
- local smoke: `/tmp/exp082-smoke-artifacts` に audit summary / CSV / JSONL / README を保存し、Kaggle source 不在により想定通り `blocked_missing_required_sources 16`。保存済み local source inspection は 4 件、candidate submission は 0 件。

## 次のアクション

Kaggle 実行:

```bash
kaggle kernels push -p experiments/exp082_public_artifact_replay_followup/kaggle/train
kaggle kernels pull kentookumura/exp082-artifact-followup-train -p /tmp/kaggle-pull/exp082-artifact-followup-train -m
kaggle kernels logs kentookumura/exp082-artifact-followup-train
kaggle kernels output kentookumura/exp082-artifact-followup-train -p /tmp/kaggle-output/exp082_public_artifact_replay_followup/train_v1
```

- push: version 1 successfully pushed.
- pull: `/tmp/kaggle-pull/exp082-artifact-followup-train` に metadata 取得成功。
- logs: `Audit status: blocked_missing_required_sources`、Missing required sources 1、Candidate submissions 19、Notebook inspections 7、Pairwise distances 153。
- output: `/tmp/kaggle-output/exp082_public_artifact_replay_followup/train_v1` に取得。
- output files:
  - `artifacts/exp082_public_artifact_replay_followup_summary.json`
  - `artifacts/exp082_public_artifact_replay_followup_submission_summary.csv`
  - `artifacts/exp082_public_artifact_replay_followup_pairwise_distances.jsonl`
  - `artifacts/exp082_public_artifact_replay_followup_README.md`
  - `metrics.json`

## 結果

- Audit status: `blocked_missing_required_sources`
- Missing required sources: 1
  - `sp45_wellbore_for_blend_prediction`: `rogii-sp45-wellbore-for-blend-prediction` -> `/kaggle/input/rogii-sp45-wellbore-for-blend-prediction`
- Source inspections: 7 (`ipynb` 5, `py` 2)
- Candidate files: 19
- Valid submission CSVs: 18
- Read errors: 1 (`sp45_fleongg_blend_report.csv` は report CSV で `id` 列なし)
- Pairwise distances: 153

Source risk hits:

- ridge-sp: `writes_submission_csv=1`, `reads_submission_csv=2`, `mentions_public_or_visible=1`
- fle3n v4: `writes_submission_csv=5`, `reads_submission_csv=4`, `hardcoded_working_submission=1`, `exact_match_or_override=1`
- jaemin SP45/Fleongg v2: `writes_submission_csv=5`, `reads_submission_csv=4`, `hardcoded_working_submission=1`, `exact_match_or_override=1`
- Koolbox best 8.188: `writes_submission_csv=1`, `reads_submission_csv=2`, `mentions_public_or_visible=1`
- package manager source: risk hits 0

代表 SHA:

- ridge-sp final: `de1766fa3037be4a53e60b8d95bb0fe83ec094d981050c6c4e315c6e4861580d`
- fle3n v4 final: `359b3e779d360ac8117a7da8040ef780905381aec160d385b72e354ef710279b`
- fle3n v4 SP45 projection: `9aa8a5d0f6ea3ef60dcad11983d40fd42884c0d9e0031956e22f40aeda7c3d0b`
- jaemin SP45/Fleongg v2 final: `d8b0af2cc9b3d7f299dd63a6cf6333918c222c6790eba8a69eab40de3e8fef45`
- jaemin SP45 projection: `ca09d625aef8e23440bdc1710d7a58282f2e0a00766e141c918cb1b314914f9d`
- Koolbox best 8.188 final: `8520111d5b7e1812c5298cdf2e18b1a1c6a59feb8bc368aa8fd33bf453075341`

Pairwise 抜粋:

- ridge-sp vs fle3n v4 final: RMSE 1.890560287
- ridge-sp vs fle3n v4 SP45 projection: RMSE 1.384232857
- ridge-sp vs jaemin SP45/Fleongg final: RMSE 1.949888855
- ridge-sp vs jaemin SP45 projection: RMSE 1.413346840
- ridge-sp vs Koolbox best 8.188: RMSE 1.551098293
- fle3n v4 final vs jaemin SP45/Fleongg final: RMSE 0.275729904
- fle3n v4 final vs Koolbox best 8.188: RMSE 1.546923624
- jaemin SP45/Fleongg final vs Koolbox best 8.188: RMSE 1.486587152

## 次のアクション

Addability 確認:

```bash
kaggle kernels pull rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction -p /tmp/kaggle-pull/rauffauzanrambe-rogii-sp45-wellbore-for-blend-prediction -m
kaggle kernels output rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction -p /tmp/kaggle-output/source-check/rauffauzanrambe-rogii-sp45-wellbore-for-blend-prediction
kaggle kernels list --user rauffauzanrambe --search rogii-sp45-wellbore-for-blend-prediction
kaggle kernels push -p /tmp/kaggle-source-addability-probe
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/rauff-source-addability-probe
kaggle kernels output kentookumura/rauff-source-addability-probe -p /tmp/kaggle-output/source-check/rauff-source-addability-probe-v1
```

- `rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction` は public kernel として存在し、`kaggle kernels pull` と `kaggle kernels list` で確認できた。
- metadata: `kernel_type=notebook`, `is_private=false`, `keywords=["utility script"]`, dataset sources は `phongnguyn23021656/koolbox-offline` / `fleongg/rogii-claude-models-pub` / `ravaghi/wellbore-geology-prediction-artifacts`。
- `kaggle kernels output` では output CSV を直接取得できた。取得先は `/tmp/kaggle-output/source-check/rauffauzanrambe-rogii-sp45-wellbore-for-blend-prediction`。
- ただし単独 source addability probe `kentookumura/rauff-source-addability-probe` v1 では `/kaggle/input` が root のみ、`matches=[]`、`dir_count=1`、`file_count=0`。`kernel_sources=["rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction"]` としても Kaggle runtime に mount されなかった。
- 判定: output は直接取得可能だが、Kaggle Notebook の input source としては addable / mountable ではない。`exp082` の `blocked_missing_required_sources` は config typo ではなく Kaggle source mount 側の制約と扱う。
- 直接取得した代表 output:
  - `submission.csv`: rows 14151, SHA `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
  - `sp45_projection_submission.csv`: rows 14151, SHA `4e2bfc43b4b2202a5a9fc8808f3a42d408d86cee9c401f1eae4a1fcaa2c5edb9`
  - `fleongg_pretrained_submission.csv`: rows 14151, SHA `0be7ab1d24aee3a83e66f648b18868ef6f5bad475954de034bd569677589f619`
  - `submission_sp45_fleongg_w0.55.csv`: rows 14151, SHA `8cac8dba34a126ae384ad458f4cd299a7cab4bf769e306b321346a58599b4667`
  - `submission_sp45_fleongg_w0.60.csv`: rows 14151, SHA `fa95d6906d5aa2aa29bc99843dded981f6f366d9ea8a463a87fe8a6597250c42`

Mountable source only rerun:

```bash
uv run python scripts/validate_experiment.py --experiment exp082_public_artifact_replay_followup
uv run ruff check experiments/exp082_public_artifact_replay_followup/public_artifact_integrity_audit.py experiments/exp082_public_artifact_replay_followup/settings.py
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp082_public_artifact_replay_followup --notebook train --kernel-id kentookumura/exp082-artifact-followup-train --title "exp082 artifact followup train" --run-on-push --strict
kaggle kernels push -p experiments/exp082_public_artifact_replay_followup/kaggle/train
kaggle kernels pull kentookumura/exp082-artifact-followup-train -p /tmp/kaggle-pull/exp082-artifact-followup-train-v2 -m
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp082-artifact-followup-train
kaggle kernels output kentookumura/exp082-artifact-followup-train -p /tmp/kaggle-output/exp082_public_artifact_replay_followup/train_v2
```

- config から `rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction` を `runtime.kaggle.kernel_sources` と `audit.source_specs` から外した。
- `rauffauzanrambe` は direct-output reference として config / metrics / notes に残した。
- v2 push: version 2 successfully pushed.
- v2 log: `Audit status: audit_completed`、Missing required sources 0、Candidate submissions 19、Notebook inspections 7、Pairwise distances 153。
- v2 output: `/tmp/kaggle-output/exp082_public_artifact_replay_followup/train_v2` に取得。
- v2 source specs: `ridge_sp_lb_7776`, `fle3n_rogii_v4`, `sp45_fleongg_blend_v2`, `koolbox_best_8188`。

## 次のアクション

SP45 projection submit-check / row-level guard:

```bash
kaggle kernels output fleongg/fle3n-rogii-v4 -p /tmp/kaggle-output/source-check/fleongg-fle3n-rogii-v4
kaggle kernels output jaemin3404/rogii-sp45-fleongg-blend-v2 -p /tmp/kaggle-output/source-check/jaemin3404-rogii-sp45-fleongg-blend-v2
kaggle kernels output lightningv08/lb-7-776-rogii-ridge-sp -p /tmp/kaggle-output/source-check/lightningv08-lb-7-776-rogii-ridge-sp
kaggle kernels output pilkwang/rogii-target-free-tvt-geosteering -p /tmp/kaggle-output/source-check/pilkwang-rogii-target-free-tvt-geosteering
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/source-check/fleongg-fle3n-rogii-v4/sp45_projection_submission.csv
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/source-check/jaemin3404-rogii-sp45-fleongg-blend-v2/sp45_projection_submission.csv
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/source-check/rauffauzanrambe-rogii-sp45-wellbore-for-blend-prediction/sp45_projection_submission.csv
uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/source-check/fleongg-fle3n-rogii-v4/sp45_projection_submission.csv --sample data/raw/sample_submission.csv
uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/source-check/jaemin3404-rogii-sp45-fleongg-blend-v2/sp45_projection_submission.csv --sample data/raw/sample_submission.csv
uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/source-check/rauffauzanrambe-rogii-sp45-wellbore-for-blend-prediction/sp45_projection_submission.csv --sample data/raw/sample_submission.csv
uv run python experiments/exp082_public_artifact_replay_followup/sp45_projection_candidate_guard.py
uv run ruff check experiments/exp082_public_artifact_replay_followup/sp45_projection_candidate_guard.py
```

- `fleongg/fle3n-rogii-v4`、`jaemin3404/rogii-sp45-fleongg-blend-v2`、`lightningv08/lb-7-776-rogii-ridge-sp`、`pilkwang/rogii-target-free-tvt-geosteering` の output を取得した。`rauffauzanrambe` output は addability 確認時の取得済みファイルを使用した。
- 3 つの SP45 projection 候補は `scripts/validate_submission.py` と `.agents/skills/kaggle-submit-check/scripts/check_submission.py --sample data/raw/sample_submission.csv` の両方で pass。FAIL/WARN なし、rows 14151、columns 2、header / row count は sample と一致、重複 ID なし、NaN/Inf 相当値なし。
- guard output:
  - `artifacts/sp45_projection_candidate_guard_summary.json`
  - `artifacts/sp45_projection_candidate_guard_submission_summary.csv`
  - `artifacts/sp45_projection_candidate_guard_pairwise.csv`
  - `artifacts/sp45_projection_candidate_guard_README.md`
- 候補 SHA:
  - fle3n SP45 projection: `9aa8a5d0f6ea3ef60dcad11983d40fd42884c0d9e0031956e22f40aeda7c3d0b`
  - jaemin SP45 projection: `ca09d625aef8e23440bdc1710d7a58282f2e0a00766e141c918cb1b314914f9d`
  - rauff direct-output SP45 projection: `4e2bfc43b4b2202a5a9fc8808f3a42d408d86cee9c401f1eae4a1fcaa2c5edb9`
- anchor distance:
  - fle3n vs ridge-sp: RMSE `1.384232857`、p95 abs `2.336443468`
  - jaemin vs ridge-sp: RMSE `1.413346840`、p95 abs `2.710165004`
  - rauff direct output vs ridge-sp: RMSE `1.303650505`、p95 abs `2.495747798`
  - fle3n vs Pilkwang raw projection: RMSE `1.256802613`、p95 abs `2.165704547`
  - jaemin vs Pilkwang raw projection: RMSE `1.192134839`、p95 abs `2.095592234`
  - rauff direct output vs Pilkwang raw projection: RMSE `1.208265533`、p95 abs `2.123860901`
- candidate-to-candidate distance:
  - fle3n vs jaemin: RMSE `0.324981626`、p95 abs `0.618058119`、max abs `0.733854728`、abs > 1 count `0`
  - fle3n vs rauff direct output: RMSE `0.325712149`、p95 abs `0.655126366`、max abs `1.564184415`、abs > 1 count `334`
  - jaemin vs rauff direct output: RMSE `0.230330314`、p95 abs `0.559615205`、max abs `1.219190593`、abs > 1 count `10`

## 次のアクション

1. submit するなら mountable / code-submit 再現可能な候補を 1 件だけ選ぶ。保守的には ridge-sp からの drift が小さい `fle3n` SP45 projection を優先し、Pilkwang raw projection への近さを重視するなら `jaemin` SP45 projection を候補にする。
2. `rauffauzanrambe` は ridge-sp には最も近いが Kaggle source として mount できないため、direct-output reference に留め、code-submit 再現候補にはしない。
3. この時点では直接 submit はまだ実行していない。

## Submit 実行

2026-06-20 JST に、第一候補の `fle3n` SP45 projection を提出した。

事前チェック:

```bash
uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/source-check/fleongg-fle3n-rogii-v4/sp45_projection_submission.csv --sample data/raw/sample_submission.csv
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/source-check/fleongg-fle3n-rogii-v4/sp45_projection_submission.csv
sha256sum /tmp/kaggle-output/source-check/fleongg-fle3n-rogii-v4/sp45_projection_submission.csv
```

- submit-check: PASS、FAIL/WARN なし。
- validation: PASS。
- SHA: `9aa8a5d0f6ea3ef60dcad11983d40fd42884c0d9e0031956e22f40aeda7c3d0b`。

CSV 直接提出:

```bash
kaggle competitions submit rogii-wellbore-geology-prediction -f /tmp/kaggle-output/source-check/fleongg-fle3n-rogii-v4/sp45_projection_submission.csv -m "exp082 fle3n SP45 projection guard-selected sha=9aa8a5d0"
kaggle competitions submit rogii-wellbore-geology-prediction -f /tmp/kaggle-submit/exp082-fle3n-sp45/submission.csv -m "exp082 fle3n SP45 projection guard selected sha 9aa8a5d0"
```

- どちらも upload 後の `CreateSubmission` で 400。
- API response body: `Submission not allowed: This competition only accepts Submissions from Notebooks.`
- 対応: exp082 inference notebook を submit 用に変更し、Kaggle Notebook version から提出した。

Kaggle inference:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp082_public_artifact_replay_followup --notebook inference --kernel-id kentookumura/exp082-fle3n-sp45-infer --title "exp082 fle3n sp45 infer" --run-on-push --strict
kaggle kernels push -p experiments/exp082_public_artifact_replay_followup/kaggle/inference
kaggle kernels pull kentookumura/exp082-fle3n-sp45-infer -p /tmp/kaggle-pull/exp082-fle3n-sp45-infer-v2 -m
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp082-fle3n-sp45-infer
kaggle kernels output kentookumura/exp082-fle3n-sp45-infer -p /tmp/kaggle-output/exp082_public_artifact_replay_followup/inference_v2
uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp082_public_artifact_replay_followup/inference_v2/submission.csv --sample data/raw/sample_submission.csv
```

- v1 は固定パス `/kaggle/input/fle3n-rogii-v4/sp45_projection_submission.csv` が存在せず失敗。
- v2 では `/kaggle/input` 配下から `sp45_projection_submission.csv` を検索し、SHA `9aa8a5d0...` に一致する候補を選択するよう修正。
- v2 log:
  - `Audit status: audit_completed`
  - `Candidate submissions: 19`
  - selected path: `/kaggle/input/notebooks/fleongg/fle3n-rogii-v4/sp45_projection_submission.csv`
  - `Submission created: /kaggle/working/submission.csv`
  - rows `14151`
  - SHA `9aa8a5d0f6ea3ef60dcad11983d40fd42884c0d9e0031956e22f40aeda7c3d0b`
  - range `[11589.393529812714, 12239.252531477006]`
- output: `/tmp/kaggle-output/exp082_public_artifact_replay_followup/inference_v2/submission.csv`
- output submit-check: PASS、FAIL/WARN なし。

Notebook submission:

```bash
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp082-fle3n-sp45-infer -v 2 -f submission.csv -m "exp082 fle3n SP45 projection notebook v2 sha 9aa8a5d0"
kaggle competitions submissions rogii-wellbore-geology-prediction
```

- ref: `53853237`
- status: `SubmissionStatus.COMPLETE`
- Public LB: null
- raw API errorDescription: `Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset`
- description: `exp082 fle3n SP45 projection notebook v2 sha 9aa8a5d0`
- 判定: failed submission。通常 commit run では public notebook output を mount して `submission.csv` を作れたが、提出時の hidden rerun では public-output-copy 前提が成立しない。

## 次のアクション

1. ref `53853237` は hidden rerun error として失敗扱いにする。
2. 再提出するなら public output CSV を copy せず、`fle3n-rogii-v4` / `rogii-sp45-fleongg-blend-v2` の生成ロジック自体を submit notebook に port する。
3. 追加提出前に、hidden rerun で `/kaggle/input/notebooks/.../sp45_projection_submission.csv` のような public output dependency が残らないことを source inspection する。

## Source-port submit

2026-06-20 JST に、copy wrapper の hidden rerun error を受けて `fleongg/fle3n-rogii-v4` の Engine A / SP45 projection 生成ロジックを source-port した notebook を作成し、提出した。

実装:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp082_public_artifact_replay_followup --notebook inference --kernel-id kentookumura/exp082-fle3n-sp45-source-infer --title "exp082 fle3n sp45 source infer" --run-on-push --strict
kaggle kernels push -p experiments/exp082_public_artifact_replay_followup/kaggle/inference
kaggle kernels pull kentookumura/exp082-fle3n-sp45-source-infer -p /tmp/kaggle-pull/exp082-fle3n-sp45-source-infer-v1 -m
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp082-fle3n-sp45-source-infer
kaggle kernels output kentookumura/exp082-fle3n-sp45-source-infer -p /tmp/kaggle-output/exp082_public_artifact_replay_followup/source_inference_v1
uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp082_public_artifact_replay_followup/source_inference_v1/submission.csv --sample data/raw/sample_submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp082-fle3n-sp45-source-infer -v 1 -f submission.csv -m "exp082 fle3n SP45 source-port v1 sha 9fb152e8"
```

- source-port notebook は保存済み `fleongg/fle3n-rogii-v4` の cell 0-37 を使い、Engine A / SP45 projection までで停止する。
- `kernel_sources: []`、dataset sources は `phongnguyn23021656/koolbox-offline`、`fleongg/rogii-claude-models-pub`、`ravaghi/wellbore-geology-prediction-artifacts` の 3 件。
- `/kaggle/input/notebooks/...` 参照はなし。public notebook output CSV copy は使わない。
- commit run は約 10.7 分で完了。
- output: `/tmp/kaggle-output/exp082_public_artifact_replay_followup/source_inference_v1/submission.csv`
- rows: `14151`
- submission SHA: `9fb152e8ecc045b602597d8bdf87578d1f3ec4aa34eff0e857aceccfb2e75eb1`
- sp45 projection file SHA: `63bdefba748ffc2153c0dd7cb33dd2ddf7b66eefce8ff15b12ff50d34880ac34`
- submit-check: PASS、FAIL/WARN なし。
- public fle3n SP45 output との差: RMSE `0.512948371`、p95 abs `1.072252461`、max abs `1.297735026`。
- ridge-sp anchor との差: RMSE `1.266065211`。

提出:

- ref: `53854058`
- date UTC: `2026-06-19 15:46:07.987000`
- status: `SubmissionStatus.COMPLETE`
- Public LB: `7.857`
- raw API errorDescription: null
- scriptVersionId: `328675253`

判定:

- ref `53854058` は hidden rerun が完了し、Public LB 7.857 を記録したため採用候補。
- ref `53853237` は hidden rerun error のため不採用。
- exp082 は `submitted_source_port_public_lb_7_857` として記録する。

## Source-port next candidates guard

2026-06-20 JST に `sp45_fleongg_source_port_next_candidates` の実装と監査を追加した。目的は、追加提出前に fle3n final、jaemin SP45/Fleongg final、Pilkwang branch shortlist が hidden-compatible source-port 候補として扱えるかを public sample 生成物、archived source、source risk、pairwise distance で確認すること。

実装:

```bash
uv run ruff check experiments/exp082_public_artifact_replay_followup/sp45_fleongg_source_port_next_candidates.py
uv run python experiments/exp082_public_artifact_replay_followup/sp45_fleongg_source_port_next_candidates.py
```

- ruff: pass。
- guard status: `next_candidate_guard_completed`。
- output:
  - `artifacts/sp45_fleongg_source_port_next_candidates_summary.json`
  - `artifacts/sp45_fleongg_source_port_next_candidates_submission_summary.csv`
  - `artifacts/sp45_fleongg_source_port_next_candidates_source_risk.csv`
  - `artifacts/sp45_fleongg_source_port_next_candidates_pairwise.csv`
  - `artifacts/sp45_fleongg_source_port_next_candidates_README.md`

判定:

- `fle3n_final_blend`: `ready_for_one_hidden_compatible_source_port_run`。SHA `359b3e779d360ac8117a7da8040ef780905381aec160d385b72e354ef710279b`。archived source あり、`/kaggle/input/notebooks` / input submission CSV read / hardcoded input submission は 0。exp082 source-port submission との差は RMSE `1.517454052`、p95 abs `3.323760085`。
- `jaemin_sp45_fleongg_final`: `ready_for_one_hidden_compatible_source_port_run`。SHA `d8b0af2cc9b3d7f299dd63a6cf6333918c222c6790eba8a69eab40de3e8fef45`。archived source あり、blocking risk なし。exp082 source-port submission との差は RMSE `1.501956246`、p95 abs `3.318228552`。
- `pilkwang_raw_projection`: `blocked_missing_archived_source`。SHA `2caccb1019fec9f1cb07961d1dfe68af33e84b3843a656ab51f9bbebef138b8f`。output はあるが `pilkwang/rogii-target-free-tvt-geosteering` の exact source が `docs/notebooks` にない。
- `pilkwang_w0_60_blend`: `blocked_missing_archived_source`。SHA `320a08151fb29ace415c6a6e88c5ecd5fc565ba24526eabe0eb83826242b6981`。output はあるが exact source がない。

Pairwise:

- fle3n final vs jaemin final: RMSE `0.275729904`、p95 abs `0.640721295`、max abs `1.086850887`。
- Pilkwang raw projection vs Pilkwang w0.60 blend: RMSE `1.347751473`、p95 abs `2.989224055`。

次のアクション:

1. 追加提出する場合は、まず `fle3n_final_blend` を 1 回だけ source-port run する。既存 fle3n SP45 source-port notebook を cell 37 以降へ伸ばす方向が最小差分。
2. Kaggle commit output の submit-check、runtime、生成 notebook の `/kaggle/input/notebooks` 依存なしを確認するまで submit しない。
3. jaemin final は fle3n final と近いため、fle3n final の実行結果を見てから代替候補にする。

## fle3n final blend source-port run

2026-06-20 JST に、next-candidate guard の推奨どおり `fle3n_final_blend` を Kaggle 上で source-port 実行した。既存の SP45 source-port notebook を `fleongg/fle3n-rogii-v4` の Engine B と final blend セルまで伸ばし、public notebook output copy ではなく hidden test 上で `sp45_projection_submission.csv`、`fleongg_pretrained_submission.csv`、final `submission.csv` を再生成する構成にした。

実装 / push:

```bash
uv run python scripts/validate_experiment.py --experiment exp082_public_artifact_replay_followup
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp082_public_artifact_replay_followup --notebook inference --kernel-id kentookumura/exp082-fle3n-final-source-infer --title "exp082 fle3n final source infer" --run-on-push --strict
kaggle kernels push -p experiments/exp082_public_artifact_replay_followup/kaggle/inference
kaggle kernels pull kentookumura/exp082-fle3n-final-source-infer -p /tmp/kaggle-pull/exp082-fle3n-final-source-infer-v1 -m
kaggle kernels logs kentookumura/exp082-fle3n-final-source-infer
kaggle kernels output kentookumura/exp082-fle3n-final-source-infer -p /tmp/kaggle-output/exp082_public_artifact_replay_followup/fle3n_final_source_inference_v1
uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp082_public_artifact_replay_followup/fle3n_final_source_inference_v1/submission.csv --sample data/raw/sample_submission.csv
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp082_public_artifact_replay_followup/fle3n_final_source_inference_v1/submission.csv
```

- kernel: `kentookumura/exp082-fle3n-final-source-infer`
- version: `1`
- metadata: CPU, internet off, kernel sources なし, dataset sources は `koolbox-offline` / `rogii-claude-models-pub` / `wellbore-geology-prediction-artifacts` の 3 件。
- runtime: logs 上の final metrics 出力は `770.8s` 付近。およそ 13 分。
- output: `/tmp/kaggle-output/exp082_public_artifact_replay_followup/fle3n_final_source_inference_v1`
- submit-check: PASS。FAIL/WARN なし。
- `scripts/validate_submission.py`: PASS。

生成物:

- `submission.csv`: SHA `40ffcd3daf554fc6b79f472bc5da8d0e4f7d0cb88f8a464a87bbb826c5a15ceb`
- `sp45_projection_submission.csv`: SHA `6cf719da66759e023873a277ef491fd1ce6c11395d6ec0932ca59e6f2d40a329`
- `fleongg_pretrained_submission.csv`: SHA `5c161e22d3e7c2cabb7f4cd26eb11502a64ac7b4702e49df2e2d5afcbcc640db`
- `sp45_fleongg_blend_report.csv`: SHA `9ff7368bba91d2ac9b06d45ee683a1e27b0744e905324b6275a4c9de5c872b86`
- rows: `14151`
- final blend weight: `0.55 * SP45 + 0.45 * fleongg`

差分:

- final vs source-port SP45 sidecar: RMSE `1.228861886`、p95 abs `2.634026025`、max abs `4.237593929`
- final vs source-port fleongg sidecar: RMSE `1.501942305`、p95 abs `3.219365142`、max abs `5.179281468`
- source-port SP45 vs fleongg sidecar: RMSE `2.730804191`、p95 abs `5.853391168`、max abs `9.416875397`
- final vs public fle3n final output: RMSE `0.292760267`、p95 abs `0.817765881`、max abs `1.233608528`
- final vs public jaemin final output: RMSE `0.372714273`、p95 abs `0.861308148`、max abs `1.651757724`
- final vs previous exp082 SP45 source-port submission: RMSE `1.665882481`、p95 abs `3.530793391`、max abs `5.405683588`
- final vs ridge-sp anchor: RMSE `2.042215415`、p95 abs `4.796506820`、max abs `5.829032861`

注意:

- Kaggle v1 は完了し output も取得済みだが、nbconvert が markdown cell に `execution_count` / `outputs` が残っている warning を出した。スコア対象の `submission.csv` 生成には影響していない。ローカルの正 notebook は warning 原因を除去済みで、`validate_experiment` は pass。
- competition submit は後続で実行済み。`kentookumura/exp082-fle3n-final-source-infer` v1 の `submission.csv` を notebook submission として提出し、ref `53885305` / Public LB `7.601` を記録した。

## fle3n final blend submit

ユーザーが `kentookumura/exp082-fle3n-final-source-infer` v1 を提出済み。2026-06-21 JST に Kaggle submissions と monitor script で確認した。

確認コマンド:

```bash
kaggle competitions submissions rogii-wellbore-geology-prediction | head -10
python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py exp082_fle3n_final_source_port_v1 --competition rogii-wellbore-geology-prediction
```

- ref: `53885305`
- submitted UTC: `2026-06-20 14:32:10.007000`
- kernel: `kentookumura/exp082-fle3n-final-source-infer`
- kernel version: `1`
- file: `submission.csv`
- status: `SubmissionStatus.COMPLETE`
- monitor status: `complete`
- Public LB: `7.601`
- Private LB: `-`
- submission SHA: `40ffcd3daf554fc6b79f472bc5da8d0e4f7d0cb88f8a464a87bbb826c5a15ceb`

判定:

- fle3n final source-port v1 は exp082 SP45 source-port ref `53854058` / Public LB `7.857` を `-0.256` 改善した。
- 現時点の ensemble route / public notebook replay anchor を ref `53885305` / Public LB `7.601` に更新する。
- これは LightGBM/CatBoost/Ridge stack、SP45 PF/Beam selector、fleongg pretrained branch の final blend であり、ML route anchor でも PF/Beam 単独 route anchor でもない。

## jaemin final source-port run

2026-06-21 JST に、`jaemin_final_source_port_once` を実装し、`jaemin3404/rogii-sp45-fleongg-blend-v2` の archived source から SP45 branch、fleongg pretrained branch、final `0.55 * SP45 + 0.45 * fleongg` blend を hidden-compatible notebook として再生成した。public notebook output copy は使わず、kernel sources は空、dataset sources は `koolbox-offline` / `rogii-claude-models-pub` / `wellbore-geology-prediction-artifacts` の 3 件。

実装 / push:

```bash
uv run python scripts/validate_experiment.py --experiment exp082_public_artifact_replay_followup
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp082_public_artifact_replay_followup --notebook inference --kernel-id kentookumura/exp082-jaemin-final-source-infer --title "exp082 jaemin final source infer" --run-on-push --strict
kaggle kernels push -p experiments/exp082_public_artifact_replay_followup/kaggle/inference
kaggle kernels output kentookumura/exp082-jaemin-final-source-infer -p /tmp/kaggle-output/exp082_public_artifact_replay_followup/jaemin_final_source_inference_v1
uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp082_public_artifact_replay_followup/jaemin_final_source_inference_v1/submission.csv --sample data/raw/sample_submission.csv
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp082_public_artifact_replay_followup/jaemin_final_source_inference_v1/submission.csv
python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py exp082_jaemin_final_source_port_v1 --competition rogii-wellbore-geology-prediction
```

- kernel: `kentookumura/exp082-jaemin-final-source-infer`
- version: `1`
- metadata: CPU、internet off、GPU off、kernel sources なし。
- output: `/tmp/kaggle-output/exp082_public_artifact_replay_followup/jaemin_final_source_inference_v1`
- runtime: logs の final metrics 出力まで約 `735.8s`。
- submit-check: PASS。FAIL/WARN なし。
- `scripts/validate_submission.py`: PASS。

生成物:

- `submission.csv`: SHA `f789960d9a2e9f8bdaa107dd56f723d35035f1b0fe82673d148cc77f5071c5b9`
- `sp45_projection_submission.csv`: SHA `30ba6b0b238b9e2a95e9c70949085c39224022d051f323c4fddbd3aa3d2bc506`
- `fleongg_pretrained_submission.csv`: SHA `4bfa4d16051049db173499367ddb70e6a9fdfeb614826a122bc337d486fdee90`
- `sp45_fleongg_blend_report.csv`: SHA `0ac0d9bb21365676176122cbf2b79f31f797ffdde32c4e2277cf3119a490f953`
- rows: `14151`
- prediction range: `[11592.354866195146, 12239.478352003349]`
- prediction mean/std: `11904.457569629783` / `278.55495262937006`

差分:

- final vs source-port SP45 sidecar: RMSE `1.304190152`、p95 abs `2.554989859`、max abs `3.512800818`
- final vs source-port fleongg sidecar: RMSE `1.594010186`、p95 abs `3.122765383`、max abs `4.293423222`
- source-port SP45 vs fleongg sidecar: RMSE `2.898200338`、p95 abs `5.677755242`、max abs `7.806224040`
- final vs fle3n final source-port output: RMSE `0.334413867`、p95 abs `0.818307248`、max abs `1.171138134`
- final vs public jaemin final output: RMSE `0.402517811`、p95 abs `0.942300394`、max abs `1.514561042`
- final vs public fle3n final output: RMSE `0.478734898`、p95 abs `1.099144655`、max abs `1.617784432`
- final vs ridge-sp anchor: RMSE `2.046233718`、p95 abs `4.773755008`、max abs `5.914153680`

提出状況:

- User submitted notebook output after v1 completion.
- `kaggle competitions submissions` で 2026-06-21 に observed refs を確認した。
- `53896556`: `SubmissionStatus.COMPLETE`、Public LB `7.602`。
- `53896658`: `SubmissionStatus.COMPLETE`、Public LB 空。
- `53896594` は exp096 の Public LB `8.651` 提出として再帰属した。
- jaemin final source-port は fle3n final source-port ref `53885305` / Public LB `7.601` を `+0.001` 下回れなかったため不採用。現 ensemble route anchor は ref `53885305` のまま。
