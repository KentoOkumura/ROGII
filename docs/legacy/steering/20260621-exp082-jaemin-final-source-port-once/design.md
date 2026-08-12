# 設計

## アプローチ

既存 exp082 の fle3n final source-port notebook を、jaemin archived source に差し替える。jaemin source は `docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_20260611/jaemin3404__rogii-sp45-fleongg-blend-v2/rogii-sp45-fleongg-blend-v2.py` を正とし、`# %%` 区切りを notebook cell に展開する。Kaggle hidden rerun で path が揺れても動くように、competition data root と artifact root は fallback search を持たせる。

## 実験範囲

- 対象実験: `experiments/exp082_public_artifact_replay_followup`
- Route: `ensemble`
- 親実験: `exp079_public_artifact_replay_integrity_audit`
- 変更する変数: source-port 対象を fle3n final から jaemin SP45/Fleongg final に切り替える。
- 固定する変数: dataset sources、CPU / internet off、final blend weight `0.55 * SP45 + 0.45 * fleongg`、public output copy 不使用。

## 再現性設計

- seed policy: archived jaemin source に従う。SP45 branch は PF / seed ensemble を含むため、hidden-compatible rerun candidate であり deterministic ML anchor とは扱わない。
- stochastic 処理の有無: あり。PF / likelihood-PF / numba RNG を含む。
- PF/Beam / likelihood-PF / seed bagging の有無: あり。SP45 branch と fleongg pretrained branch の両方を source-port する。
- 並列処理と乱数の関係: archived source の実装を保つ。出力 SHA を主証拠として記録し、byte-identical deterministic anchor とは主張しない。
- CPU/GPU runtime と deterministic flags: CPU、internet off。GPU は使わない。
- train cache / test feature regeneration の SHA 記録方針: source-port output の `submission.csv` と sidecar CSV SHA を記録する。feature cache SHA は対象外。
- model manifest / prediction / submission SHA 記録方針: mounted pretrained boosters の個別 SHA は今回は追わず、prediction sidecar SHA と final submission SHA を記録する。
- Kaggle package bootstrap 確認方針: `prepare_kaggle_notebooks --strict` 後の metadata で `kernel_sources=[]`、dataset sources 3 件、internet off、GPU off を確認する。

## リスク

- リークリスク: public output CSV を読むと hidden rerun error になるため、notebook source から生成する。`/kaggle/input/notebooks` 依存を残さない。
- CV/LB 不一致リスク: CV はない。fle3n final と近い public replay candidate なので改善幅は小さい可能性が高い。
- ランタイム/メモリリスク: fle3n final source-port が約 13 分で完了しており、jaemin source も同程度を想定する。Kaggle timeout / memory は logs で確認する。
- 再現性リスク: PF / seed ensemble を含むため deterministic anchor としては扱わない。Kaggle version と output SHA を採用判断の証拠にする。
