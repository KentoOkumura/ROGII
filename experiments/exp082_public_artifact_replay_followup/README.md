# exp082_public_artifact_replay_followup

## 状態

- ルート: ensemble
- 状態: submitted_jaemin_final_source_port_public_lb_7_602_not_adopted
- CV: -
- Public LB: 7.601
- Private LB: -
- Submit ID: 53885305; jaemin ref 53896556
- 作成日: 2026-06-19
- 親実験: exp079_public_artifact_replay_integrity_audit

## 仮説

`exp079` で placeholder のまま残った SP45 / fle3n / Koolbox 系は、exact source slug と dependency slug を固定して同じ integrity audit を通せば、公開 LB title ではなく生成物・source risk・anchor distance に基づいて submit 候補から採否を判断できる。

## 変更点

- `fleongg/fle3n-rogii-v4`、`jaemin3404/rogii-sp45-fleongg-blend-v2`、`debatreyabiswas/wellboregeology-prediction-with-koolbox-best-8-188` を mountable exact source として固定した。
- `rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction` は direct-output reference として記録した。kernel source としては mount できない。
- `phongnguyn23021656/koolbox-offline`、`fleongg/rogii-claude-models-pub`、`ravaghi/wellbore-geology-prediction-artifacts`、`packagemanager/pm-121774751-at-06-05-2026-09-29-28` を dependency source として固定した。
- `.ipynb` だけでなく `.py` kernel source も static CSV writer / hardcoded submission / public-visible branch risk の検査対象にした。
- train notebook は no-submit audit。inference notebook は guard-selected `fle3n` SP45 projection の生成ロジックを hidden-compatible に source-port し、`submission.csv` として出力する。提出ロジックは LightGBM/CatBoost/Ridge stack と PF/Beam selector の ensemble であり、MLルート anchor ではなく public notebook replay / ensemble candidate として扱う。
- `sp45_fleongg_source_port_next_candidates.py` で fle3n final / jaemin final / Pilkwang branch shortlist の次候補 guard を追加した。
- `jaemin_final_source_port_once` として `jaemin3404/rogii-sp45-fleongg-blend-v2` の SP45 branch、fleongg pretrained branch、final blend を hidden-compatible に source-port した。

## 検証方針

- Fold: なし
- Group: なし
- Stratification: なし
- Leakage Check: static visible CSV、public sample branch、hardcoded input submission、exact/override pattern を source code と生成物 inventory から確認する。

## 実行入口

- 学習 notebook: `exp082_public_artifact_replay_followup_train.ipynb`
- 推論 notebook: `exp082_public_artifact_replay_followup_inference.ipynb`
- Kaggle 準備: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp082_public_artifact_replay_followup --notebook train --run-on-push --strict`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | 7.601 |
| Private LB | - |
| Audit status | audit_completed |
| Kaggle kernel | kentookumura/exp082-artifact-followup-train v2 |
| Submit kernel | kentookumura/exp082-fle3n-sp45-source-infer v1 |
| Submit ref | 53885305 |
| Candidate files | 19 |
| Valid submission CSVs | 18 |
| Source inspections | 7 |
| Pairwise distances | 153 |
| SP45 projection submit-check | pass |
| SP45 projection row-level guard | completed |
| Source-port next candidates guard | completed |
| fle3n final source-port run | submitted / Public LB 7.601 / ref 53885305 |
| jaemin final source-port run | submitted / Public LB 7.602 / ref 53896556 |

fle3n v4、jaemin SP45/Fleongg blend v2、Koolbox best 8.188、ridge-sp は mountable source として監査完了。`rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction` は単独 addability probe でも mount されないため、direct-output reference として別枠で扱う。

`fle3n` SP45 projection は最初に public output copy wrapper として提出したが、ref `53853237` は hidden rerun error になった。次に `fleongg/fle3n-rogii-v4` の Engine A / SP45 projection 生成ロジックを source-port し、public notebook output に依存しない notebook として ref `53854058` を提出した。Public LB は `7.857`。

next-candidate guard では、fle3n final blend と jaemin SP45/Fleongg final は archived source があり、`/kaggle/input/notebooks` / input submission CSV read / hardcoded input submission の blocking risk 0 と判定した。Pilkwang raw projection と w0.60 blend は output はあるが exact archived source がないため、code-submit 再現候補から外す。その後 `fle3n_final_blend` の source-port run を 1 件実行し、commit output の submit-check と runtime を確認した。

`fle3n_final_blend` source-port run は `kentookumura/exp082-fle3n-final-source-infer` v1 として完了した。output は `/tmp/kaggle-output/exp082_public_artifact_replay_followup/fle3n_final_source_inference_v1`、submission SHA は `40ffcd3daf554fc6b79f472bc5da8d0e4f7d0cb88f8a464a87bbb826c5a15ceb`。submit-check と `validate_submission.py` は PASS。その後 ref `53885305` として提出され、Public LB `7.601` を記録した。

`jaemin_final_source_port_once` は `kentookumura/exp082-jaemin-final-source-infer` v1 として完了した。output は `/tmp/kaggle-output/exp082_public_artifact_replay_followup/jaemin_final_source_inference_v1`、submission SHA は `f789960d9a2e9f8bdaa107dd56f723d35035f1b0fe82673d148cc77f5071c5b9`。submit-check と `validate_submission.py` は PASS。ref `53896556` は Public LB `7.602`。ref `53896658` は `SubmissionStatus.COMPLETE` だが Public LB は空。現 anchor は fle3n final ref `53885305` / Public LB `7.601` のままにする。

## 所見

### 良かった点

- `exp079` の audit runner を再利用し、未監査だった public artifact 系 source を exact slug 付きで追加した。
- `.py` kernel source も risk inspection 対象にしたため、SP45/Fleongg blend の script kernel も監査できる。
- Kaggle v2 で missing required sources 0、18 件の valid submission CSV、153 件の pairwise distance、7 件の source inspection を保存できた。
- Kaggle commit run では source-port notebook が public output copy なしで `submission.csv` を生成し、submit-check PASS のまま hidden rerun でも完了した。

### 悪かった点

- `rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction` は output を直接取得できるが、Kaggle input source としては使えない。

### リスク / 注意

- この実験は deterministic anchor ではない。
- CSV 直接提出は Kaggle API で拒否されるため、notebook version から提出する。
- public notebook output CSV を copy するだけの wrapper は hidden rerun で成立しない。生成ロジック自体を source-port した ref `53854058` は完了し、Public LB 7.857 を記録した。
- fle3n v4 final と jaemin SP45/Fleongg final は RMSE 0.275729904 と近く、重複候補として扱う。
- SP45 projection でも fle3n vs jaemin は RMSE 0.324981626、p95 abs 0.618058119 と近いため、提出する場合は両方ではなく 1 件に絞る。
- fle3n SP45 projection は ridge-sp との差 RMSE 1.384232857、jaemin SP45 projection は 1.413346840。保守的には fle3n を第一候補にする。
- ridge-sp との差は fle3n final RMSE 1.890560287、jaemin final RMSE 1.949888855、Koolbox best 8.188 RMSE 1.551098293。
- fle3n final と jaemin final は next source-port run 候補だが、重複度が高いため両方を続けて提出しない。
- fle3n final source-port run は public fle3n final output と RMSE 0.292760267、previous exp082 SP45 source-port と RMSE 1.665882481。提出後 Public LB 7.601 で exp082 SP45 source-port anchor 7.857 を改善した。
- jaemin final source-port run は fle3n final source-port output と RMSE 0.334413867、public jaemin final output と RMSE 0.402517811。Public LB は `7.602` で fle3n final `7.601` に届かなかった。

## リスク / 注意

- この実験は deterministic anchor ではない。
- ref `53853237` は hidden rerun error のため採用しない。ref `53854058` は旧 SP45 source-port anchor として保持し、ref `53885305` を現 ensemble route anchor にする。
- jaemin ref `53896556` は Public LB `7.602` で、Public LB 7.601 を更新できていない。public replay route の追加 submit は一旦止める。
- direct-output reference は code-submit 再現候補として扱わない。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
