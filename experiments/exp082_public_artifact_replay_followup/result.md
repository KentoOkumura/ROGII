# exp082_public_artifact_replay_followup 結果

## 仮説

SP45 / fle3n / Koolbox 系公開 notebook は exact source slug と dependency slug を固定し、source risk と生成物 distance を監査してからでないと replay / submit 候補として扱えない。

## 設定

- 親: exp079_public_artifact_replay_integrity_audit
- 検証: target-free integrity audit
- メトリック: integrity audit counts / pairwise submission distance
- シード: no_rng_used
- Kaggle kernel: `kentookumura/exp082-artifact-followup-train`
- 正の実行: version 2
- output: `/tmp/kaggle-output/exp082_public_artifact_replay_followup/train_v2`

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | 7.601 |
| Private LB | - |
| Submit ref | 53885305 |
| Audit status | audit_completed |
| Missing required sources | 0 |
| Source inspections | 7 |
| Candidate files | 19 |
| Valid submission CSVs | 18 |
| Read errors | 1 |
| Pairwise distances | 153 |

## 主な確認結果

- v1 では `rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction` が Kaggle input tree に mount されず required source missing になった。単独 addability probe でも mount されないことを確認したため、v2 では required source から外し、direct-output reference として扱う。
- v2 で mount された候補は ridge-sp、fle3n v4、jaemin SP45/Fleongg blend v2、Koolbox best 8.188、package manager source。
- source inspection は 7 件。内訳は `.ipynb` 5 件、`.py` 2 件。
- fle3n v4 と jaemin SP45/Fleongg blend v2 は `writes_submission_csv=5`、`reads_submission_csv=4`、`hardcoded_working_submission=1`、`exact_match_or_override=1`。
- ridge-sp と Koolbox best 8.188 は `writes_submission_csv=1`、`reads_submission_csv=2`。Koolbox は `mentions_public_or_visible=1`。
- package manager source は source risk hits 0。
- candidate CSV は 19 件、うち `sp45_fleongg_blend_report.csv` は report CSV で `id` 列がないため read error。valid submission CSV は 18 件。
- すべての valid submission CSV は `14151` rows、missing IDs 0、extra IDs 0。

## 代表 SHA

- ridge-sp final: `de1766fa3037be4a53e60b8d95bb0fe83ec094d981050c6c4e315c6e4861580d`
- fle3n v4 final: `359b3e779d360ac8117a7da8040ef780905381aec160d385b72e354ef710279b`
- fle3n v4 SP45 projection: `9aa8a5d0f6ea3ef60dcad11983d40fd42884c0d9e0031956e22f40aeda7c3d0b`
- jaemin SP45/Fleongg v2 final: `d8b0af2cc9b3d7f299dd63a6cf6333918c222c6790eba8a69eab40de3e8fef45`
- jaemin SP45 projection: `ca09d625aef8e23440bdc1710d7a58282f2e0a00766e141c918cb1b314914f9d`
- Koolbox best 8.188 final: `8520111d5b7e1812c5298cdf2e18b1a1c6a59feb8bc368aa8fd33bf453075341`

## Pairwise 抜粋

- ridge-sp vs fle3n v4 final: RMSE 1.890560287
- ridge-sp vs fle3n v4 SP45 projection: RMSE 1.384232857
- ridge-sp vs jaemin SP45/Fleongg final: RMSE 1.949888855
- ridge-sp vs jaemin SP45 projection: RMSE 1.413346840
- ridge-sp vs Koolbox best 8.188: RMSE 1.551098293
- fle3n v4 final vs jaemin SP45/Fleongg final: RMSE 0.275729904
- fle3n v4 final vs Koolbox best 8.188: RMSE 1.546923624
- jaemin SP45/Fleongg final vs Koolbox best 8.188: RMSE 1.486587152

## 再現性

- deterministic anchor: false
- seed policy: no_rng_used
- kernel version: 2
- feature content SHA: 対象外
- model SHA / manifest SHA: 対象外
- prediction SHA: candidate CSV SHA を `exp082_public_artifact_replay_followup_submission_summary.csv` に保存
- submission SHA: candidate CSV SHA を `exp082_public_artifact_replay_followup_submission_summary.csv` に保存

## 解釈

SP45/Fleongg 系の主要 branch と Koolbox best 8.188 は、生成物としては sample 互換で、既存 ridge-sp から RMSE 1.38-1.95 程度の差を持つ。fle3n v4 final と jaemin SP45/Fleongg final は RMSE 0.2757 と近く、同系統の重複候補として扱うのが妥当。

v2 で mountable source の audit は完了した。`rauffauzanrambe/rogii-sp45-wellbore-for-blend-prediction` は Kaggle source として mount できないが、CLI output は直接取得できるため direct-output reference として別枠で扱う。code-submit 再現候補としては採用しない。

## SP45 projection guard

3 件の SP45 projection を追加で submit-check / row-level guard に通した。

| 候補 | 種別 | SHA256 | ridge-sp RMSE | Pilkwang raw projection RMSE |
| --- | --- | --- | --- | --- |
| fle3n SP45 projection | mountable candidate | `9aa8a5d0f6ea3ef60dcad11983d40fd42884c0d9e0031956e22f40aeda7c3d0b` | 1.384232857 | 1.256802613 |
| jaemin SP45 projection | mountable candidate | `ca09d625aef8e23440bdc1710d7a58282f2e0a00766e141c918cb1b314914f9d` | 1.413346840 | 1.192134839 |
| rauff SP45 projection | direct-output reference | `4e2bfc43b4b2202a5a9fc8808f3a42d408d86cee9c401f1eae4a1fcaa2c5edb9` | 1.303650505 | 1.208265533 |

3 件とも `scripts/validate_submission.py` と `.agents/skills/kaggle-submit-check/scripts/check_submission.py --sample data/raw/sample_submission.csv` で pass。rows 14151、columns 2、header / row count は sample と一致、重複 ID / NaN / Inf 相当値なし。

候補間では fle3n vs jaemin が RMSE 0.324981626、p95 abs 0.618058119、max abs 0.733854728 で近い。jaemin vs rauff は RMSE 0.230330314 とさらに近いが、rauff は Kaggle source として mount できない。

提出候補としては、code-submit 再現可能な mountable 候補から 1 件に絞る。ridge-sp anchor からの drift を重視するなら fle3n SP45 projection、Pilkwang raw projection への近さを重視するなら jaemin SP45 projection。保守的な次アクションは fle3n SP45 projection を 1 件だけ提出候補にすること。

## Submit

`fle3n` SP45 projection を Kaggle Notebook から提出した。最初の public output copy wrapper は hidden rerun error になったため失敗扱い。次に `fleongg/fle3n-rogii-v4` の Engine A / SP45 projection 生成ロジックを notebook に source-port し、hidden test 上で再生成する形で提出した。

### 失敗した copy wrapper

- kernel: `kentookumura/exp082-fle3n-sp45-infer`
- kernel version: 2
- output: `/tmp/kaggle-output/exp082_public_artifact_replay_followup/inference_v2/submission.csv`
- submission SHA: `9aa8a5d0f6ea3ef60dcad11983d40fd42884c0d9e0031956e22f40aeda7c3d0b`
- ref: `53853237`
- status: `SubmissionStatus.COMPLETE` with hidden rerun error
- Public LB: null

CSV 直接提出は `CreateSubmission` 400 で拒否された。API response は `Submission not allowed: This competition only accepts Submissions from Notebooks.`。そのため、inference notebook が `/kaggle/input/notebooks/fleongg/fle3n-rogii-v4/sp45_projection_submission.csv` を SHA 検証して `/kaggle/working/submission.csv` に byte copy する形で提出した。

raw API では `errorDescription` が返っている。内容は `Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset`。通常 commit run では public notebook output を mount して `submission.csv` を作れたが、code competition の hidden rerun ではこの前提が成立しない。したがって、この提出 wrapper は hidden-rerun compatible ではない。

### 成功した source-port submit

public output copy をやめ、保存済み `fleongg/fle3n-rogii-v4` notebook の cell 0-37 を元に Engine A / SP45 projection までを再実行する notebook を作成した。metadata は `kernel_sources: []`、dataset sources は `koolbox-offline`、`rogii-claude-models-pub`、`wellbore-geology-prediction-artifacts` の 3 件のみ。`/kaggle/input/notebooks/...` には依存しない。

- kernel: `kentookumura/exp082-fle3n-sp45-source-infer`
- kernel version: 1
- scriptVersionId: `328675253`
- output: `/tmp/kaggle-output/exp082_public_artifact_replay_followup/source_inference_v1/submission.csv`
- submission SHA: `9fb152e8ecc045b602597d8bdf87578d1f3ec4aa34eff0e857aceccfb2e75eb1`
- sp45 projection file SHA: `63bdefba748ffc2153c0dd7cb33dd2ddf7b66eefce8ff15b12ff50d34880ac34`
- rows: `14151`
- submit-check: PASS
- ref: `53854058`
- status: `SubmissionStatus.COMPLETE`
- Public LB: `7.857`
- raw API errorDescription: null

source-port 出力は public fle3n SP45 output と byte-identical ではない。public fle3n SP45 output との差は RMSE `0.512948371`、p95 abs `1.072252461`、max abs `1.297735026`。ridge-sp anchor との差は RMSE `1.266065211`。

## SP45/Fleongg source-port next candidates

`sp45_fleongg_source_port_next_candidates.py` を追加し、次に source-port できる公開 notebook route 候補を監査した。生成物は `artifacts/sp45_fleongg_source_port_next_candidates_*` に保存済み。

| 候補 | 判定 | SHA256 | exp082 submitted source-port RMSE | 備考 |
| --- | --- | --- | --- | --- |
| fle3n final blend | ready_for_one_hidden_compatible_source_port_run | `359b3e779d360ac8117a7da8040ef780905381aec160d385b72e354ef710279b` | 1.517454052 | archived source あり。`/kaggle/input/notebooks` と input submission CSV 依存なし。 |
| jaemin SP45/Fleongg final | ready_for_one_hidden_compatible_source_port_run | `d8b0af2cc9b3d7f299dd63a6cf6333918c222c6790eba8a69eab40de3e8fef45` | 1.501956246 | archived source あり。fle3n final と RMSE 0.275729904 で近い。 |
| Pilkwang raw projection | blocked_missing_archived_source | `2caccb1019fec9f1cb07961d1dfe68af33e84b3843a656ab51f9bbebef138b8f` | 1.352145079 | output はあるが `pilkwang/rogii-target-free-tvt-geosteering` の exact source がローカル archive にない。 |
| Pilkwang w0.60 blend | blocked_missing_archived_source | `320a08151fb29ace415c6a6e88c5ecd5fc565ba24526eabe0eb83826242b6981` | 1.528199330 | output はあるが exact source がローカル archive にない。 |

source risk では fle3n / jaemin とも `writes_submission_csv=5`、`mentions_public_or_visible=1` はあるが、blocking 扱いの `/kaggle/input/notebooks/...`、input 側 submission CSV read、hardcoded input submission は 0。public output copy ではなく、生成ロジックを hidden test 上で再実行する source-port 候補として扱える。

推奨次アクションは `fle3n_final_blend` の source-port を 1 回だけ実行すること。理由は、既存の fle3n SP45 source-port notebook を cell 37 以降へ伸ばすだけで差分が小さく、jaemin final は fle3n final と近いため重複候補として後回しにできるため。Kaggle commit output が submit-check を通り、生成 notebook に `/kaggle/input/notebooks` 依存がないことを確認するまでは submit しない。

## fle3n final blend source-port run

`fle3n_final_blend` の hidden-compatible source-port run を Kaggle で実行した。public output copy ではなく、`fleongg/fle3n-rogii-v4` の Engine A SP45 projection、Engine B fleongg pretrained branch、final `0.55 * SP45 + 0.45 * fleongg` blend を notebook 内で再生成する構成。

- kernel: `kentookumura/exp082-fle3n-final-source-infer`
- version: `1`
- output: `/tmp/kaggle-output/exp082_public_artifact_replay_followup/fle3n_final_source_inference_v1`
- runtime: logs の final metrics 出力まで約 `770.8s`
- submit-check: PASS、FAIL/WARN なし
- validate_submission: PASS
- rows: `14151`
- submission SHA: `40ffcd3daf554fc6b79f472bc5da8d0e4f7d0cb88f8a464a87bbb826c5a15ceb`
- SP45 sidecar SHA: `6cf719da66759e023873a277ef491fd1ce6c11395d6ec0932ca59e6f2d40a329`
- fleongg sidecar SHA: `5c161e22d3e7c2cabb7f4cd26eb11502a64ac7b4702e49df2e2d5afcbcc640db`

差分:

| 比較 | RMSE | p95 abs | max abs |
| --- | ---: | ---: | ---: |
| final vs source-port SP45 sidecar | 1.228861886 | 2.634026025 | 4.237593929 |
| final vs source-port fleongg sidecar | 1.501942305 | 3.219365142 | 5.179281468 |
| source-port SP45 vs fleongg sidecar | 2.730804191 | 5.853391168 | 9.416875397 |
| final vs public fle3n final output | 0.292760267 | 0.817765881 | 1.233608528 |
| final vs public jaemin final output | 0.372714273 | 0.861308148 | 1.651757724 |
| final vs previous exp082 SP45 source-port | 1.665882481 | 3.530793391 | 5.405683588 |
| final vs ridge-sp anchor | 2.042215415 | 4.796506820 | 5.829032861 |

v1 の Kaggle run は完了し、提出形式も通った。その後、ユーザーが `kentookumura/exp082-fle3n-final-source-infer` v1 を提出し、ref `53885305` / Public LB `7.601` を記録した。

## fle3n final blend submit

- ref: `53885305`
- submitted UTC: `2026-06-20 14:32:10.007000`
- kernel: `kentookumura/exp082-fle3n-final-source-infer`
- kernel version: `1`
- submission SHA: `40ffcd3daf554fc6b79f472bc5da8d0e4f7d0cb88f8a464a87bbb826c5a15ceb`
- status: `SubmissionStatus.COMPLETE`
- Public LB: `7.601`
- Private LB: `-`

この提出は、前回の exp082 SP45 source-port ref `53854058` / Public LB `7.857` を `-0.256` 改善した。現時点の public notebook replay / ensemble route anchor は ref `53885305` / Public LB `7.601` とする。

## jaemin final source-port run

`jaemin_final_source_port_once` として、`jaemin3404/rogii-sp45-fleongg-blend-v2` の archived source から SP45 branch、fleongg pretrained branch、final `0.55 * SP45 + 0.45 * fleongg` blend を source-port した。public output copy は使わず、Kaggle hidden rerun 上で raw competition input と mounted datasets から再生成する構成。

- kernel: `kentookumura/exp082-jaemin-final-source-infer`
- version: `1`
- output: `/tmp/kaggle-output/exp082_public_artifact_replay_followup/jaemin_final_source_inference_v1`
- runtime: logs の final metrics 出力まで約 `735.8s`
- submit-check: PASS、FAIL/WARN なし
- validate_submission: PASS
- rows: `14151`
- submission SHA: `f789960d9a2e9f8bdaa107dd56f723d35035f1b0fe82673d148cc77f5071c5b9`
- SP45 sidecar SHA: `30ba6b0b238b9e2a95e9c70949085c39224022d051f323c4fddbd3aa3d2bc506`
- fleongg sidecar SHA: `4bfa4d16051049db173499367ddb70e6a9fdfeb614826a122bc337d486fdee90`
- blend report SHA: `0ac0d9bb21365676176122cbf2b79f31f797ffdde32c4e2277cf3119a490f953`

差分:

| 比較 | RMSE | p95 abs | max abs |
| --- | ---: | ---: | ---: |
| final vs source-port SP45 sidecar | 1.304190152 | 2.554989859 | 3.512800818 |
| final vs source-port fleongg sidecar | 1.594010186 | 3.122765383 | 4.293423222 |
| source-port SP45 vs fleongg sidecar | 2.898200338 | 5.677755242 | 7.806224040 |
| final vs fle3n final source-port output | 0.334413867 | 0.818307248 | 1.171138134 |
| final vs public jaemin final output | 0.402517811 | 0.942300394 | 1.514561042 |
| final vs public fle3n final output | 0.478734898 | 1.099144655 | 1.617784432 |
| final vs ridge-sp anchor | 2.046233718 | 4.773755008 | 5.914153680 |

## jaemin final submit

ユーザーが `kentookumura/exp082-jaemin-final-source-infer` v1 を提出済み。Kaggle submissions 上では、jaemin source-port として ref `53896556` / `53896658` を確認した。ref `53896594` は exp096 の Public LB 8.651 提出として再帰属した。

- scored ref: `53896556`
- status: `SubmissionStatus.COMPLETE`
- Public LB: `7.602`
- Private LB: `-`
- submission SHA: `f789960d9a2e9f8bdaa107dd56f723d35035f1b0fe82673d148cc77f5071c5b9`
- duplicate / related ref `53896658`: `SubmissionStatus.COMPLETE`、Public LB 空

Public LB `7.602` は fle3n final source-port ref `53885305` / Public LB `7.601` に `+0.001` 届かなかった。現 ensemble route anchor は fle3n final のままにする。public replay route の追加 submit は一旦止める。

## 次

1. ref `53885305` を exp082 の現採用候補として扱う。Public LB 7.601 は現時点の public notebook replay / ensemble anchor。提出ロジックは LightGBM/CatBoost/Ridge stack、SP45 PF/Beam selector、fleongg pretrained branch の final blend であり、MLルート anchor ではない。
2. ref `53896556` は jaemin final source-portとして保持するが、Public LB 7.602 のため採用しない。
3. ref `53853237` は hidden rerun error のため採用しない。
4. ref `53854058` は旧 exp082 SP45 source-port anchor として保持するが、現 anchor は ref `53885305` に更新する。
