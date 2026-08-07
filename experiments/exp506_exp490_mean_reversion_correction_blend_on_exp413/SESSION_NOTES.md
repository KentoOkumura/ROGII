# exp506_exp490_mean_reversion_correction_blend_on_exp413 セッションノート

## 目的

exp490をstandaloneやhard selectorとして採用せず、`exp490-exp357`のnovel correctionだけを
現在のexp413-family anchorへ小さく加えるfold-safeな最終アンサンブルを設計する。

## 現在の状態

- Route: `ensemble`
- 状態: Stage A version 2 COMPLETE / primary gate FAIL / terminal close
- resolved anchor: exp413 Stage D保存OOF
- primary / selected CV: `7.902068462119896 / 7.884802794404715`
- LB: なし
- implementation / Kaggle package / run: 1 / 1 / v1 error・v2 complete
- inference / submission: 0 / 0
- 正規train Notebook: compact self-contained候補を採用
- 正規inference Notebook: template placeholder、科学ロジックなし

## 2026-08-03 設計確定

- 次の未使用番号としてexp506を採用した。
- root parentをexp413、correction sourceをexp490、correction parentをexp357とした。
- exp497が実行中のため、Stage Eを先に終端判定する依存関係を固定した。
- exp497が自身の事前gateをPASSしてnon-exp413 predictionを選んだ場合だけ、その保存OOFをanchorにする。
- その他はexp413 Stage D保存OOFをanchorにする。exp506 outcomeを使うanchor選択は禁止した。
- primaryを`anchor + lambda*(exp490-exp357)`に固定した。
- lambdaはother-four-fold closed-form SSE fit、範囲`[0.00, 0.10]`、interceptなしとした。
- held meta-foldごとにfitし、deployment値は5係数の中央値、full-OOF再fitなしとした。
- direct convex blendはprimary freeze後のreport-only controlとし、selectable=falseとした。
- tau fade、depth/scope weight、row/well gate、router、negative weight、3-way stackを禁止した。
- primary gateをgain`>=0.03 ft`、5/5 folds、固定5 scope非悪化、p95/worst各`<=+0.25 ft`、
  全meta weightがstrict interior、range`<=0.05`の全ANDに固定した。
- FAIL時はweight/scope/component/gate救済を行わず、推論・提出へ進まない。

## 変更点

- 最終anchorの予測面は変更せず、`exp490-exp357`の単一補正成分だけを追加対象にした。
- 学習器、selector、HMM/PF/Beam、fade、routerを追加せず、0-model meta-fold監査に限定した。
- exp497の終端結果だけでanchorをresolveし、exp506結果によるanchor選択を禁止した。

## 2026-08-04 exp497終端とanchor freeze

- exp497 Stage E version 1は`completed_gate_failed_closed`で終端した。
- exp497 candidate / selected CVは`7.87448814999802 / 7.884802794404715`。
- pooled gain不足、fold、fixed scope、by-well tailがFAILし、promotion gateはfalseだった。
- selected predictionは`exp413_oof`。exp506の結果を一切見ず、exp413 Stage D保存OOFをanchorへ固定した。
- anchor prediction列は`scale5_x1p0_full_replacement__lgb_mean__pred_tvt`、file SHAは
  `9bd2d17778b3b27d771b12cbff72def8b87e6cdf14062e1c0ba192434cef4a9d`。
- exp497 gate SHAは`95c82331e89171a735f858b0f6be36f6af035b95206aff0dc31a4b73d24c332e`、
  reproducibility manifest SHAは`65777defb047e2d13e0b20877d2ae682e5a934e1ef61429d8039eb2d46aa6c48`。

## 2026-08-04 Stage A実装

- ユーザーの実装依頼を、Stage A compact self-contained候補とcontract testsの承認として記録した。
- 正規train Notebook、Kaggle package、Kaggle run、inference、submissionは承認範囲に含めない。
- Jupytext percent sourceは12章構成で、同じexp内helperをimportしないself-contained実装とした。
- exp413 OOFは最初にtruthなし5列だけを読み、IDから`well / row_idx`を復元する。
- exp490 gzipは`well / row_idx / suffix_offset / md_since / exp490 / exp357`のexact 6-column
  allowlistだけを読み、upstream fold、truth、error、episode、role、scope、by-well、gate列を除外する。
- anchor/correctionのkey、suffix offset、MD、outer foldを照合し、`exp490-exp357`と全prediction SHAを
  truth接続前にfreezeする。
- lambdaはother-four-foldのunweighted SSE closed formだけでfitし、`[0,0.10]`へclipしてheld foldへ適用する。
- primary OOFとgateをfreezeした後だけ、非選択のdirect convex controlとresidual診断を計算する。
- output contract 12生成物、各file SHA、config/source/contract SHA、0-model inventoryを記録する。
- 親exp497 compactは8章 / 1,044行、exp506候補は12章 / 1,376行で、Stage Aの入力・phase分離・
  meta-fold・scope/tail・control・再現性orchestrationがNotebook上で追えることを確認した。
- version 2 implementation SHAはsource `213cc926...13f2f4`、candidate/canonical Notebook
  `bbdedbc7...fbf25`、contract test `cf3e7277...28dbf`。

## 2026-08-04 Stage A終端結果

- Kaggle private CPU version 2（id_no `129631767`）は`294.943 sec`でCOMPLETEした。
- primary CVは`7.902068462119896`、anchor `7.884802794404715`比`0.017265667715181 ft`悪化。
- lambdaはfold順に`[0, 0.041578388382360124, 0, 0.004513713824165949, 0]`、
  deployment中央値は`0.0`。nonworseは`3/5 folds`、strict interiorは`2/5`だった。
- MD 3面とhidden-like 2面は全て悪化し、fixed scope nonworseは`0/5`。
- by-well delta p95は`+0.054729023 ft`でPASSしたが、worstはwell `fb03ae90`の
  `+1.816049513 ft`でFAIL。`+0.25 ft`超悪化wellは15、`+1 ft`超は2だった。
- technical / leakage / SHA checksは全PASS。pooled gain、fold、scope、worst-well、lambda positiveを
  FAILし、`FAIL_CLOSE_WITHOUT_WEIGHT_SCOPE_COMPONENT_OR_GATE_RESCUE`で終端した。
- report-only convex controlはCV`7.7345312772318815`だが、全fold weightが上限`0.10`へ張り付き、
  selectable=false / may_rescue_primary=falseを維持する。
- primary OOF file SHA `d459963d...5e1098`、prediction logical SHA `083f379c...da725d`、
  gate SHA `047c13ac...a1f4f6`、reproducibility manifest SHA `11dde33a...55050`。

## 根拠

- exp490: exp357 `9.737195157`から`8.480155260`へ`1.257039898 ft`改善したが、
  by-well p95 / worstは`+7.257814 / +49.602560 ft`、Public LBは`9.680`。
- exp501: fixed13はfixed12から`0.387641747 ft`改善、5/5 folds、全scope改善だが、
  p95 / worstは`+2.904594 / +18.394664 ft`。
- exp499: target-free hard routerはalways exp490より`0.034155367 ft`悪化。
- exp502: exp413 selector置換のgainは`0.002658891 ft`でfold 3/4とhidden-likeを悪化。
- exp500: PFへの機構移植はexp404から`2.101017446 ft`改善したが、p95 / worst
  `+6.653601 / +46.154671 ft`。
- exp505: tau500 selectorはraw exp501から`0.021574771 ft`改善したが、tail縮小は実質なし。
- exp494: CVを`0.057351909 ft`、5/5 folds改善したstackでもPublic LBはexp413より
  `0.027`悪化したため、fold・scope・tail全ANDを維持する。

## 将来の実行量契約

| 項目 | Stage A |
| --- | ---: |
| scientific primary | 1 |
| report-only control | 1 |
| outer / meta folds | 5 / 5 |
| model / booster | 0 / 0 |
| HMM / PF / Beam | 0 / 0 / 0 |
| parent/control再学習 | 0 |
| GPU | 0 |

## 再現性メモ

- `docs/06_reproducibility.md`を確認済み。
- Stage Aはno RNG、固定fold、stable key order、float64 fixed-order reductionとする。
- exp490 gzipはraw SHAとdecompressed content SHAを分け、後者を主証拠にする。
- anchor resolution、input、fold、correction、weight、primary OOF、metrics/gate SHAを記録する。
- Stage A rerun一致まではdeterministic anchorと呼ばない。
- package作成時はmetadataとbootstrap ZIP内config/input contractを照合する。
- inferenceへ進む場合は両pipelineのhidden-dynamic再生成、test inventory、prediction/submission SHAを追加監査する。

## コマンドログ

- 2026-08-03: `make new-steering EXP=exp506_exp490_mean_reversion_correction_blend_on_exp413`
- 2026-08-03: `make new-exp EXP=exp506_exp490_mean_reversion_correction_blend_on_exp413`
- 2026-08-03: steering、config、README、SESSION_NOTES、result、metrics、contract、backlog、summaryをdesign-onlyで記録。
- Kaggle API、Notebook実行、model fit、OOF生成、inference、submissionは実行していない。
- 2026-08-04: compact self-contained Stage A source、候補Notebook、dedicated contract testsを追加。
- 2026-08-04: `py_compile`、Ruff、focused pytest 7件、Jupytext round-trip、strict experiment validatorをPASS。
- 2026-08-04: repository全体pytestは`1832 passed / 8 skipped / 4 failed`。失敗は既存の
  exp293 contract SHA 2件とexp296完了後status/run flag 2件だけで、exp506起因は0件だった。
- 2026-08-04: ローカルNotebook実行、Kaggle package/push/run、OOF生成、inference、submissionは実行していない。
- 2026-08-04: ユーザーが「実行してください」と指示し、正規train Notebook採用、Kaggle package、
  Stage A private CPU runを承認した。inference / submissionは承認範囲外のままとした。
- 2026-08-04 push前実行量: scientific primary 1、report-only control 1、outer/meta fold 5/5、
  LightGBM config 0、trained model 0、booster 0、HMM/PF/Beam 0/0/0、GPU 0、親/control再学習0。
- 2026-08-04 Kaggle入力は保存済みexp413、exp490、exp115 hidden-likeの3 kernel sourceに固定した。
- 2026-08-04: canonical kernelを
  `kentookumura/exp506-exp490-mean-revert-correction-exp413-train`、同一slugへ解決されるtitleを
  `exp506 exp490 mean revert correction exp413 train`に固定した。CPU / internet off / run-on-push。
- 2026-08-04: self-contained Notebookのためrepository `src/`を含めずKaggle packageを生成した。
- 2026-08-04: canonical kernelへversion 1をpushし、run-on-pushでStage Aを開始した。
  URL: `https://www.kaggle.com/code/kentookumura/exp506-exp490-mean-revert-correction-exp413-train`
- 2026-08-04: version 1は329.7秒でERROR。入力3,783,989行 / 773 wells、key・suffix・MD整合、
  truth-late freeze、5 meta-fold lambda推定までは完了した。最後の`print(json.dumps(metrics))`だけが
  `numpy.bool_`を直接serializeして`TypeError: Object of type bool is not JSON serializable`になった。
- 2026-08-04: 科学計算やgateを変更せず、表示時も既存`to_jsonable()`を通す修正と回帰テストを追加した。
- 2026-08-04: version 2修正後、Jupytext round-trip、py_compile、Ruff、focused pytest 8件、
  strict experiment validatorをPASSした。
- 2026-08-04: 同じcanonical kernelへversion 2をpushし、run-on-pushで再実行を開始した。
- 2026-08-04: version 2 COMPLETE。logsでCV / fold / scope / by-well / lambda / gate / 生成物SHAを確認した。
- 2026-08-04: `kaggle kernels output`でversion 2の12生成物を
  `kaggle/output/stage_a_v2/artifacts/`へ取得し、reproducibility manifest記載SHAと全件一致を確認した。
- 2026-08-04: primary OOFは`3,783,989 rows / 773 wells / folds 0--4 / null 0`。
  local pandasのstring inference差をobject dtypeへ復元してlogical prediction SHA`083f379c...da725d`を再確認した。
- 2026-08-04: inference、current-test prediction、submissionは生成していない。
- 2026-08-04: 終端記録更新後にfocused pytest 8件、Jupytext round-trip、py_compile、Ruff、
  strict experiment validator、template validator、metrics JSON validationを全PASSした。

## 次のアクション

1. exp506 primary仮説を終端閉鎖し、inference / submissionへ進まない。
2. 現行P0/P1のexp509 / exp510を優先する。
3. 必要な場合だけ、report-only固定10% convex controlのscope / tail寄与を説明する
   saved-artifact-only readoutをP4で別設計・別承認する。exp506 gateの再評価や推論候補化は禁止する。
