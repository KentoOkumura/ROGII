# exp502_exp501_fixed13_selector_replacement_on_exp413

## 状態

- ルート: `ml_model`
- 状態: Kaggle train version 1完了、primary gate FAILで終端閉鎖
- CV: `7.882143903310376`
- Public LB: 未実行
- Private LB: -
- Submit ID: -
- 作成日: 2026-08-02
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- selector source: `exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`

## 仮説

exp501 fixed13 selector は corrected exp264 fixed12 selector に対し hard OOFを
`8.652531956 -> 8.264890209`（`-0.387641747 ft`）改善した。この fold-safe な
selector compact 77列で exp413 の既存 nested selector compact 74列を置換すると、
exp413の他の特徴量・TVTモデル設定を変えずに downstream TVT OOFを改善できる可能性がある。

## 変更点

- exp413 `nested compact74` を final matrix から除去する。
- 同じ位置へ保存済みexp501 `compact77`を挿入する。
- exp413 `clean273`と`signed23`は保持する。
- final幅は`273 + 77 + 23 = 373`。
- old74とnew77の併存、concat、blend、gateは行わない。

## 検証方針

- Fold: exp263由来outer 5 folds。exp413 / exp501 fold manifest SHAはともに
  `fa41084c5fcb4adffb88d44211b4cc5d2d2f46b5bd4d65828b6af941184b2a6d`。
- Group: `well_id`。
- Treatment: 1 variant、exp413 LightGBM configs 3、outer folds 5、15 GPU boosters。
- Control: 保存済みexp413 OOF RMSE `7.884802794404715`、再学習0。
- Leakage Check: exp501 outer-trainはinner OOF、outer-validは4 inner model ensembleのみ。
  key/fold mismatch、duplicate、missingを0にする。
- Primary gate: gain `>=0.03 ft`、nonworse `>=3/5 folds`、固定5 scopeの悪化
  `<=+0.02 ft`、technical/leakage全PASS。
- Tail: by-well p95/worst/悪化well数をreport-onlyで必須出力する。

## 実行量契約

| 対象 | 承認済みの実行量 |
| --- | ---: |
| treatment variant | 1 |
| TVT LightGBM config | 3 |
| outer fold | 5 |
| 新規GPU booster | 15 |
| exp413 control再学習 | 0 |
| exp501 selector再学習 | 0 |
| exp413 signed selector再学習 | 0 |
| HMM / PF / Beam再実行 | 0 / 0 / 0 |

この範囲の正規train Notebook採用、package、Kaggle実行は2026-08-02に承認済み。
inferenceとsubmissionは未承認。

## 実行入口

- 学習 notebook: `exp502_exp501_fixed13_selector_replacement_on_exp413_train.ipynb`
- 推論 notebook: `exp502_exp501_fixed13_selector_replacement_on_exp413_inference.ipynb`
- canonical train notebookへcompact self-contained候補を採用。inferenceはplaceholderのまま。
- 別名`*_compact_selfcontained_train.py`をJupytextの編集元として維持する。
- version 1は`NvidiaTeslaT4`、internet disabledで15 / 15 modelsを完走。
- 完了後はlocal run flagと`run_on_push`をfalseへ封印する。

## 結果

| メトリック | 値 |
| --- | --- |
| exp413 saved control CV | 7.884802794404715 |
| exp502 CV | 7.882143903310376 |
| gain | +0.002658891 ft（必要+0.03 ftを未達） |
| nonworse fold | 3 / 5 |
| 最大scope delta | +0.140943998 ft（上限+0.02 ftを超過） |
| primary gate | FAIL_CLOSE |
| Public LB | 未実行 |
| Private LB | - |

## 所見

### 良かった点

- exp501 compactはexp413と同一fold manifestで、下流置換のfold契約を事前固定できる。
- selectorとcontrolを再学習せず、TVT 15 boosterだけで仮説を反証できた。
- technical checks、final373、15-model grid、SHA監査はすべてPASSした。

### 悪かった点

- pooled改善は`0.002659 ft`に留まり、fold 3 / 4は`+0.116027 / +0.234686 ft`悪化した。
- hidden-like spatial / typewell-purgedは`+0.139587 / +0.140944 ft`悪化した。
- report-only tailもp95 `+1.293098 ft`、worst `+8.159899 ft`だった。

### リスク / 注意

- exp501 compactはexp264 fixed12+exp490 candidate面、保持するclean/signedはexp413 scale5面であり、
  hybrid surfaceになる。これは意図した差分だが、追加再計算を混ぜない。
- exp501のFAILを維持し、exp502もscientific FAILとして閉じる。

## 次

same-OOF feature subset / blend / weight / threshold / gate救済は行わず、inferenceと
submissionへ進めない。原因説明が必要な場合だけ保存OOFによる低優先readoutを別承認で行う。

## 表記

用語は`KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせ、実験名や設定名を除いて
日本語優先で記録する。
