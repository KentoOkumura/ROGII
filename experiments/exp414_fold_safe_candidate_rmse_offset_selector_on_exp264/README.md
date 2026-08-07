# exp414_fold_safe_candidate_rmse_offset_selector_on_exp264

## 状態

- ルート: `ml_model`
- 状態: Stage B implementation-only完了、Kaggle未実行
- CV: 未実行
- Public / Private LB: 対象外
- 作成日: 2026-07-26
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 原因比較: `exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264`

## 仮説

候補別RMSEはcandidate taskの学習重要度として使うと、低重み候補の局所的に有用な
行まで共有木のgradient / splitから弱め、行ごとのscore surfaceと候補順位を
不安定化させる。一方、各fit partitionのRMSEを候補の大域的なexpected-error
baselineとして加算し、shared modelにはそこからの局所偏差だけをunweighted L1で
学習させれば、損失の重要度を変えずRMSEを利用できる。

## 変更点

- 各outer modelのexact sampled fit rowsだけで12候補のRMSEを計算する。
- 学習targetを
  `candidate_abs_error - fit_candidate_rmse`へ変換する。
- scoreを
  `max(0, residual_prediction + fit_candidate_rmse)`で再構成する。
- sample weightは全く渡さない。
- binary `p_within10` modelは学習しない。
- 親/control、exp407、candidate generatorは再学習・再生成しない。

## 検証方針

- Fold: 親と同じouter 5 GroupKFold
- Group: well
- Candidate: 親と同じ12候補、hard-selectionは同じ11候補domain
- Feature: 親と同じraw-test-safe 88列、logical SHA固定
- 計算量: 1 variant × 1 objective × 5 folds = CPU booster 5
- Root cause:
  parent/exp407保存OOFをcandidate×foldの平均shiftとrow-local変化へ分解する。
- Scientific gate:
  expected-error MAE、hard RMSE、fold、near、1000+、hidden-like 2面、
  worst well、row-local instabilityを親と比較する。
- Leakage:
  offsetはexact fit rowsだけから計算し、outer-valid / global OOF /
  hidden-like / current-test truthは入力しない。

## 実行入口

- 編集元候補:
  `exp414_fold_safe_candidate_rmse_offset_selector_on_exp264_compact_selfcontained_train.py`
- 変換済み候補:
  `exp414_fold_safe_candidate_rmse_offset_selector_on_exp264_compact_selfcontained_train.ipynb`
- 正規train Notebook:
  template placeholderのまま。明示承認なしに上書きしない。
- Kaggle Notebook実行を正とし、ローカルNotebook実行は行わない。
- Stage C、inference、submissionは実装・実行しない。

## 現時点の所見

保存OOFの事前解析では、exp407のcandidate×fold平均shiftだけなら
RMSE `8.580477`で親`8.587004`を悪化させなかった。一方、平均shiftを除いた
row-local変化だけでは`8.673599`となり、exp407`8.668141`の悪化を再現した。
したがって、原因は候補一律のbiasではなく、逆RMSE task weightingが作った
分散したrow-local score driftである。

RMSE offset treatmentの有効性はKaggle Stage B実行後にだけ判定する。

## 所見

### 良かった点

- exp407の悪化をcandidate定数biasとrow-local driftに分離できた。
- RMSEを使いながら元のunweighted L1を保つ単一手法へ絞れた。

### 悪かった点

- treatmentは未実行であり、現時点では方法の有効性を主張できない。

### リスク / 注意

- Stage B合格だけではcurrent-test inferenceやLB改善を保証しない。
- 結果を見た後のoffset scale、clip、candidate subset探索は禁止している。

## 次

1. canonical Notebook採用とprivate CPU Stage B実行の承認を得る。
2. 5 boosterを実行し、入力・offset・model・OOF・gate SHAを記録する。
3. 全科学gateを満たした場合だけ、RMSE additive offsetをStage Bで確立とする。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、
実験名や設定名を除いて日本語優先で記録する。
