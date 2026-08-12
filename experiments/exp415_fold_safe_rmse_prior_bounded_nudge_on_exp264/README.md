# exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264

## 状態

- ルート: `ensemble`
- 状態: 保存OOF診断上の方法確立、実験完了
- Kaggle: private CPU version 1、`COMPLETE`
- 保存OOF: `8.587004 -> 8.563474`、`-0.023530 ft`
- Public / Private LB、inference、submission: 対象外
- 親: `exp264_exp263_candidate_confidence_dual_selector`
- 原因元: `exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264`

## 仮説

exp407で逆RMSEを共有selectorのsample weightにした結果、行単位のscore surfaceと
候補順位が崩れた。exp415は候補別RMSEを重みにせず、fold-safeな候補方向priorとして
使い、親TVTからその方向へ最大`0.25 ft`だけ補正する。

```text
parent_pos = argmin(parent_pred_abs_error)
prior_pos  = argmin(parent_pred_abs_error + fit_candidate_rmse)
correction = clip(0.5 * (prior_tvt - parent_tvt), -0.25, +0.25)
prediction = parent_tvt + correction
```

model / booster / candidate生成 / GPUはすべて0。保存済みcorrected exp264 OOFと、
exp407が保存したexact fit-partition候補RMSEだけを使う。

## 変更点

- inverse-RMSE sample weightを廃止し、RMSEを候補方向priorだけに使う。
- 親候補のTVTからprior候補方向への補正を各行`±0.25 ft`へ制限する。
- truth-free policy freezeとtruth-late evaluationを二相に分ける。
- overall、fold、距離、hidden-like、wellの全scopeで数学的risk boundを監査する。

## 検証方針

- overall `0.01 ft`以上改善
- fold 5 / 5、距離bucket 4 / 4、hidden-like 2 / 2でnonworse
- worst-well悪化`+0.25 ft`以下
- technical contract、truth-read ledger、risk inequalityを全PASS

## 確認結果

Kaggle kernel
`kentookumura/exp415-fold-safe-rmse-prior-bounded-nudge-train`
version 1（id_no `128717911`）を`126.338 sec`で完了した。

- technical gate: 15 / 15 PASS
- scientific gate: 6 / 6 PASS
- overall improvement: `0.023530 ft`
- fold: 5 / 5 改善
- 距離bucket: 4 / 4 改善
- hidden-like: 2 / 2 改善
- worst-well悪化: `+0.171379 ft <= +0.25 ft`
- 任意scopeの数学的risk bound: 785 / 785 PASS

これにより「候補別RMSEをfold-safe additive priorとして使い、補正量を
数学的に上限化する」方法を保存OOF診断上で確立した。current-testやLBへの
一般化は未確認であり、この実験から推論・提出へは進めない。

## 所見

exp407の問題はRMSEという統計量ではなく、その使い場所だった。共有木の学習重みに
すると局所scoreを壊したが、候補方向を示すpriorに限定すれば全fold・全補助scopeで
改善した。capにより、効果が一般化しないscopeでも悪化幅を実行前に上限化できる。

## 実装

- 編集元:
  `exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264_compact_selfcontained_train.py`
- 正規Notebook:
  `exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264_train.ipynb`
- reusable helper:
  `src/candidate_rmse_bounded_nudge.py`
- test:
  `experiments/exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264/tests/test_exp415_candidate_rmse_bounded_nudge.py`
- 監査済み小容量output:
  `kaggle/output/train_v1_small/`

truth-free phaseでpolicy freezeとSHAを確定し、その後のevaluation phaseだけで
truthを読む。input、freeze、prediction、metrics、gate、risk certificateのSHAは
`metrics.json`と`result.md`に記録した。

## 検証

- 専用tests: 8 PASS
- 関連tests: exp264 / 414 / 415をrepo rootから31 PASS、
  exp407を自身の実験cwdから9 PASS、合計40 PASS
- py_compile / Ruff / Jupytext round-trip: PASS
- strict experiment / project validation: PASS
- ダウンロードした小容量artifactのmanifest SHA監査: PASS

全体`make test`は今回と無関係なexp408のnumba stubとexp411の`find_spec`が
collection時に衝突する既知エラーで停止した。exp415由来のfailureは0。
4実験をrepo rootから一括実行した場合も、未変更のexp407がrootの別実験用
`config.yaml`を優先する既知問題で1件失敗するが、exp407自身のcwdでは9件PASSした。

## 次

exp415は診断実験として完了する。current-testへ展開する場合は、5 fold modelの
scoreとfit-partition RMSE priorのensemble順を固定する別設計から始める。
