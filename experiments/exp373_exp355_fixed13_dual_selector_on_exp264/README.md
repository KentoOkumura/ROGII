# exp373_exp355_fixed13_dual_selector_on_exp264

## 状態

- ルート: ensemble
- 状態: Kaggle CPU train version 1完了 / scientific gate FAILで閉鎖
- CV: 8.695437630439221
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-24
- 親実験: exp264_exp263_candidate_confidence_dual_selector

## 仮説

平均では改善するがwellごとの得失が大きいexp355 direct HMMを、既存HMMと置換せず
13番目の候補として残せば、dual selectorが有効区間だけを選択できる。

## 変更点

- corrected exp264 fixed12へ`exp355_dip_rate_hmm`を1本追加した。
- 既存`exact_hmm`、その派生pair、fixed formula、fixed fallback 7本は維持する。
- ユーザー判断によりadd-one novelty監査は先行させない。
- exp355 OOFは`well_id,row_idx,fold,candidate_tvt`だけを読み、
  global key join後にexp263 selector foldへrepartitionする。
- Kaggle CPU trainはversion 1を完了し、inference、submissionは行っていない。

## 検証方針

- Fold: outer 5 × inner 4
- Group: well
- Stratification: exp263 selector foldを維持
- Leakage Check: exp355 source foldはprovenance-only、global key join後にselector foldへrepartition
- 学習量: 1 variant / 2 objectives / 5 outer / 4 inner / 40 CPU boosters
- 親/control再学習: 0

## 実行入口

- 学習 notebook: `exp373_exp355_fixed13_dual_selector_on_exp264_train.ipynb`
- 推論 notebook: `exp373_exp355_fixed13_dual_selector_on_exp264_inference.ipynb`
- 実装済み学習候補:
  `exp373_exp355_fixed13_dual_selector_on_exp264_compact_selfcontained_train.ipynb`
- 実装済みfail-closed推論候補:
  `exp373_exp355_fixed13_dual_selector_on_exp264_compact_selfcontained_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp373_exp355_fixed13_dual_selector_on_exp264`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

2026-07-24のユーザー承認により、compact self-contained版を正規trainへ採用し、
fail-closed inferenceも正規名へ反映した。

## 結果

| メトリック | 値 |
| --- | --- |
| fixed13 hard OOF RMSE | 8.695437630439221 |
| 親fixed12 hard OOF RMSE | 8.652531955610227 |
| 差 | +0.04290567482899377 |
| fixed fallback OOF RMSE | 8.238331546485645 |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- candidate順序、77 compact、fixed fallback不変、exp355 allowlist/SHA、
  selector-fold repartition、readoutを専用testで固定した。
- 専用9 tests、exp371/exp264/exp355を含む回帰51 tests、py_compile、Ruff、
  Jupytext、strict experiment validationを通過した。
- 40 models、25 partitions、18,919,945 compact rows、49,191,857
  candidate-score rowsを生成し、score guardとleakage auditをPASSした。
- exp355はpooled 12.3192%・5/5 foldsでtop-1利用され、hidden-like 2面では
  親より`-0.137525 / -0.125127 ft`改善した。

### 悪かった点

- pooledは親より`+0.042906 ft`悪化し、改善は2/5 foldsだった。
- by-well p95は`+1.008261 ft`、worst well `b19b0395`は
  `+29.062587 ft`悪化し、scientific gateをFAILした。

### リスク / 注意

- 内部selector指標とusageが良くても、hard TVT readoutとwell tailの安全性を
  保証しない。
- scientific gate FAILを保持し、同一OOFでのweight/threshold/domain救済はしない。

## 次

- `FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`としてexp355固定13枝を閉じる。
- downstream TVT、inference、submissionへ進めない。
- 独立候補のexp375は別仮説・別結果として扱う。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
