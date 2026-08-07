# ROGII discussion insights update

取得日: 2026-06-03

対象:

- Kaggle competition discussion の `recent` / `top` / `active` 上位トピック。
- 重要そうな recent topic は本文を `kaggle competitions topics show` で確認。
- 個別全文アーカイブではなく、実験設計と提出運用に使うための差分要約。

取得コマンド:

```bash
kaggle competitions topics list rogii-wellbore-geology-prediction --sort-by recent -v
kaggle competitions topics list rogii-wellbore-geology-prediction --sort-by top -v
kaggle competitions topics list rogii-wellbore-geology-prediction --sort-by active -v
kaggle competitions topics show rogii-wellbore-geology-prediction 704001
kaggle competitions topics show rogii-wellbore-geology-prediction 703883
kaggle competitions topics show rogii-wellbore-geology-prediction 703867
kaggle competitions topics show rogii-wellbore-geology-prediction 703344
```

## 重要結論

1. `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` と typewell `Geology` は hidden test で使える列として期待しない。train-only として扱う。
2. Notebook-only hidden rerun では、commit 時の visible 3 wells と submit 時の hidden 約 200 wells で実行負荷と sample shape が変わる。visible rows 固定の CSV writer は危険。
3. 提出前には shape / id order だけでなく、id merge、per-well range、prediction start 近傍の連続性、smoothing 前後の挙動、候補間距離を見る。
4. 直近の議論でも、単一 global offset より segmented alignment、lateral GR self-correlation、typewell/time-warp matching、neighbor/dip consistency、PF/beam/Viterbi path search が有望という見方は変わらない。
5. Transformer / sequence model を使う場合、bfloat16 で copy task すら壊れた報告がある。sequence model の sanity check は float32 で通してから mixed precision を使う。

## 新規・更新トピック

### 704001: formation columns / Geology availability

質問:

- `EGFDU`, `ANCC`, `ASTNL`, `ASTNU`, `EGFDL`, `BUDA`, `Geology` が hidden test set で使えるか。

確認内容:

- Chris Deotte のコメントでは、これらは train data only と説明されている。
- 既存方針どおり、直接 test feature として扱わない。使う場合は train wells から fold-safe に impute / plane-fit する。

影響:

- `exp009_formation_surface_guide` の悪化後も、formation idea 自体を捨てるのではなく、fold-safe imputer と uncertainty / divergence として再検討する。
- typewell `Geology` も hidden availability を前提にしない。

### 703883: reproducing and stress-testing notebooks

主な実務メモ:

- notebook は `/kaggle/input/competitions/...` から毎回 prediction を再構築し、root-level `submission.csv` を書く。
- row position ではなく id merge で sample と結合する。
- visible rows の候補間距離は smoke test として有効。極端に近い候補は重複、極端に遠い候補は broken state / misalignment の疑いがある。
- GR NaN、interpolation gap、smoothing、`nan_to_num`、positional writing が重なると、shape は正しいが地質的に壊れた smooth curve が出る。
- per-well range、prediction start 近傍の連続性、smoothing 前後の挙動を確認する。
- modeling では single global offset より segmented alignment、self-correlation、typewell matching、neighbor/dip consistency、PF/beam/Viterbi path search が妥当。

影響:

- `task submit-check` の前段で、candidate output 同士の pairwise distance と per-well continuity check を追加する価値が高い。
- 実験 summary には public score だけでなく、submission writer の hidden-compatible 設計を記録する。

### 703867: notebook timeout on submit

確認内容:

- Commit run は fake / visible 3 test wells のみを infer する。
- Submit run では hidden 約 200 wells に差し替わるため、commit が数分でも submit が timeout し得る。

影響:

- PF 128/256 seeds、TabICL、beam、multi-scale DTW は hidden runtime で見積もる。
- `dateRun` recent の PF variants は public score 以前に、hidden 200 wells での runtime / memory を必ず見る。

### 703344: transformer and bfloat16

確認内容:

- 投稿者は copy transformer で input seq = target TVT の sanity check を行い、bfloat16 が原因で transformer が壊れることを確認した。

影響:

- CNN / Transformer / sequence model を試す場合、最初の copy / identity / short-fold sanity check は float32 固定。
- mixed precision は最後に切り替え、OOF と output hash の差を見る。

### 703830: missing key discovery

内容:

- LB 低下が続いているが、具体的な新情報はない。

影響:

- 方針変更なし。public notebook 側の実体からは PF/physical/beam/artifact stack family が伸びている。

## 参照トピック

- Formation / Geology hidden availability: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/704001
- Practical notes from reproducing notebooks: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703883
- Notebook timeout run: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703867
- Transformer bfloat16 caution: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703344
- Geology in typewell files in test: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703558
- Difference in train and test files: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703532
- Battle for LB 9.0: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703038
