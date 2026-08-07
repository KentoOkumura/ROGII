# 要件

## 依頼

exp284 で使った same-well self-GR 候補を完全に除外し、visible prefix の
Type Well GR shift likelihood に複数の局所極大がある場合は、そのすべてを
safe base とともに候補として保持したまま future checkpoint へ進める仮説を、
実装前の固定 backtest として設計する。

対象実験は `exp291_prefix_masked_typewell_gr_multimode_safe_beam` とする。

## 2026-07-19 実装承認

ユーザーの `exp291を実装してください` を、固定済みcontractの実装と静的検証の承認として扱う。
Kaggle CPU実行、canonical notebookへの採用、推論、submissionの承認には広げない。

## 2026-07-19 実行承認と結果

ユーザーの `実行してください` をcanonical train notebook採用と固定contractのKaggle CPU 1回実行の
承認として扱った。推論・submissionには広げなかった。version 1は766 eligible wells / 5 foldsで
technical guardを全PASSしたが、H256 all-mode RMSE 22.199818対safe 4.827483、false switch 34.9462%、
matched shuffle比も悪化してscientific / safety guardをFAILした。停止条件どおりparameter rescueなしで
branchを閉じる。

## 仮説

exp284 の悪化要因は、safe base の欠落ではなく、same-well self-GR が作る偽 branch と
visible 時点で Type Well GR mode を1本に早期確定したことにある。self-GR を使わず、
固定 shift bank 上の eligible な Type Well GR 局所極大を全件保持し、safe base を
絶対に prune せず、複数 checkpoint で同じ代替 mode が safe を継続して上回った場合だけ
切り替えれば、false switch を抑えつつ exp226 safe base を改善できる。

## 制約

- Route は `pf_beam` とする。ただし本実験は decoder 実装ではなく、既知 prefix を使う
  train-side の deterministic masked backtest である。
- 親実験は `exp284_prefix_masked_wrong_mode_branch_recovery_backtest` とし、
  exp280 の固定 shift bank、exp209 の Type Well GR emission、exp226 の group-safe geometry を参照する。
- same-well self-GR、NCC、donor window、orientation、self-GR shuffle は候補生成にも evidence にも使わない。
- safe base は全 event、全 checkpoint、全 policy で必ず残し、prune、平均化、softmax 混合しない。
- 代替候補は固定13 shift bankのうち `abs(shift) >= 10 ft` で、visible score が隣接 slot 以上の
  局所極大すべてとする。端点は片側だけと比較し、tie は shift bank 順で固定する。
- eligible な局所極大が0件なら代替候補を強制作成せず safe-only とする。
- 候補数上限、top-K、margin、window、horizon、shift bank、likelihood、veto を同一結果上で調整しない。
- post-cut true TVT は mask、candidate、branch path、checkpoint evidence、policy selection の
  content SHA を凍結するまで参照しない。
- active contract 1件、policy 4件、LightGBM config 0件、学習 fold 0件、booster 0本、
  HMM/PF 再生成0件、親/control再学習なしとする。
- Kaggle 実行、notebook 実装、推論、submission は本設計作業の範囲外とする。
- 再現性は `docs/06_reproducibility.md` に従う。実 policy は deterministic、
  matched-count shuffled negative control だけ stable SHA256 local RNG を使う。

## 受け入れ基準

### 設計確定

- `.steering/`、実験ディレクトリ、`config.yaml`、`SESSION_NOTES.md`、`result.md` に、
  仮説、固定候補契約、safe保持契約、truth freeze、guard、停止条件が同じ意味で記録されている。
- exp284 の単なる K 変更ではなく、候補源を Type Well GR local modes のみに変え、
  same-well self-GR を完全除外する別仮説として lineage が記録されている。
- `experiment_summary.md` と `KAGGLE_DIRECTION.md` に未実装の設計済み実験として記録されている。
- notebook/helper/test/Kaggle package を実装していない。

### 将来の実行時に固定する technical guard

- eligible well が750以上で、fold 0〜4がすべて存在する。
- mask identity、safe candidate 保持、全 local-mode 保持、branch finite、evidence finite の coverage が1.0。
- candidate/evidence/selection freeze 前の post-cut truth access が0件。
- self-GR由来候補が0件である。
- candidate 数は safe 1本 + 固定 bank 上の eligible local maxima 数に一致する。

### 将来の実行時に固定する scientific guard

- H256 の全-mode policy が safe-only より pooled RMSE で0.10 ft以上改善し、5 fold中4 fold以上で改善する。
- H256 の全-mode policy が top1-mode policy より pooled RMSE で0.05 ft以上改善し、5 fold中3 fold以上で改善する。
- safe と real alternative の pairwise Type Well evidence AUC が各 fold 0.60以上、
  pooled balanced choice accuracy が0.60以上、各 foldの balanced choice accuracy が0.50を超える。
- safe が truth unique-best の event における false switch rate が5%以下。
- H512 の safe比 gain が H256 を下回らない。
- matched-count shuffled modes に対し pooled で改善し、5/5 folds で非悪化。
- いずれかを満たさなければ parameter/K/window/shift/horizon/margin rescue をせず close する。

この backtest が全 guard を通っても、それだけで decoder 実装、raw-test inference、提出を許可しない。
次段階は別途ユーザー承認を得る。

### 実装受け入れ基準

- Jupytext percent形式の別名compact self-contained train/inference sourceとnotebookが存在する。
- train notebook内でexp226 fold-safe geometry replay、固定shift bank、local-mode全件保持、
  matched-count control、persistent checkpoint commit、truth freeze、post-freeze guardを追える。
- same-well self-GR候補生成関数を実装せず、candidate/evidence source guardでも侵入を拒否する。
- synthetic contract test、`py_compile`、Ruff、Jupytext round-trip、strict experiment validation、
  repository testが通る。
- 既存canonical notebookは採用判断なしに上書きせず、Kaggle packageも生成しない。
