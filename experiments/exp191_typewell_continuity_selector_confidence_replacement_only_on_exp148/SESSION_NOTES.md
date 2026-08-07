# exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148 セッションノート

## 2026-07-05 実装開始

ユーザー依頼:

- `exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148` backlog を実装する。
- CPU 実行で timeout 対策のため、学習コードを `lgb0`、`lgb1`、`lgb2` に分ける。

既に `exp191_typewell_late_range_continuity_selector_on_exp176` が存在するが、`KAGGLE_DIRECTION.md` の backlog がこの slug を明示しているため、親 exp191 continuity artifact を使う replacement-only 実験として同名番号の別 directory を作成した。

## 実験設計

- route: `ml_model`
- parent: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- continuity selector parent: `exp191_typewell_late_range_continuity_selector_on_exp176`
- candidate ranker parent: `exp176_typewell_late_range_pfbeam_candidate_prior`
- active variant: `exp191_continuity_selector_confidence_replacement_only`
- feature groups: `projection_correction`, `u_disagreement`, `exp191_continuity_selector_confidence`
- disabled control: `exp148_fulltrain_control`
- removed group: `learned_likelihood_confidence`
- direct selected TVT replacement / blend / postprocess / hard gate: なし

特徴量は exp191 best Viterbi OOF selected path と exp176 saved booster から再構築する predicted-error score surface を使う。`true_tvt`、`abs_error`、`oracle_candidate`、`oracle_label` は downstream model feature として読まない。初回 variant では raw `tlr191_selected_tvt` と selected-minus-exp148 は入れない。

## 学習コスト確認

- active variants: 1
- active modes: 1 (`cpu_deterministic_threads8`)
- LightGBM configs: 3 split notebooks (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- total boosters: 15
- parent/control 再学習: なし

各 split notebook は 1 config x 5 folds = 5 boosters のみを実行する。
各 split の output 内では `lgb_mean` は実質的にその split の 1 config と同じ予測になる。3 config ensemble の正式な `lgb_mean` は、`train_lgb0` / `train_lgb1` / `train_lgb2` 完了後に各 output の OOF prediction を `id` / `well` / `target_tvt` で align して別途集計する。

## 実装メモ

- `.steering/20260705-exp191-typewell-continuity-selector-confidence-replacement-only-on-exp148/` を作成。
- `experiments/exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148/` を exp194 replacement-only pattern から作成。
- helper を `typewell_continuity_selector_confidence_replacement_only_on_exp148.py` に変更。
- exp191 score reconstruction 用に `typewell_late_range_continuity_selector_on_exp176.py` を同梱。
- 正規 train notebook は split plan 表示用。実学習は `exp191_..._train_lgb0.py`、`train_lgb1.py`、`train_lgb2.py`。

## 実装時点の未実行事項

この時点では Kaggle train push、logs 監視、output 取得、CV 記録はまだ行っていなかった。

## 2026-07-05 実装検証

Jupytext 変換:

- `exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148_train.py` -> `.ipynb`
- `exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148_train_lgb0.py` -> `.ipynb`
- `exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148_train_lgb1.py` -> `.ipynb`
- `exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148_train_lgb2.py` -> `.ipynb`
- `exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148_inference.py` -> `.ipynb`

検証:

- `jupytext --to ipynb --test` は train / train_lgb0 / train_lgb1 / train_lgb2 / inference すべて pass。
- `python -m py_compile` は helper、settings、train、train_lgb0、train_lgb1、train_lgb2、inference すべて pass。
- `ruff check --select F821` は同一対象ですべて pass。
- `make validate-exp EXP=exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148` は strict pass。

Kaggle package:

- `make prepare-kaggle-notebooks ... --notebook train_lgb0 --kernel-id kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb0 --title 'exp191 typewell continuity repl exp148 train lgb0' --run-on-push --strict` pass。
- `make prepare-kaggle-notebooks ... --notebook train_lgb1 --kernel-id kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb1 --title 'exp191 typewell continuity repl exp148 train lgb1' --run-on-push --strict` pass。
- `make prepare-kaggle-notebooks ... --notebook train_lgb2 --kernel-id kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb2 --title 'exp191 typewell continuity repl exp148 train lgb2' --run-on-push --strict` pass。

生成先:

- `experiments/exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148/kaggle/train_lgb0`
- `experiments/exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148/kaggle/train_lgb1`
- `experiments/exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148/kaggle/train_lgb2`

## 2026-07-05 Kaggle train v1 起動

事前確認:

- `make validate-exp EXP=exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148` は strict pass。
- Kaggle package metadata は `train_lgb0` / `train_lgb1` / `train_lgb2` すべて `enable_gpu=false`。
- active variant 1、active mode 1、各 split 1 LGB config x 5 folds = 5 boosters、合計 15 boosters。
- parent/control 再学習なし。

push:

- `kaggle kernels push -p experiments/exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148/kaggle/train_lgb0`
  - result: Kernel version 1 successfully pushed.
  - URL: https://www.kaggle.com/code/kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb0
- `kaggle kernels push -p experiments/exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148/kaggle/train_lgb1`
  - result: Kernel version 1 successfully pushed.
  - URL: https://www.kaggle.com/code/kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb1
- `kaggle kernels push -p experiments/exp191_typewell_continuity_selector_confidence_replacement_only_on_exp148/kaggle/train_lgb2`
  - result: Kernel version 1 successfully pushed.
  - URL: https://www.kaggle.com/code/kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb2

status:

- `kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb0`: 起動直後は実行中
- `kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb1`: 起動直後は実行中
- `kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb2`: 起動直後は実行中

Kaggle CLI は `installed: 2.2.0`、latest `2.2.2` の warning を出したが、push / status は成功した。CLI logs は実行中に空の可能性があるため、完了判定は status と Kaggle UI、完了後 logs を併用する。

## 2026-07-05 Kaggle train v1 完了

status:

- `kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb0`: `KernelWorkerStatus.COMPLETE`
- `kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb1`: `KernelWorkerStatus.COMPLETE`
- `kentookumura/exp191-typewell-continuity-repl-exp148-train-lgb2`: `KernelWorkerStatus.COMPLETE`

split metrics:

- `lgb0`: RMSE TVT 9.464292702215204、RMSE target 9.464292714534768、elapsed 20,752.006 sec、prediction SHA `bd1859c555da1a03403409c32e686dfb970d191753b14bee56cab800350e662d`
- `lgb1`: RMSE TVT 9.331742862377020、RMSE target 9.331742858173510、elapsed 17,484.660 sec、prediction SHA `0ff72745577dce2b705c26cebfe69cf1281b8b6b9dd4c37471ea0982a67d46b3`
- `lgb2`: RMSE TVT 9.313152705849234、RMSE target 9.313152573055143、elapsed 17,333.009 sec、prediction SHA `622aef9042653af53b7136dd33e633b06143cdf0ad3a0256ab6fde8733ed2d2f`

outputs:

- `kaggle/output/train_lgb0_v1`
- `kaggle/output/train_lgb1_v1`
- `kaggle/output/train_lgb2_v1`

3 split OOF prediction を `id` / `well` / `target_tvt` で align し、streaming chunk 集計で `lgb_mean_split3` を作成した。最初の全量同時 load は local aggregation memory issue で killed されたため、実験失敗ではなく集計方法を chunked に変更した。Kaggle run 自体はすべて COMPLETE。

`lgb_mean_split3`:

- rows: 3,783,989
- wells: 773
- features: 295
- RMSE TVT: 9.321908826194106
- RMSE target: 9.321909300319088
- MAE TVT: 5.9722260309015995
- within1 / within2 / within5 / within10: 0.17904544648517742 / 0.3240664811657751 / 0.6066122285239202 / 0.8190697700231159
- prediction SHA: `10f9d4ef35b2371c196a65ba7b533f70a5045dd90b4d096f211532e5bdb6c668`

比較:

- vs exp148 `lgb_mean` 8.50128118189582: +0.8206276442982858
- vs exp193 `lgb_mean` 8.456665438542778: +0.8652433876513275
- vs exp194 replacement-only `lgb_mean` 9.329893102424073: -0.007984276229967335

distance bucket RMSE:

- `000_050`: 1.1332989418492245
- `050_100`: 1.4087628312847422
- `100_250`: 2.206831820774099
- `250_500`: 3.507212864737638
- `500_1000`: 5.23423028478711
- `1000_plus`: 10.23001742147714

worst wells top5:

- `86454a6f`: RMSE 56.312288514840745
- `fb03ae90`: RMSE 40.03936893730723
- `1b1eba53`: RMSE 39.196409500135466
- `91b301ce`: RMSE 35.97889473145367
- `389ae58f`: RMSE 34.63016547647348

判定:

- `completed_negative_no_submit`
- exp191 continuity selector confidence block は exp148 の `learned_likelihood_confidence` (`ll_*`) block の代替にならない。
- current-test feature generation、inference port、submit は行わない。
- backlog は完了/不採用として外す。
