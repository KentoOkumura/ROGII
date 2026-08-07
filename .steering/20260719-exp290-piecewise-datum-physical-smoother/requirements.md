# 要件

## 依頼

物理モデル単独で Public LB 6.5 を狙う高リスク案として、exp226 の group-safe 幾何場を
固定基準座標にし、各 well の既知 `TVT_input` prefix、Type Well GR、近隣 well から得る
不確実性事前だけを用いて、評価 suffix の piecewise datum を一つの semi-Markov smoother で
事後推定する実験を設計する。2026-07-19 の追加依頼では、確定済み契約を変えずに Stage 0 の
compact self-contained train notebook、専用テスト、fail-closed inference notebook まで実装する。
Kaggle 実行、Stage 1、raw-test inference、提出は別承認まで行わない。

## 制約

- Route: `pf_beam`。LightGBM/CatBoost/XGBoost/NN、learned selector、blend、別モデルによる postprocess を予測生成に使わない。
- 単一モデル: 出力は一つの階層 semi-Markov 物理モデルの posterior mean だけとする。粒子、state、mode は同じ確率モデルの数値積分に限り、候補曲線 bank、best-of-N、hard/soft selector、モデル平均は作らない。
- 基準場: group-safe exp226 と同じ手順で生成する `g_w(t)=tvt_geop_w(t)` を一つのモデル内部の固定幾何場として使う。保存済み exp226 prediction を別 anchor として blend したり、well ごとに採否選択したりしない。
- 潜在量: `TVT_w(t)=g_w(t)+delta_w(t)` とし、`delta_w(t)` は絶対補正量 `[-15,+15] ft` の範囲で piecewise constant とする。jump を累積する自由 random walk は禁止する。
- well 固有調整: known prefix から initial datum、GR affine calibration、likelihood temperature、shrinkage reliability を推定する。official suffix の true TVT/error は使わない。
- Type Well / 近隣 well 調整: outer-train と known prefix から datum scale、reset hazard、GR noise の階層事前だけを推定する。group/neighbor の平均 bias 符号を直接足すことは禁止する。
- 観測: outer-valid / raw test では `MD/X/Y/Z/GR/TVT_input` と対応 Type Well の `TVT/GR` だけを使う。formation 6列、true suffix TVT、target error、oracle rank は prediction freeze 前に使わない。
- inference: log-space exact semi-Markov forward-backward を canonical とし、posterior mean を全評価 row に一意に出す。Viterbi path、hard shift、GR top-1 shift を提出予測にしない。
- per-well empirical Bayes は known prefix 内の固定 pseudo-cut `128/256/512 rows` と各cut直後の固定128-row validation windowだけで行う。各pseudo-tail予測時は時間的に前のcut結果だけからreliabilityを作り、final suffixでは3 cutすべてを使う。official suffixを見たwell別weight、clip、hazard、温度選択は禁止する。
- 初回 scientific contract は一つだけとし、offset grid、duration、jump scale、GR sigma、typewell/neighbor weight の同一 OOF grid 探索を行わない。
- exp226/280/281/285 の artifact は設計根拠・比較・固定 fold identity に限って使う。exp281 HMM prediction を初期値、候補、補正、blend に使わない。
- oracle は診断根拠に限定し、row/segment/block/well oracle、truth-nearest shift、oracle prediction 保存、oracle 値による promotion を禁止する。
- 再現性: `docs/06_reproducibility.md` に従い、fold、base geometry、prefix pseudo-cut、hyperprior、state grid、OOF/prediction の schema/content SHA を記録する。
- Kaggle Notebook 実行を正とし、ローカル実行はユーザーが明示承認した smoke debug 以外では行わない。
- 実装前に active variant 1、ML config 0、trained fold 0、booster 0、control 再学習 0 であることを `SESSION_NOTES.md` へ再確認する。

## 受け入れ基準

### 初回の設計確定

- `.steering/20260719-exp290-piecewise-datum-physical-smoother/` に要件、設計、タスクリストがある。
- `experiments/exp290_piecewise_datum_physical_smoother/` に planned 状態の scaffold、`config.yaml`、日本語の記録がある。
- `KAGGLE_DIRECTION.md` の未着手 backlog に、exp289 と異なる仮説、先行 guard、停止条件、禁止事項を記録する。
- `experiment_summary.md` に planned 実験として記録する。
- 初回設計時点では notebook/source/test 実装、Kaggle package prepare/push/run、inference、submissionを行っていない。2026-07-19 の追加依頼で Stage 0 source/notebook/tests と disabled inference だけを実装した。

### Stage 0: known-prefix pseudo-tail 識別監査

- outer-valid well の formation/true suffix を除外し、known prefix 内だけに固定 pseudo-cut `128/256/512 rows` と各cut直後の128-row validation windowを作る。
- pseudo-cut より後の `TVT_input` を一時的に mask し、同一の固定 state grid、transition、GR likelihood、階層事前から一つの posterior mean を生成する。
- 各windowのwell reliabilityはそれより前のpseudo-cut結果だけから作り、未来pseudo-cutや評価window自身のtruthを使わない。official suffix用reliabilityはknown prefix内3 windowのfreeze後metricだけから作る。
- pseudo-tail truth を結合する前に prediction、entropy、reset probability、well reliability を freeze し content SHA を取る。
- 保存済み exp226 相当の基準場に対し、pooled pseudo-tail RMSE 改善 `>=0.20 ft`、`|base error|>=5 ft` の correction sign accuracy `>=0.58`、fold 改善 `>=4/5`、well RMSE p95 非悪化をすべて満たした場合だけ Stage 1 へ進む。
- Stage 0 不通過時は grid、clip、pseudo-cut、group、neighbor、GR likelihood の救済調整を行わず branch を閉じる。

### 今回の Stage 0 実装

- Jupytext percent 形式の compact self-contained train source / notebook を追加し、同一 exp 内 helper import に依存しない。
- exp226 fold map / fold別 kappa の SHA を固定し、outer-valid well を donor field、kappa fit、ANCC surface から除外する。
- outer-valid horizontal は `MD/X/Y/Z/GR/TVT_input` だけを materialize し、formation 6列と `TVT` を reader 段階で除外する。
- 各 pseudo-cut で exp226 geometry を cut の既知 `TVT_input` へ再 anchor するため、Stage 0 datum prior mean は厳密に0とする。Type Well / neighbor は scale、hazard、noiseだけを更新する。
- outer-train masked-prefix backtest から fold別 pooled / exact-Type-Well scale prior と target-free event threshold を freezeする。spatial k=16 は hyperprior variance だけを更新する。
- 128-row Stage 0 window は minimum duration 256 rows より短いため、expanded duration state は locked phase内に留まる。Stage 0 は reset性能ではなく bounded constant-datum / GR 識別性を反証する。
- prediction、entropy、reset probability、reliability を window ごとに SHA freezeし、held-known `TVT_input` はその後にだけ結合する。
- fail-closed inference notebook は Stage 1 別承認前の予測・提出を例外終了させる。
- `py_compile`、`ruff`、専用 pytest、Jupytext `--test`、strict `validate-exp` を通す。

### Stage 1: 単一 piecewise-datum direct OOF

- active variant は `hierarchical_piecewise_datum_posterior_mean` の1本だけ。保存済み exp226 OOF RMSE `9.4271095966` を比較基準とし、control を再生成しない。
- 予測対象全 row に一つの direct OOF prediction を生成し、coverage 1.0、fallback 0、finite 1.0、correction bound 違反 0 を満たす。
- raw-test shadow/inference 検討 guard は direct OOF RMSE `<=8.0`、exp226 比 `>=4/5 folds` 改善、1000+ と hidden-like spatial/typewell-purged の全てで非悪化、well RMSE p95 `<=15.0 ft`、worst-well RMSE `<=45.0 ft` とする。
- 物理モデル単独の inference 候補化 guard は direct OOF RMSE `<=7.0`、exp226 比 `5/5 folds` 改善、hidden-like 2面改善、well RMSE p95 `<=13.0 ft`、worst-well RMSE `<=40.0 ft` とする。
- guard 未通過では raw-test inference、submission、blend、selector、posthoc offset、parameter rescue を行わない。
- deterministic anchor として扱う場合は、base geometry / hyperprior / state-space manifest / OOF prediction / submission の各 content SHA と Kaggle kernel version を記録する。
- gzip 生成物を比較する場合は raw `.csv.gz` SHA ではなく decompressed logical content SHA を主証拠として記録する。
