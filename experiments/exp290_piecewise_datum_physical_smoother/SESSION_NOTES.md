# exp290_piecewise_datum_physical_smoother セッションノート

## 目的

exp226のgroup-safe geometryを内部基準にし、known-prefix calibration、robust Type Well GR likelihood、
Type Well/近隣の不確実性事前から、各wellのbounded piecewise datumを一つのsemi-Markov posteriorとして
推定する単独物理モデルを設計する。Public LB 6.5を目標にするが、今回はStage 0だけを実装・実行し、
Stage 1、inference、submissionは行わない。

## 現在の状態

- Route: `pf_beam`
- 状態: Stage 0 scientific guard FAIL・branch closed（kernel version 1）
- Stage 0 pseudo-tail RMSE: base 1.436926 / model 1.403407
- LB: まだなし
- 実装承認: 2026-07-19 ユーザー依頼「exp290を実装してください」
- Kaggle push承認: 2026-07-19 ユーザー依頼「実行してください」

## 今回確定した契約

- parent geometry: exp226 group-safe `tvt_geop`
- model: `TVT=g+delta`の階層semi-Markov physical smoother 1本
- datum: absolute `[-15,+15] ft`、0.5 ft grid、非累積、minimum duration 256 rows
- calibration: known prefix末尾から512/256/128 rows戻した3 cutの直後128-row window。Stage 0では過去cutだけ、final suffixでは3 cutすべてからreliabilityを作る
- Type Well/neighbor: datum meanやjump signではなくscale/hazard/noise事前だけ
- inference: exact log-space forward-backward、posterior meanのみ
- Stage 0: known-prefix pseudo-tail audit
- Stage 1: Stage 0通過・別承認後のdirect OOF
- forbidden: ML、candidate bank、Viterbi/hard shift、blend、selector、posthoc offset、oracle

## 実行規模確認

今回実装した実行契約は次のとおり。

- active Stage 0 audit contract: 1
- LightGBM config: 0
- trained fold: 0
- booster: 0
- parent/control再学習: 0
- Kaggle run: Stage 0を1回（承認済み、CPU）
- inference/submission: 0/0

実行量はouter-valid pseudo-tail solveが`773 wells x 3 cuts = 2,319`、
outer-train calibration replayが5-fold合計`3,092 well-folds x 3 cuts = 9,276`、
exp226 geometry replayは合計11,595 well-windowである。ML学習・booster・親control再学習はない。

`config.yaml`と専用testで、contract 1、ML config 0、trained fold 0、booster 0、control再生成0を再確認した。

## コマンドログ

2026-07-19:

- `make new-steering EXP=exp290_piecewise_datum_physical_smoother`相当でsteeringを作成した。
- steeringの要件、数理モデル、guard、再現性、禁止事項を記入した。
- `make new-exp EXP=exp290_piecewise_datum_physical_smoother`でtemplate scaffoldを作成した。
- 追加依頼をStage 0実装承認として記録し、`config.yaml`を`stage0_implemented_not_run`へ更新した。
- `exp290_piecewise_datum_physical_smoother_compact_selfcontained_train.py`をJupytext percent形式で実装した。
- exp226 geometry replay、outer-train hyperprior calibration、Type Well Huber affine、stable spatial k=16、305 expanded-state exact forward-backward、truth-after-freeze metricsをself-contained sourceへ実装した。
- `exp290_piecewise_datum_physical_smoother_compact_selfcontained_inference.py`をfail-closedで実装した。
- `tests/test_exp290_piecewise_datum_physical_smoother.py`に専用test 11件を追加した。
- Jupytextでcompact train/inference notebookを新規生成した。既存の正規名template notebookは上書きしていない。
- `py_compile`、`ruff check`、専用`pytest`、Jupytext `--test`、`make validate-exp EXP=exp290_piecewise_datum_physical_smoother`を実行した。
- `make validate-template`をPASSし、`make update-summary`で`experiment_summary.md`を更新した。
- 親exp226にはcompact self-contained版がないため、関連するexp285 compact trainと比較した。exp285は1,902行/9章、exp290は2,427行/9章で、runtime/config、geometry、pseudo-cut/hierarchy、solver、freeze/metrics、orchestrationをnotebook上で追える。

- 追加依頼「実行してください」を、compact self-contained trainの正規train notebook採用とStage 0 Kaggle CPU pushの明示承認として記録した。
- canonical packageを`kentookumura/exp290-piecewise-datum-physical-smoother-train`へpushし、Kaggle kernel version 1を開始した。
- push直後に同じkernel IDを`kaggle kernels pull ... -m`で取得し、`id_no=127881061`、CPU、GPU/internet off、exp226 kernel sourceを確認した。
- 同じcanonical IDを監視し、kernel version 1が`COMPLETE`となったことを確認した。空ログ中の再pushやslug変更は行っていない。
- 完了ログで296,832 rows / 773 wells / 3 windows / 5 folds、finite prediction 1.0、truth-after-freeze 2,319 hashes、bound violation 0を確認した。
- Stage 0はbase RMSE 1.4369259414からmodel RMSE 1.4034066170へ0.0335193244 ft改善し、5/5 foldsで改善した。
- correction sign accuracyは0.4831112488、well RMSE p95は2.4371833720から2.4400096405へ微悪化し、固定scientific guardはFAILした。
- runtime 587.041739秒、peak RSS 1,215.824 MB、Kaggle 9時間上限へのmargin 8.836933時間でtechnical guardは全PASSした。
- SHA実ファイル確認が必要なためoutputを`/tmp/kaggle-output/exp290_piecewise_datum_physical_smoother/train_v1`へ一度だけ取得した。
- 取得predictionのraw gzip / decompressed SHAとstate-space manifest SHAを再計算し、Kaggle summaryと一致した。

Stage 1、inference、submissionは引き続き実装・実行しない。

## 再現性メモ

- seed policy: canonical exact solverはRNGなし。fold/well/row/typewell/neighbor/stateをstable順で処理する。
- stochastic components: なし。PF particles、seed bagging、random pseudo-cutを使わない。
- CPU/GPU runtime: CPU float64、single process、BLAS thread 1、GPU/internet off。587.041739秒、peak RSS 1,215.824 MB。
- Kaggle kernel id / version: `kentookumura/exp290-piecewise-datum-physical-smoother-train` version 1、id_no `127881061`。
- input manifest content SHA: `64ad13d9addc5131cb4c4a51607fb621f36f959da3e8b039961853e2c029506f`
- fold map SHA: `353da574012a955fc99586bb08e2dd52258daf94df7ae7cf37de2fe79ead8b01`
- pseudo-cut content SHA: `5dd42c2ee7abf1190ae0b5b70f49db9d0d27f5a86d8945b4ff8b6d532649caad`
- hyperprior content SHA: `ba69f0a69d11bab1ddc22d96bb5a06b5501acd5f49548e8b575cf91586d02b9a`
- spatial neighbor content SHA: `3a41ed592e8713fcb9e56dce3454e2e9964dc98dd0ab5e34cc8375b755716197`
- state-space manifest SHA: `a004699db75078f0becba80c874e65e172eaf3d4f3c5f184a9dab3cc43c7b224`
- prediction raw gzip SHA: `7312ca1e85ae3b1d4e0e03158ca16e6c9305e8780befb7b9540056f55a4b3a0a`
- prediction decompressed SHA: `b0a598c82bb083b61a1c445752caf90e908899bb15afcbb45658f3baf9a3f956`
- submission SHA: Stage 0ではsubmissionを作らない。
- rerun check: 未実施。deterministic anchorはfalse。

## 次のアクション

1. exp290 branchを固定failure policyどおり閉じる。
2. parameter/group/neighbor/likelihood救済、Stage 1、raw-test inference、submissionへ進まない。
3. 本結果だけを根拠に新しい救済backlogは追加しない。
