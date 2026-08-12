# 設計

## 仮説

exp226 `tvt_geop`の局所形状が概ね正しく、失敗の主要因が低周波offsetなら、fixed raw-GR /
typewell likelihoodは真値に最も近い縦shiftを5 foldsでstable shuffledより良く順位付けできる。
この識別力がなければ、offsetだけを状態にする後続HMMへ進む根拠はない。

## アプローチ

各wellのexp226 group-safe OOF `tvt_geop(t)`に固定shift `delta`だけを加える。

```text
candidate_tvt(t, delta) = tvt_geop(t) + delta
delta in [-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80]
```

raw GRを`g_t`、candidate TVT上のtypewell GRを`mu(t, delta)`、known-prefix residualから
固定したscaleを`sigma`とし、exp209と同じrow scoreを作る。

```text
ll(t, delta) = -0.5 * min(((g_t - mu(t, delta)) / sigma)^2, 600)
block_score(delta) = mean_t(ll(t, delta))
```

unknown suffixは先頭から非重複512行blockへ分け、末尾short blockも保存する。target-free
score tableを全wellで作成してcontent SHAを凍結した後にだけtrue TVTを結合する。truth-nearest
shiftはblock SSEが最小の候補とし、そのlikelihood rank、top1/top3、MRR、符号一致、margin、
top1 regret、bank range/quantization coverageをreadoutする。

negative controlは各blockの13 candidate scoreだけをstable permutationする。truth label、block、
fold、score分布は固定し、real/shuffledのtop1/top3/MRR/signを同じ関数で比較する。

## 実験範囲

- 対象実験: `exp280_exp226_shift_likelihood_separability_readout`
- Route: `pf_beam`
- 科学的親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 失敗根拠: `exp279_exp226_geop_centered_exact_hmm_redecode`
- 後続候補: `exp226_residual_offset_exact_hmm_transition_probe`
- 変更する変数: exp226 `tvt_geop`の周囲に固定shift bankを作り、raw GR likelihoodだけを読む。
- 固定する変数: exp226 OOF/fold/SHA、raw/typewell data、shift 13候補、512行block、
  exp209 Gaussian emission、missing処理、tie policy、stable shuffle、scope、guard。
- 実行量: audit variant 1、LightGBM config 0、trained fold 0、booster 0、HMM well-run 0。
- 除外: exp226再生成、HMM/PF、candidate prediction保存、補正、blend、selector、inference、submit。

## 再現性設計

- seed policy: real scoreはRNGなし。shuffled controlだけを
  `SHA256(experiment, seed, well, block)`から作るlocal `np.random.default_rng`で固定する。
- stochastic 処理の有無: stable shuffled negative controlだけ。global RNGとPython `hash()`は禁止。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規実行なし。exp226保存OOFのみを読む。
- 並列処理と乱数の関係: single process、well文字列昇順、block/shift固定順。
- CPU/GPU runtime: Kaggle CPU、GPU/TPU/internet off。model fittingなし。
- train cache SHA: exp226 gzipはdecompressed content SHAをhard guardする。raw horizontal/typewellと
  hidden-like assignmentはraw SHA、score/readout gzipはraw/decompressed SHAを記録する。
- model/prediction/submission SHA: fitted model、最終prediction、submissionは生成しない。
  scientific contract manifest SHAとtarget-free score content SHAをmodel/prediction代替証拠とする。
- Kaggle bootstrap: push承認後にcanonical packageをprepareし、source/loose/bootstrap内configと
  train sourceをbyte比較する。今回の実装作業ではpushしない。
- deterministic anchor: prediction/submission anchorではなく、固定exp226/raw入力に対する
  deterministic diagnosticだけを主張する。

## リスク

- リークリスク: exp226 OOFからsafe列だけを`usecols`で読み、score APIでtruth/error列をrejectする。
  true TVTの再読込はscore table完成とcontent SHA確定後に限定する。
- GR二重利用リスク: exp226 `tvt_pred`と`gr_delta`を禁止し、geometry-only `tvt_geop`だけを使う。
- CV/LB不一致リスク: train-side separability診断であり、raw-test生成やLB判断を行わない。
- multiplicityリスク: shift/grid/calibration/score/guardは1契約だけ。結果後のbest scoreやfold選択をしない。
- coverageリスク: bank外のcontinuous optimal offsetをrange coverage、bank内量子化誤差、oracle gainで分離する。
- ランタイム/メモリリスク: 3.78M x 13 row scoreはwell単位で生成・block集約し、row x shift long tableを常設しない。
- 再現性リスク: gzip metadata差を避けmtime=0で保存し、decompressed content SHAを主証拠とする。

## 次

Kaggle CPU readout後、top1/top3/MRR/signの4指標がstable shuffledを5/5 foldsで上回る場合だけ、
別実験のresidual-offset HMMを検討する。FAIL時はshift/grid/calibration救済なしで優先度を下げる。
