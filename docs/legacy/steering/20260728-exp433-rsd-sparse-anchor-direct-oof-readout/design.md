# exp433 設計

## 結論

exp426を変更または再分類せず、凍結済みRSD scoreを実際のexp226 OOFへ
固定Viterbiで直接適用する、1-primary・0-modelの独立readoutを作る。

exp426は「全体へ密にscoreを供給できるか」というtechnical gateをFAILした。
exp433は別の問いとして、「疎なabsolute anchorでも実予測を改善するか」を
全OOF RMSEで判定する。coverage不足だけではtruth readを停止しない。

## 実験範囲

- 対象実験:
  `exp433_rsd_sparse_anchor_direct_oof_readout`
- Route:
  `pf_beam`
- 親実験:
  `exp426_rsd_binned_pattern_absolute_reanchor`
- 基準:
  exp226 final OOF `tvt_pred`、RMSE `9.427109596582213`
- 変更する変数:
  凍結済みRSD scoreを固定coarse-datum Viterbiへ通した補正だけ
- 固定する変数:
  score bank、512-row block、13 offsets、valid mask、score値、decoder、
  interpolation、全gate、fold、scope
- 作らないもの:
  model、HMM、PF、Beam、ML feature、inference、submission

## 入力契約

### exp426 target-free score

- Kaggle kernel:
  `kentookumura/exp426-rsd-binned-pattern-absolute-reanchor-train`
- version / id_no:
  `1 / 128930757`
- file:
  `exp426_rsd_binned_pattern_absolute_reanchor_stage_a_target_free_score_bank.csv.gz`
- rows:
  `101,231`
- blocks / offsets:
  `7,787 / 13`
- logical content SHA:
  `463aa32bef9a1045469466e2cf5fd68e038258e75f11fc88153fd9ca7f8dd2fd`
- decompressed CSV SHA:
  `6adb009b83c884681fa64e29c03bc05c6dac15d3bb6826df1a000961c8bbe575`
- schema SHA:
  `6e86f76bf7df038e5b3b8077db80888c737bfa3880377ac24fe74d236706e9bd`

well manifestとinput manifestもexp426 version 1から読み、それぞれlogical /
decompressed content SHAを照合する。score、valid、rank、top-3は再生成しない。

### exp226 OOF

- file:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train_oof_predictions.csv.gz`
- decompressed SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- inventory:
  `3,783,989 rows / 773 wells / folds 0..4`
- columns used before prediction freeze:
  `well_id, row_idx, suffix_offset, fold, tvt_pred`
- columns read after prediction freeze:
  `tvt_true`

hidden-like assignmentとexp226 persistent episodeは、prediction freeze後の
scope / mechanism評価だけに使う。

## primary decoder

block `j`、offset state `d_j`に対し、exp426のRSD score `U_j(d_j)`を使う。

```text
objective =
  sum_j U_j(d_j)
  - 0.5 * (d_0 / 5 ft)^2
  - 0.5 * sum_j ((d_j - d_(j-1)) / 10 ft)^2
```

- states:
  `[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`
- initial center / sigma:
  `0 / 5 ft`
- transition sigma:
  `10 ft`
- first-block hard step:
  `|d_0| <=20 ft`
- adjacent hard step:
  `|d_j-d_(j-1)| <=40 ft`
- all candidates invalid:
  全state emissionを0とし、transitionだけでcarry
- partially invalid:
  invalid state emissionを`-inf`
- tie:
  score、0、小さい絶対値、negative、positiveの固定順

suffix boundaryのcorrection `0`と各block centerのMAP datumを線形補間し、
最後のblock center以降だけlast valueを保持する。

```text
primary_prediction(t) = exp226_tvt_pred(t) + datum_correction(t)
```

構成上の最大row correction slopeは`40 / 512 = 0.078125 ft/row`。

## 診断

### target-free support診断

- blockごとのvalid offset数 `0..13`
- 0 ft valid、左右対称pair valid、all-13 validの割合
- invalid理由:
  raw finite GR `<32`、paired bins `<16`、残余の低分散 / 相関不能
- wellごとのsupported block率
- supported block間のmedian / p90 / max gap
- 0--50、50--100、100--500、500--1000、1000+のsupport

これらは結果説明に使うが、primary activation、threshold、well選択には使わない。

### truth-late診断

- fixed blockwise top-1、unsupportedはoffset 0
- fixed13 discrete oracleとのtop-1 / top-3 / direction
- persistent episode SSEとwell改善率
- by-well delta分布

blockwise top-1はprimaryではなく、Viterbi改善の機構説明だけに使う。

## 実行順

1. input file、inventory、schema、logical / decompressed SHAを検証する。
2. score bankをstable sortし、support診断を生成する。
3. primary Viterbiとblockwise diagnosticをtruth-freeで全773 wellsに生成する。
4. config、input、support、datum path、row predictionのlogical SHAをfreezeする。
5. fixed probeと全体decoderを独立rerunし、prediction SHA parityを確認する。
6. ここで初めて`tvt_true`、hidden-like role、persistent episodeをjoinする。
7. 全row、fold、scope、by-well、episode metricsを計算する。
8. technical / scientific AND gateを判定する。

## technical gate

- exp426 score / manifest SHAとinventoryが完全一致
- exp226 rows / wells / folds / row identityが完全一致
- score生成、score変更、truth前activationが0
- truth / hidden role / episode read before prediction freezeが0
- parent RMSE parity absolute error `<=1e-6 ft`
- duplicate / missing prediction / missing datum pathが0
- independent decoder prediction logical SHA一致
- row correction slope `<=0.078125 + 1e-12 ft/row`
- CPU runtime `<=1,800 sec`
- peak RSS `<=25 GB`

support率はtechnical checkへ入れず、必須reportとする。

## scientific gate

すべてANDで要求する。

- pooled RMSE gain vs exp226 `>=0.10 ft`
- improvement folds `>=4/5`
- 1000+ RMSE gain `>=0.20 ft`
- persistent episode SSE reduction `>=10%`
- persistent episode wells improved `>=60%`
- 0--50 / 50--100、raw-GR missing、hidden-like spatial、
  hidden-like typewell-purgedのregressionを各`<=0.02 ft`
- corrected pathで新規検出されるpersistent episode SSEがcorrected全SSEの`<=5%`
- by-well RMSE delta p95 `<=+0.25 ft`
- worst-well RMSE delta `<=+5.0 ft`

## 実行量

- primary deterministic decoder:
  1
- diagnostic blockwise replay:
  1
- wells decoded:
  773
- reporting folds:
  5
- model / LightGBM config / trained fold / booster:
  `0 / 0 / 0 / 0`
- HMM / PF / Beam / GPU:
  `0 / 0 / 0 / 0`
- parent / control / score regeneration:
  0

## 判定後

- technical FAIL:
  scientific claimをせずterminal close
- scientific FAIL:
  exp426のtechnical FAILを維持し、sparse-anchor branchもterminal close
- all PASS:
  PF/Beam routeのtrain-side deterministic candidateとする。ただしraw-testで
  同じscoreを生成できる証拠はないため、inference / submissionへ直接進まない。

PASS / FAIL後とも、score、bin、block、offset、support、Viterbi sigma /
hard step、interpolation、activation、clip、blend、well gateをsame-OOFで変更しない。

## 再現性設計

- seed policy:
  RNGなし
- stochastic処理:
  なし
- parallel RNG:
  非該当。well結果はstable key順にreduceする
- runtime:
  Kaggle private CPU、GPU / internetなし、single worker
- SHA:
  input、schema、support、datum path、prediction、metricsのlogical SHAを記録
- gzip:
  raw gzip SHAではなくdecompressed content SHAを主証拠にする
- model / submission SHA:
  非該当
- deterministic anchor:
  独立rerun、Kaggle kernel version、prediction SHAが揃うまでfalse
- bootstrap:
  package前にmetadata、embedded config、exp426 / exp226 / exp115 sourceを照合

## リスク

- sparse support:
  supported anchorがerror発生位置より前後に偏り、全体RMSEへ寄与しない可能性
- partial candidate support:
  Type Well端で利用可能offsetが非対称となり、Viterbiが境界側へ偏る可能性
- CV/LB:
  train OOFでPASSしてもhidden testのsupport分布とraw score再生成parityは未検証
- same-OOF:
  decoderとgateを事前固定し、実行後の選択や救済を禁止する
- leakage:
  score / prediction freeze前のtruth、episode、hidden role readをfail-fastする

## 今回の実装境界

この設計セッションではsteering、design-only experiment scaffold、
config、README、SESSION_NOTES、result、metrics、backlog、summaryだけを作る。
compact source、tests、正規Notebook編集、Kaggle package / push / run、
inference、submissionは作成しない。

## 2026-07-28 実装追記

後続のユーザー実装依頼により、次を実装した。

- compact self-contained Jupytext train候補と、そこから変換した未実行Notebook候補
- exp426 score / well / input manifestのdecompressed / logical / schema SHA検証
- score / support / rank / top-3を再生成しないfixed Viterbi
- suffix offset 0のcorrection 0、固定512-row block center、last holdによる補間
- support、datum path、row predictionのSHA freezeと全decoder独立rerun
- prediction freeze前のtruth / hidden-like role / episode readを禁止するledger
- 全row / fold / scope / by-well / original episode / corrected episode readout
- technical / scientific AND gateとfail-close decision

「corrected pathで新規検出されるpersistent episode SSE」は、exp226根本原因監査と
同じ`abs(error) >= 10 ft`の連続128行以上をcorrected path上で再検出し、そのrunの
うち凍結済みexp226 original episode union外にある行のcorrected SSEと定義した。
original episodeの境界が少し動いただけでrun全体を新規扱いしないための決定的な
row-level定義であり、結果後に変更しない。

正規train / inference Notebookはplaceholderのまま、package / push / runも
falseのままとする。
