# exp418_exp226_signed_segment_rate_residual 結果

## 状態

Stage 0 Kaggle version 1完了、technical FAILでfail closed。Stage 1 / inference /
submissionは未実行。deployable model CV / LBはない。

## 仮説

exp226の低周波offsetはsegment levelそのものではなく、targetとdonor fieldの
小さなsigned rate mismatchを再anchorなしで累積した結果である。K16ごとのrateを
学習し、first unknown rowの補正0から連続積分すれば、exp333のconstant-offset
broadcastより安全に原因へ介入できる。

## 設定

- 親: exp333
- base: exp226
- route: `ensemble`
- target: zero-intercept K16 cumulative residual rate (`ft/row`)
- feature: exp333と同じtarget-free 136列
- model: exp333と同じLightGBM `lgb1` 1 config
- planned training: 1 variant ×5 folds = 5 CPU boosters
- exp226/control再学習、GPU、PF/HMM/Beam再生成: 0
- metric: row-level TVT RMSEとfold/scope/well-tail AND gate

## 結果

compact self-contained train候補と対応Notebook、専用contract testを実装し、
Stage 0をKaggle private CPUで実行した。

- 親exp333の保存nested prediction / fold manifest / feature schema / SHA manifestを
  fail closedで検証する。
- exp333-compatible target-free 136特徴を再構築し、row content SHAと
  feature-freeze SHAを照合する。
- exp226 donor field / kappa fit / nested prediction再生成を呼ばない。
- Stage 0: 0 model / 0 boosterのcontinuous-rate oracle。
- Stage 1: 1 variant ×1 config ×5 folds = 5 CPU boosters。
- Stage 1はSHA固定したStage 0 `PASS_STAGE0` summaryがなければ開始できない。
- feature importance CSVとtop-30 plot、fold/scope/tail/rate target、model、
  OOF、SHA manifestを保存する。

検証:

- 専用pytest: `14 passed`
- `py_compile`: PASS
- Ruff `F821`: PASS
- Jupytext `--to ipynb --test`: PASS
- `__file__`: 0件
- 親compact比較: exp333 2,297行 / 12章、exp418 2,342行 / 12章
- repository全体の`make test`: 既存6件のcollection errorで未完走
  （exp297/301/333/336/349のroot config誤読、exp411の`numba.__spec__`）。
  exp418専用testとstrict validationはPASS。

Kaggle実行:

- kernel: `kentookumura/exp418-exp226-signed-segment-rate-train`
- version / id_no: `1` / `128832515`
- runtime: 約146秒でStage 0 summary出力、kernelはCOMPLETE
- 実行量: 1 readout、0 model、0 booster、0 exp226 fit、CPU
- rows / wells / segments: `3,783,989` / `773` / `12,368`
- exp226 parity RMSE: `9.427109596582220`
- signed-rate oracle RMSE: `0.646951416159574`
- gain vs exp226: `8.780158180422646 ft`
- fold gain: 5/5 PASS（各`8.3797–9.6805 ft`）
- basis rank: 全wellで16、finite fraction: target/correctionともに1.0
- first-row correction最大絶対値: `0.0 ft`
- matrix vs sequential integration最大絶対差:
  `6.295408638834488e-12 ft`
- 事前固定上限: `1.0e-12 ft`
- technical checks: 8/9 PASS、`integration_parity`だけFAIL
- decision: `FAIL_CLOSE_BRANCH`
- summary file SHA:
  `07c719e0f174b1712650563620f6331504dbd1333969c8777f41ce46419dc412`
- rate-target content SHA:
  `5c936b03e86e7250afdfef551e796e0beead22d50715d025b84acb9b13a9e2ff`
- rate-target file SHA:
  `a705f37ba3529f92fcd7d5da441550d0e0742a462bb6d1131fab562cb0b3695e`

oracle predictionは保存しておらず、model、prediction candidate、LBは存在しない。

## 解釈

pooledと全foldの科学閾値は満たしたためsigned cumulative-rate targetのoracle
headroomは非常に大きい。一方、行列積と逐次加算の演算順によるfloat64差が
`6.30e-12 ft`となり、事前固定した`1e-12 ft` gateを超えた。実用誤差としては
極小でも、同一OOFを見た後のgate緩和は禁止しているためexp418はtechnical FAIL
として閉じる。この結果をStage 0 PASS、prediction候補、route anchorとは扱わない。

## 次

Stage 1、inference、submissionへは進まない。次候補は、truthやoracle gainを使わず
cross-runtimeで累積演算のULP/scale-aware contractと単一canonical integrationを
先に固定する独立numerical auditである。exp418内の閾値緩和や再実行ではなく、
別仮説・別実験・別承認が必要。
