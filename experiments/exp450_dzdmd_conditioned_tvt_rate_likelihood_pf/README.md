# exp450_dzdmd_conditioned_tvt_rate_likelihood_pf

## 状態

- Route: `pf_beam`
- 状態: version 3 COMPLETE・Stage 0B mechanism FAIL_CLOSED
- CV / LB / Submit: なし
- 親endpoint: exp417
- 実装参照・保存control: exp404 temperature-5 x1.0

## 仮説

well自身のvisible prefixから`TVT-rate = beta*dZ/dMD + intercept`を推定し、
その平均との差だけを持続させれば、TVT-rate状態を使いながら既知Zの駆動を
transitionへ戻せる。

## 変更点

親の`(U, U-rate)`を厳密に`(TVT, TVT-rate)`へ座標変換し、

```text
mu_t = beta * dZ/dMD_t + intercept
q_t = mu_t + 0.998 * (q_previous - mu_previous) + noise
```

とする。`beta/intercept`はvisible prefixの最低10 valid stepsからwell別OLSで
推定し、不足時は`-1/0`へfallbackする。PF_Zの追加rate likelihoodは使わない。

## 検証方針

- Stage 0A: exp410 sentinel12で`beta=-1, intercept=0`の親U-rate PFとの
  paired exact-coordinate parity。
- Stage 0B: exp411/exp446 fixed32でprefix backtest、under-response、
  persistent/control安全性。fixed32はCVではない。
- Stage 1: 全事前gate PASS・別承認後だけ773 wellsを保存exp404と比較。
- primaryはRMSE 0.05 ft以上改善、4/5 folds、全安全scopeとwell-tailのAND gate。

## 実行量

- Stage 0A: 24 PF well-runs、3,072 seed-well、1,536,000 particle starts。
- Stage 0B: 32 PF well-runs、4,096 seed-well、2,048,000 particle starts。
- Stage 1: 773 PF well-runs、98,944 seed-well、49,472,000 particle starts。
- 保存control rerun、ML、HMM、Beam、booster、GPU: すべて0。
- version 3実run: Stage 0A 24 + Stage 0B 32 = 56 PF well-runs。

## 所見

### 良い点

prefix backtestはSSE ratio`0.241989`、`5/5 folds`非悪化。persistent scope
pooled RMSEも`12.785573 -> 12.462589 ft`へ改善し、10/16 wellsを改善した。

### リスク

prefix内のaffine関係がunknown suffixで持続しないこと、gのrangeが狭いwellで
係数が不安定になること、平均改善とwell-tail悪化が同時に起こることが主なリスク。
係数clip/shrink、window/grid、well gate、blend/selectorでsame-run救済しない。

実際にmatched control pooledは`+0.292528 ft`、p95は`+1.678265 ft`悪化し、
under-response shareも`0.004650`悪化した。persistent改善foldも`2/5`だった。

## 成果物

- compact self-contained train候補:
  `exp450_dzdmd_conditioned_tvt_rate_likelihood_pf_compact_selfcontained_train.py`
  / `.ipynb`
- inference fail-closed guard候補:
  `exp450_dzdmd_conditioned_tvt_rate_likelihood_pf_compact_selfcontained_inference.py`
  / `.ipynb`
- 専用test:
  `experiments/exp450_dzdmd_conditioned_tvt_rate_likelihood_pf/tests/test_exp450_dzdmd_conditioned_tvt_rate_likelihood_pf.py`
- prefix OLS、tail20 backtest、exact-coordinate paired PF、学習型
  residual-AR PF、candidate/control freeze、truth-late mechanism readout、
  Stage 0A/0B全AND gate、SHA出力を実装した。
- 正規train Notebookへcompact self-contained実装を採用済み。正規inferenceは
  guard scaffoldのまま。
- Kaggle private CPU version 1（id_no `129167787`）は`COMPLETE`。
- Stage 0A exact-coordinate parityはFAILし、`5f4d2a52`で57回の
  resampling decision mismatch、最大seed prediction差`21.176790850 ft`。
  他11 wellsにも概ね`1e-9`級の丸め差があり、固定`1e-10` gateを超えた。
- Stage 0Bのcandidate生成とtruth-late mechanism gateを完了した。
  Stage 1、inference、submissionは実行していない。
- version 2は改訂Stage 0AをPASSし、Stage 0B candidate 32 wellsを生成したが、
  exp404 typed logical SHAをCSV文字列SHAで照合する実装ミスでERRORになった。
- version 3はtyped SHA修正後に`COMPLETE`。Stage 0AはPASS、Stage 0Bは
  10/16 gate PASS・6 FAILで`stage0b_mechanism_failed_closed`。

## 実行入口

- 編集元はJupytext percent形式のcompact sourceで、正規train `.ipynb`へ
  変換済み。
- run後は`execution.selected_stage=null`へ戻し、train flagを無効化した。
- Stage 0Aは最終temperature-5予測差`<=1e-6 ft`をhard gateとし、
  seed/weight/state/resampling差は診断値として保存する。
- Stage 1、raw-test inference、submissionは実装・有効化していない。

## 次

現行exp450を終了する。Stage 1、rerun、parameter/grid救済、inference、
submissionへ進めない。
