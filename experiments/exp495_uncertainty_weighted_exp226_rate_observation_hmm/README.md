# exp495 exp226 rate不確実性重み付き観測HMM

## 状態

- Route: `pf_beam`
- 状態: Stage 0B fixed32完了・technical 1件 / mechanism 7件FAIL・fail-closed
- CV / Public LB / Private LB: なし
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- rate signal: `exp355_exp226_dip_rate_prior_on_exp209`

## 仮説

exp355では、exp226 geometryの相対rate変化をexp209 HMMへ与えると平均RMSEと
5 foldsすべてが改善した一方、hidden-likeとworst-wellが大きく悪化した。
exp226 rateを一律に信頼せず、known prefixで測ったrate残差scaleを観測分散として
使えば、平均signalを残しながら危険なwellでは親HMMへ縮退できる可能性がある。

## 変更点

exp209のrate状態と遷移を残し、exp226 geometry U-rateをGaussian観測として
rate transitionへ掛ける。exp226 final`tvt_pred`の差分でTVT遷移量を置き換えない。

```text
mu_226[t] = r_prefix_exp209 + r_geom[t] - r_geom[first_segment]
sigma_226[w] = max(0.002, 1.4826 * MAD(last128 prefix rate residual))
P495_t(j|i) ∝ P209(j|i) * Normal(r_j; mu_226[t], sigma_226[w])
ΔTVT = r_j * ΔMD - ΔZ
```

known-prefix transitionが32未満なら観測を無効化し、exp209と完全parityに戻す。
追加lambda、temperature、clip、threshold、row/well gateは使わない。

## 検証方針

1. Stage 0A: 773 wells、0 HMMでprefix uncertaintyのsuffix transferを検証。
2. Stage 0B: ユーザーのStage 0A停止点overrideによりfixed32で1候補×32 HMM runsを実行。
3. Stage 1: Stage 0B全PASSと別承認後だけ1候補×773 HMM runs。

各段階は全AND gateで、1件でもFAILなら同一OOF上の調整やPF救済を行わず閉じる。

## 主なリスク

- known-prefix rate残差scaleがunknown suffixのdonor mismatchを説明しない可能性がある。
- exp355と同様に平均改善とwell-tail悪化が併存する可能性がある。
- TVT-rateとU-rateを混同すると`ΔZ`を二重補正する。
- direct physical routeはCV/LB順位が安定せず、Stage 1 PASSでも提出候補とは限らない。

## 所見

- compact self-contained Jupytext train候補と変換済み候補Notebook
- outer-fold donorだけを使うexp226 geometry-only prefix replay
- 最後の128有効prefix transitionから作るcentered MAD `sigma_226`
- strict allowlist、exp209 / exp226 / fold / raw identity SHA guard
- `mu_226` / `sigma_226` / fallback / prediction freeze後のrole・episode・truth late-join
- exp209 absolute-rate exact HMMへrow-normalized Gaussian rate観測を追加
- leakage、U-rate式、K16 identity、fallback、freeze、uniform-factor parityの23契約テスト

compact候補を正規`*_train.ipynb`へ採用し、Kaggle private CPU version 1
（id_no `129285050`）でStage 0Aを完走した。technical gateは全件PASSしたが、
pooled Spearman `0.088435 < 0.20`、low/high-sigma rate RMSE gain
`0.076454 < 0.10`でmechanism gateをFAILした。

その後、ユーザーの明示overrideで事前登録条件のStage 0BをKaggle private CPU
version 4で実行した。candidate all32 RMSEは`13.069257`で保存exp355
`10.677951`より`+2.391305 ft`悪化、persistentは`+2.143391 ft`、matched controlは
保存exp209比`+5.026402 ft`悪化した。改善foldは2/5、by-well p95は
`+16.564282 ft`、worstは`+23.911032 ft`でmechanism 7/7 FAILだった。
Stage 1 / inference / submissionは未実施である。

## 次

事前登録したfail actionどおり、window / sigma / scale / temperature / emission /
grid / gateを同一fixed32で救済せずbranchを閉じる。次の調査は保存済みStage 0A/0B
artifactだけを使ったprefix-to-suffix transfer失敗の原因分解に限定する。
