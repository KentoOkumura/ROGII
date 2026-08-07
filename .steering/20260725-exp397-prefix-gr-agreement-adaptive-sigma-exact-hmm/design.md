# 設計

## 結論

exp209 exact HMMのwell-level GR scaleを捨てず、known prefixのGR shape agreementが悪いwellだけ
`1.3` 倍してGaussian emissionを弱める。agreementが良いwellと判定不能wellは係数 `1.0` とし、
保存済みexp209 predictionをそのまま使う。

primary agreementはraw finite pairのPearson相関だけに固定する。bias、RMSE、NCC、tail相関は
primary selectorへ入れない。これにより、exp209 base scaleと同じresidual dispersionを
二重に判定することを避け、自由度を1 threshold / 2 coefficientsへ限定する。

## 仮説

exp209のzero-filled residual stdは全体としてfinite-only scaleより安全だが、horizontal GRと
typewell GRのshape一致度はwellごとに異なる。一致度が低いwellではGaussian emissionが
誤ったmodeを強く固定し得るため、base scaleを `1.3` 倍してGR evidenceを弱めれば、
一致度が高いwellを変えずにexact-HMMのtail errorを減らせる可能性がある。

Kaggle discussion 728712には公開PF系Notebookの `gs` を約 `1.3` 倍すると改善したとの記述がある。
これは外部の弱い実証根拠であり、score、CV、versionの詳細は確認できない。本実験への転用、
Pearson `0.50` gate、exact HMMでの改善はすべて未検証の仮説として扱う。

## 実験範囲

- 対象実験: `exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: prefix GR agreementで選ぶwell-level `sigma_gr` 係数 `1.0 / 1.3`
- 固定する変数:
  - exp209のabsolute-TVT exact forward-backward
  - known-prefix zero-filled residual population stdとclip `[10, 60]`
  - typewell GR sort / fill / interpolation
  - evaluation GRの双方向補間とtypewell mean fallback
  - `step=0.35`、41 rates、rate span `0.10`、band pad `100`
  - `sig_r=0.002`、`sig_p=0.02`、`momentum=0.998`
  - start/rate prior、Gaussian `z^2` clip `600`、posterior mean出力
  - 5 reporting folds、unknown-suffix score rows、非加重RMSE

ML model、booster、PF、Beamは新規実行しない。保存likelihood-PF predictionは固定50:50 blendの
reporting controlにのみ使うため、Routeは `pf_beam` とする。

## 比較対象と固定証拠

| 役割 | 固定値 / 成果物 | 用途 |
| --- | --- | --- |
| scientific parent | exp209 saved Gaussian exact-HMM OOF | direct control、係数1.0 wellのprediction |
| parent direct RMSE | `11.938287234887435` | primary比較 |
| parent fixed LikPF 50:50 RMSE | `10.269696146642758` | downstream non-regression |
| parent HMM decompressed SHA256 | `8e2f42367b7b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5` | input parity |
| fold assignment | exp226 safe `well_id / fold` | reporting strataだけ |
| hidden-like roles | exp115 saved assignments | late-truth stress reportingだけ |

SHA、row identity、parent RMSEが一致しない場合はHMM前に停止する。parent controlは再実行しない。

## Prefix agreement契約

### Pairの定義

well `w` のknown prefixで、raw horizontal `GR`、`TVT_input`、typewell補間GRがすべてfiniteな
rowだけを元のrow順で残す。

1. typewellをTVTでstable sortする。
2. typewell GRはexp209と同じforward/backward fillを行う。
3. `g_tw_i = interp(TVT_input_i, typewell_TVT, typewell_GR)` をendpoint holdで計算する。
4. `g_hw_i = raw horizontal GR_i` とする。agreement計算ではGR missingを0埋めしない。
5. pair集合上のPearson相関を `rho_gr = corr(g_hw, g_tw)` とする。

affine fit、lag search、DTW、NCC、smooth、residual mean除去以外のcalibrationは行わない。
Pearson相関は定義上mean centerを含むが、係数やoffsetをfitしてGR値自体は変更しない。

### Supportと係数

- `n_pair >= 64`
- `std(g_hw) > 1e-6`
- `std(g_tw) > 1e-6`
- `rho_gr` がfinite

をすべて満たすwellをevaluableとする。

```text
if not evaluable:
    sigma_multiplier = 1.0
elif rho_gr >= 0.50:
    sigma_multiplier = 1.0
else:
    sigma_multiplier = 1.3
```

boundary `rho_gr == 0.50` は `1.0` 側に含める。相関はfloat64で計算し、well IDの辞書順にfreezeする。
fallback `1.0` は情報不足だけを理由にtrusted parentを変更しないためのno-opである。

### HMMへの適用

base scaleはexp209どおり計算する。

```text
sigma_base = clip(
    population_std(fillna(horizontal_GR, 0) - typewell_GR_at_TVT_input),
    10,
    60,
)
sigma_eff = sigma_multiplier * sigma_base
log_emission = -0.5 * min(((GR_eval - expected_GR_state) / sigma_eff)^2, 600)
```

係数はbase clip後に正確に1回だけ掛け、`sigma_eff` を再clipしない。したがって候補の範囲は
`[10, 78]` となる。Gaussian normalizationは同じrow内の全stateに共通でposteriorから相殺されるため、
exp209どおり省略する。係数はwell内の全unknown-suffix row / stateに一定で、row-varyingにはしない。

`1.3` 倍はquadratic emissionの重みを `1 / 1.3^2 = 0.5917159763` 倍することに等しい。
emission familyはGaussianのままであり、transitionやpriorは変更しない。

## Stage 0: truth-free識別性・安定性監査

Stage 0は全773 wellsでagreement surfaceだけを作る。HMM、prediction、truth join、model fitは0。

primaryはfull known prefix、stability windowはknown prefix末尾512 raw rowsとする。tail側は同じfinite
pair規則、minimum 32 pairs、同じ `rho_gr=0.50` で係数を計算する。tail係数は診断専用で、
Stage 1のselectorには使わない。

全項目をAND条件とする。

1. full-prefix evaluable well fraction `>= 0.90`。
2. fallback fraction `<= 0.10`。
3. full-prefix `1.3` group fractionが全wellの `[0.10, 0.90]`。
4. 各reporting foldのfull-prefix evaluable fraction `>= 0.80`。
5. tail evaluable well fraction `>= 0.75`。
6. full/tail両方evaluableなwellで係数一致率 `>= 0.80`。
7. full/tail両方evaluableなwellで `rho_gr` のSpearman相関 `>= 0.70`。

secondary reportとしてpair数、`rho_gr` 分位点、horizontal/typewell std、mean bias、normalized bias、
full-tail差、fold別係数件数を保存するが、primary selectorやgateを追加しない。

Stage 0 FAIL時はbranchを閉じ、threshold、support、tail長、bias条件、相関種、係数を変えて救済しない。

## Stage 1: prefix-agreement adaptive sigma exact HMM

Stage 0全gate PASSと別のユーザー承認後だけ実装・実行する。

- active variant: `prefix_rho_lt_0p50_sigma_x1p3`
- scientific variants: 1
- reporting folds: 5
- HMM well-runs: `sigma_multiplier=1.3` のwell数、上限773
- `1.0` / fallback well: SHA固定した保存exp209 predictionを再利用
- model configs / trained folds / boosters / PF runs / Beam runs / GPU: 0
- parent HMM control rerun: 0

agreement tableと係数をunknown-suffix truth読込前にfreezeし、logical content SHAを保存する。
candidate predictionもtruth、error、hidden-like roleを読む前にfreezeする。truthはrow identity、
prediction SHA、coefficient SHAの固定後に評価用としてだけjoinする。

## Stage 1 technical gate

- raw input、fold、hidden-like assignment、parent HMM、saved LikPFの期待SHAが一致する。
- 773 wells / 3,783,989 evaluation rows、ID、row order、fold、truth coverageが一致する。
- agreement tableがStage 0のSHA、support、threshold、係数件数と一致する。
- `1.0` / fallback wellのcandidate predictionがsaved exp209とabsolute tolerance `1e-12` で一致する。
- `1.3` wellのposteriorがfiniteかつ各rowでnormalizeされ、全well statusがPASSする。
- truth/errorの事前read countが0で、candidate prediction freezeがtruth joinより先である。
- actual HMM well-runsが `1.3` well数と一致し、773以下である。

## Stage 1 promotion gate

全項目をAND条件とする。

1. exp209比pooled direct RMSE gain `>= 0.05 ft`。
2. 5 reporting folds中4 folds以上でexp209以下。
3. `1.3` group内のdirect RMSE gain `>= 0.05 ft`。
4. `1.3` groupのby-well delta RMSE p95 `<= 0.00 ft`。
5. `1.3` groupの改善または同値well fraction `> 0.50`。
6. 全wellのworst-well regression `<= +0.25 ft`。
7. raw-observed、raw-missing、high-missing、1000+、hidden-like spatial、
   hidden-like typewell-purgedの各scopeがexp209比 `<= 0.00 ft`。
8. fixed LikPF/HMM 50:50が親fixed blend比 `<= 0.00 ft`。

FAIL時はnegative resultを記録してterminal closeする。同じOOFでthreshold、multiplier、support、
window、bias gate、continuous mapping、clip、temperature、blendを選ばない。

## 本実験に含めないもの

- `rho_gr` のcontinuous mapping、quantile threshold、fold別threshold、3値以上の係数
- `1.3` 以外のmultiplier、multiplier grid、post-multiplier clip
- RMSE、bias、NCC、DTW、ACF、tail agreementをprimary selectorへ追加
- finite-only / MAD scale、affine GR calibration、row-varying sigma、missing-distance weight
- Huber、Student-t、mixture、emission temperature、transition/prior/state-grid変更
- PF/Beam rerun、blend weight search、ML selector、worst-well rule、same-OOF rescue
- parent control再実行、inference、submission

## 外部根拠と既存negative evidence

- Kaggle discussion:
  `https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/728712`
- referenced public notebook:
  `https://www.kaggle.com/code/hjyact/ultimate-pf-config-strategy-a-reproducible-score`
- exp307: finite-only std / MADへのscale縮小はdirect RMSEを大幅に悪化させた。
- exp343: ACF effective-sample temperingはsupportとclip非退化gateをFAILした。
- exp346: observed-only finite sigmaはexp209より悪化した。
- exp389: fixed Huberは平均を小幅改善したがby-well tail gateをFAILした。

したがって本実験はscaleを狭めず、`1.0 / 1.3` のbounded softeningだけを許し、target-free Stage 0と
厳しいwell-tail gateを先に固定する。

## 再現性設計

- seed policy: RNGなし。well ID辞書順、raw row順、fixed fold順で処理する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: 新規実行なし。saved LikPFは固定blend reportingだけ。
- 並列処理: HMM実装を承認した場合もfixed thread countでwell単位処理し、乱数は使わない。
- CPU/GPU: CPU、GPUなし。deterministic numerical auditでありsubmission anchorではない。
- feature SHA: agreement table、係数table、input manifest、row identityはlogical content SHAを記録する。
  gzip predictionはdecompressed content SHAを主証拠にする。
- model/prediction SHA: modelなし。Stage 1承認後はcandidate prediction SHAとparent control SHAを保存する。
  inference/submission SHAは未承認のため生成しない。
- bootstrap: package/push承認後にembedded config、source、input artifact、Kaggle metadata、
  internet/GPU設定を照合する。

## リスク

- リークリスク: `rho_gr` にunknown-suffix truthやlate roleを混ぜるとwell selectorがleakする。
  agreement/coefficient/predictionをtruth前にSHA freezeする。
- CV/LB不一致: train 773 wellsでの相関分布とhidden test wellsの分布は一致しない可能性がある。
  fixed semantic thresholdとfallbackを使い、testで閾値を再fitしない。
- proxyリスク: Pearson相関はshape agreementだけを見て、絶対biasや局所ずれを見落とす。
  secondary reportは残すが同じ実験のselectorへ追加しない。
- tailリスク: emissionを弱めることでGRが有効なpoor-rho wellまで悪化し得る。changed groupのp95、
  worst、stress scopeをpromotion gateにする。
- runtime: Stage 1は最大773 exact-HMM runsで長時間になり得る。Stage 0を0-HMMで先行し、
  係数1.0 wellはsaved predictionを再利用する。
- 再現性: 並列reductionやfloat差があり得るため、float64 agreement、固定順、thread数、
  logical/decompressed SHAを保存する。
- 外部根拠: discussionの改善値は詳細不明でPF系からの転用であるため、LB根拠として扱わない。

## 現在の承認境界

2026-07-25の後続ユーザー指示によりStage 0 compact self-contained candidateと専用testを実装し、
さらに実行指示を正規train notebook採用とKaggle private CPU Stage 0 package/push/runの承認として
扱った。version 1（id_no `128540665`）は`39.35975061899995 sec`で完了した。

fixed 7 gatesのうちcoverage 4条件はPASSしたが、poor multiplier fraction
`8/773 = 0.01034928848641656`、full/tail multiplier agreement `0.666235446313066`、
full/tail Spearman `0.16746641700676126`が各固定下限をFAILした。このためdecisionを
`stage_0_failed_close_without_rescue`とし、Stage 1、inference、submission、version 2は
未実装・未実行のままbranchを閉じる。再実行承認flagはfalseに戻し、threshold、multiplier、
support、window、相関種の事後調整と同family rescueは行わない。
