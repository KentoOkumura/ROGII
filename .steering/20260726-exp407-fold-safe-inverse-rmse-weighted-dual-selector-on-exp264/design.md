# 設計

## 結論

12候補を残したまま、共有dual selectorのcandidate-task exposureだけを候補別の
fold-safe inverse-RMSE weightへ変更する。候補削除ではなく学習時の重み付けを試す理由は、
保存済みOOF scoreからBeamを後段でmaskしただけではhard RMSEが悪化しており、
Beamが学習時に起こすnegative transferとは別の問いだからである。

初回は修正版exp264 Stage B v5に対するone-factor CPU ablationだけを行う。
設計は確定するが、実装とKaggle実行は今回のscope外とする。

## 2026-07-26 実装承認追記

後続のユーザー依頼によりStage B implementation-onlyを承認済みとする。
正規Notebookは上書きせず、別名のJupytext compact self-contained候補を作成する。
共有selector pipelineには既存呼び出しを変えないoptional weight hookだけを追加し、
exp407から明示的に固定weight configを渡す。Kaggle実行flagは閉じたままにする。

## 実験範囲

- 対象実験: `exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264`
- Route: `ml_model`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 変更する変数: selector fit時のcandidate-long sample weightだけ
- 固定する変数:
  - exp264の12候補と宣言順
  - candidate TVT surfaceとconfidence
  - primitive+pair 11候補 / primitive+fixed 7候補のlegal domain
  - raw-test-safe 88 feature schema
  - `pred_abs_error`と`p_within10`の2 objectives
  - outer 5 folds、seed 42、fit/valid sampling、early stopping
  - LightGBM params、candidate/family/formula one-hot、compact adapter
  - unweighted validation metricとgate

PF/HMM/Beam候補は保存済み値をmeta featureとして読むだけで、新規PF/HMM/Beam生成は0。
最終予測を生成する主体はML selector/downstreamなのでrouteは`ml_model`とする。

## 現状の根拠

親Stage B v5は12候補を同数ずつcandidate-longへ展開し、45,407,868 OOF rowsを
1つの共有LightGBM familyで学習した。candidate ID one-hotのgain shareは小さく、
bank/context特徴が大半を占めるため、候補間の共有表現にnegative transferが起こる余地がある。

`beam_mean`はlast-known TVTとの差の中央値が約1.88 ft、5 ft以内が約88.6%で、
anchorに近いことが多い。一方、候補単体RMSEは15.774327 ftである。ただし
anchorより良い行も約54.7%、oracle-bestになる行も約8.57%あり、候補から除去する根拠には弱い。
実際、保存済みparent scoreを学習後にBeamだけmaskしたreadoutは、
expected-error hard RMSEを8.5870から8.6540へ、within10 hard RMSEを
8.7479から8.8370へ悪化させた。このためBeam削除ではなく学習配分だけを変える。

全OOFから計算した次表はweight強度のsanity check専用であり、fitには使わない。

| candidate | OOF RMSE (ft) | illustrative inverse-RMSE weight |
| --- | ---: | ---: |
| `exp226_k16` | 9.427110 | 1.0968 |
| `selfgr_hmm_a070` | 11.349943 | 0.9110 |
| `likpf_mean` | 11.594898 | 0.8917 |
| `exact_hmm` | 11.938287 | 0.8661 |
| `pf_ancc` | 14.493051 | 0.7134 |
| `beam_mean` | 15.774327 | 0.6555 |
| `exp226_k16__selfgr_hmm_a070` | 8.532715 | 1.2118 |
| `exp226_k16__exact_hmm` | 8.635074 | 1.1974 |
| `exp226_k16__likpf_mean` | 8.813822 | 1.1731 |
| `selfgr_hmm_a070__likpf_mean` | 10.123457 | 1.0214 |
| `likpf_mean__exact_hmm` | 10.269697 | 1.0068 |
| `exp226_w500_50_50` | 8.238332 | 1.2551 |

この尺度ではBeamも約0.66を維持し、完全には消えない。inverse-squareならBeam約0.41、
fixed formula約1.52となってclipへ達するため、初回の一因子検証として強すぎるので禁止する。

## fold-safe weight生成

### Stage B

各outer foldについて、親と同じdeterministic sampled outer-train base rowsを先に固定する。
feature schema、row IDs、candidate順をfreezeした後、fit rowsだけへtrue TVTをjoinする。
12候補それぞれのactual TVT RMSEを計算し、以下の固定式を使う。

```text
rmse_c       = sqrt(mean((candidate_tvt_c - true_tvt)^2))
raw_c        = 1 / max(rmse_c, 1e-6)
normalized_c = raw_c / mean(raw)
clipped_c    = clip(normalized_c, 0.5, 1.5)
weight_c     = clipped_c / mean(clipped)
```

最終weightがfinite、12本、mean 1.0、範囲`[0.5, 1.5]`でなければfit前に停止する。
outer-valid truthはweight計算にもfitにも使わない。candidate-longへ展開後、
各行へcandidate ID対応の`weight_c`を付け、同じweight列を両objectiveのLightGBM fitへ渡す。
validation dataにはweightを渡さない。

### Stage C（Stage B全PASS・別承認後のみ）

各outer × inner modelごとに、親と同じdeterministic sampled inner-train base rowsだけで
同じ式を再計算する。inner-validとouter-validのtruthはweight計算へ入れない。
outer-train compactはinner OOF score、outer-valid compactは4 inner model ensembleから作る
exp264 strict-nested契約を維持する。必要量は1 variant × 2 objectives × 5 outer ×
4 inner = 40 CPU boostersで、親control再学習は0。

### Stage D（Stage C全PASS・別承認後のみ）

clean 273 + compact 74のselector add-only variantだけを3 configs × 5 folds =
15 GPU boosters学習する。保存済みexp264 clean-273 controlを参照し、
control 15 boostersは再学習しない。Stage Dの詳細実装、推論、提出は今回承認されていない。

## 評価設計

primary comparisonは保存済み修正版Stage B v5である。

| 指標 | parent v5 | exp407 gate |
| --- | ---: | ---: |
| expected-error MAE | 3.7958011626 | 3.7858011626以下、4/5 folds改善 |
| within10 logloss | 0.3599715948 | 0.3604715948以下、3/5 folds非悪化 |
| within10 Brier | 0.1124509792 | 0.1129509792以下、3/5 folds非悪化 |
| hard primary RMSE | 8.5870043867 | 親以下、3/5 folds非悪化 |

near、1000+、2種hidden-likeは親比`+0.02 ft`以内、worst-wellは親比
`+0.25 ft`以内を追加guardとする。全metricはunweightedで再計算する。
候補別metric、candidate選択率、weight分布は診断に保存するが、同run内の調整には使わない。

Stage Bでexpected-errorだけ改善してclassificationやtail guardを外す場合もFAILとする。
RMSE weightが有力候補へ学習量を寄せる一方、弱い候補のcalibrationを壊してcompact feature価値を
下げる可能性があるためである。

## 実行予算と承認境界

| 段階 | 新規学習 | accelerator | 今回の状態 |
| --- | ---: | --- | --- |
| Stage B | 10 boosters | Kaggle CPU | 実装済み・実行は別承認 |
| Stage C | 40 boosters | Kaggle CPU | Stage B全PASS後も別承認 |
| Stage D | 15 boosters | Kaggle GPU | Stage C全PASS後も別承認 |
| inference / submission | 0 training | 未定 | 未承認 |

Stage Bでparent/controlを再学習しない。Stage Dでも保存済みcontrolを使う。
既存artifactで比較不能な契約差が見つかった場合は、独断でcontrolを再学習せず停止して確認する。

## 再現性設計

- seed policy: 親の固定seed 42とdeterministic sampled row IDを継承する。
- stochastic処理: LightGBMのsubsample/column sampleだけ。親の
  `deterministic=true`、`force_col_wise=true`、thread数を固定する。
- PF/Beam / likelihood-PF / seed bagging: 新規実行0。保存済みcandidate surfaceをload-onlyで使う。
- 並列処理と乱数: candidate順、fold順、row順、weight table順をstable sortし、
  global RNGやthread completion orderからweightを作らない。
- runtime: Stage B/CはCPU。Stage Dは別承認時だけGPU。deterministic anchorは
  同一package rerunのmodel/OOF SHA parity確認前には宣言しない。
- SHA: parent input manifests、fit row IDs、sampling manifest、candidate contract、
  88列schema/content、fold別weight table、model files/manifest、candidate-score OOF、
  compact OOF、metrics/decisionを記録する。
- Kaggle bootstrap: 実装時はprivate、internet off、CPU metadata、canonical kernel id/title、
  packaged/source `config.yaml` SHA一致、required package importをpush前に確認する。
- gzip: decompressed content SHAを主証拠にする。

## リスク

- リークリスク: global OOF RMSEやouter/inner validation truthからweightを作ると直接leakする。
  fit partition限定とtruth-read ledgerでfail closedにする。
- objective mismatch: actual TVT RMSEは`p_within10`の最適なtask importanceとは限らない。
  ただしobjective別weightを増やすと一因子でなくなるため、初回は同じweightで固定し、
  classification non-regression gateで守る。
- selector diversity低下: 強い固定blendへweightが寄り、弱い候補の局所oracle情報を失う可能性がある。
  candidate削除をせず、clipとunweighted hard/tail gateで守る。
- CV/LB不一致: Stage B selector metric改善がdownstream/LBへ転移するとは限らない。
  Stage C/Dを自動昇格させず、strict nested CVを別段階にする。
- 再現性: row sampling差だけでRMSE weightが変わる。fit row IDとweight tableをSHA固定する。
- rescueバイアス: FAIL後にinverse-square、clip、Beam削除、candidate subsetへ進むと
  同一OOF最適化になる。FAIL時はbranchを閉じる。
