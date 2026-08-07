# 設計

## 目的・仮説

複数のvisible-prefix initial-rate modeを各seedの単一PF内に維持すれば、
tail30単一centerで起こる初期mode lossを減らせる、という仮説を
target-free Stage 0で実装・監査可能にする。

## 1. 科学差分

exp404の単一tail30 initial rate centerを、exp268由来の5中心へ置換する。

```text
rate(w) = median((delta_TVT_input + delta_Z) / delta_MD)
windows = [tail30, 32, 64, 128, 256]
minimum valid steps = 3
fallback rate = 0.0
```

500 particlesは`particle_index % 5`で5 componentへinterleaveし、各componentを
ちょうど100粒子とする。各粒子は対応centerへexp404と同じGaussian spread
`0.01`を加える。position初期化、以後のdynamics、emission、resampling、
roughening、128 seeds、temperature-5 aggregationは変更しない。

これは5本のPF candidateやpost-hoc window selectorではなく、一つの
stratified mixture priorである。duplicate centerもそのまま保持し、
結果を見てcomponentを削除しない。

## 2. Target-free mechanism readout

- wellごとの5 center、range、unique count、fallback count。
- row 0、32、128、512とscore末尾でのcomponent ancestry mass。
- 最初のresamplingまでのESS、component extinction row、surviving component数。
- `particle_index % 5`ごとの初期countが常に100であること。
- component labelはlineage診断専用で、outputやweightへ追加効果を持たせない。

## 3. 段階と実行量

- Stage 0: fixed32、32 PF well-runs、4,096 seed-well、
  2,048,000 particle starts。technical/mechanism preflightでCVではない。
- Stage 1: 全PASS・別承認時だけ773 PF well-runs、98,944 seed-well、
  49,472,000 particle starts。
- 保存exp404 control rerun、HMM、Beam、model、booster、GPUは0。

## 4. Gate

Stage 0:

- 各well/seedで初期count `[100,100,100,100,100]`。
- rate計算、fallback、finite coverage、stable seed、truth-late、SHAがPASS。
- 全fixed32で少なくとも2 unique centerを要求しない。重複自体を失敗扱いせず、
  全32 wellsで全centerが同一ならmechanism-degenerateとしてFAILする。
- full runtime投影`<=30,600 sec`、RSS`<=25 GB`。

Stage 1:

- exp404 scale-5 x1.0 `10.914522073`から`0.05 ft`以上、4/5 folds以上改善。
- raw observed `0.05 ft`以上改善。
- raw missing、高missing、1000+、hidden-like 2面でregression `<=0.0 ft`。
- by-well p95 `<=0.0 ft`、worst `<=0.25 ft`。
- exp209 HMMとの固定50:50 blend `10.084909680`より非悪化。

FAIL時はwindow、component weight、spread、particle/seed、temperature、
rate selector、oracle、gate、blendで救済しない。

## 5. 再現性と承認境界

exp404 stable per-well seedを継承し、componentはparticle indexだけで決める。
train/testは別生成し、rate bank、component ledger、prediction、schema、
logical/decompressed content SHAをfreezeした後だけtruthをattachする。
初回runはanchorにしない。2026-07-30の追加依頼でStage 0実装のみ承認された。
Kaggle push/run、Stage 1、inference、submissionは別承認とする。

## 6. Runtime例外と次のアクション

Kaggle CPU version 1の元のruntime projection gate FAIL
（`30,894.444 > 30,600 sec`）は監査履歴として保持し、PASSへ書き換えない。
ユーザーがこの程度の実行時間を許容し、Stage 1を明示承認したため、
runtime以外の13/13 PASSを前提に全773 wellsのStage 1を例外実行する。
科学gate、PF設定、保存control、truth-late規則は変更しない。inferenceと
submissionはStage 1結果後の別承認とする。

## 7. Version 2 recovery

canonical kernel version 2は全773 wellsのtarget-free成果物をfreezeした後、
truth-late readout中の保存exp209 HMM gzip-content checkで停止した。
freeze前のtruth/control/fold/hidden-like readは0で、3,783,989行の予測、
3,865行のrate bank、19,325行のcomponent ancestry、773行のwell auditは
完全である。

同じStage 1を再計算せず続行するため、これらをprivate DatasetへSHA固定した。
version 3はgzipまたはKaggle自動展開CSVのcontent SHA、prediction logical SHA、
件数、実行量、scientific contractを検証して全wellを再freeze扱いにした後、
同じtruth-late readoutだけを再開する。PF rerunは0。exp209 HMMは実使用列
`id,hmm_mean_tvt`の行数とstorage非依存SHAを必須とし、科学設定とgateは
変更しない。

## 8. Stage 1 result

version 3はtechnical 19/19をPASSした。candidate RMSE
`11.092618091`は保存exp404 `10.914522073`より`0.178096018 ft`悪く、
positive foldは1/5だった。high-missing scopeだけ`0.018240364 ft`改善し、
他の5 scope、by-well p95、worst well、固定HMM+PF 50:50 guardはFAILした。

equal-strata mixtureは一部の不確実wellへheadroomを持つ一方、観測が十分な
wellでも親tail30 modeの有効粒子数を一律に減らすため、全体・raw observed・
long-tail・hidden-likeの悪化が支配したと解釈する。事前登録どおり救済せず、
inference/submissionなしでterminal closeとする。
