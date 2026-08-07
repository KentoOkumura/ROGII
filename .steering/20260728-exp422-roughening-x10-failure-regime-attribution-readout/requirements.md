# 要件

## 仮説

exp416のroughening x10は全wellへ一律適用すると失敗したが、失敗は一様ではない。
roughening後も粒子が崩壊しやすいwellでは回復余地があり、GR欠損と長いsuffixによる
無観測伝播が大きいwellではrougheningノイズの累積損傷が上回る。この2軸を正解非依存の
保存診断だけで固定すれば、persistent-offset改善と全体悪化の符号分岐を説明できる。

## 依頼

`exp422_roughening_x10_failure_regime_attribution_readout`の実験ディレクトリと
steeringを作り、実装前に入力、regime定義、評価、判定、禁止事項を確定する。
今回は設計だけとし、コード、Notebook、Kaggle package、実行、推論、提出は作らない。

## 制約

- Route: `pf_beam`
- 親: `exp416_roughening_x10_likpf_full_oof_ablation`
- exp416のterminal FAIL、CV `13.617717557749454`、exp072比較
  `+2.022823162106759 ft`、decision
  `roughening_x10_rejected_close_without_rescue`を変更しない。
- exp416 merge kernel `kentookumura/exp416-rough-x10-merge` version 2 /
  id_no `128912230`とartifact manifest SHA
  `708bb257e3ab360f09821823d5413fa9e1c5c32ef4ddb4917b41573943dffb86`
  を固定する。
- exp416のmerged prediction、well audit、by-well metrics、persistent episode
  metricsだけを主生成物として使う。必要なcontrol / truth / foldは保存済みexp072 /
  exp226から読み、新しい予測を生成しない。
- regimeはtruth、error、candidate-control gain、episode outcomeを読む前に、well auditと
  reporting foldだけから固定し、内容SHAを保存する。
- primary regimeは、4つのPF崩壊診断を等重みした
  `recovery_pressure_score`と、2つの無観測伝播診断を等重みした
  `damage_exposure_score`の2軸だけとする。
- 各指標の順位化とhigh / low閾値は、対象fold以外の4 foldsだけで作る経験分布関数と
  中央値を使う。truthを見た閾値、bucket、feature、方向の選択をしない。
- primary target cellは
  `high recovery pressure AND low damage exposure`の1セルに固定する。
- 位置readoutはnormalized suffix progressの固定4分割、raw-GR observed / missing、
  `md_since >= 1000 ft`だけとし、row-level adaptive ruleは作らない。
- 新規PF、prediction、model、LightGBM、HMM、Beam、GPUはすべて0。
- 実装と実行はそれぞれ別のユーザー承認を必要とする。

## 受け入れ基準

### Technical

- exp416 manifest、kernel version、scientific contract、terminal decisionを固定値と照合する。
- 3,783,989 rows / 773 wells / fold 0--4、well audit 773 rows、
  by-well metrics 773 rows、persistent episodes 16 / 12 wells / 55,104 rowsを
  欠落、重複、非finiteなしで確認する。
- exp416 candidate / exp072 controlのpooled RMSEを
  `13.617717557749454 / 11.594894395642696 ft`へ`1e-6 ft`以内で再現する。
- regime feature / assignment / row-scopeをoutcome読込前にfreezeし、
  schema / logical content SHAを記録する。
- 実行量がsaved-output readout 1、new prediction 0、PF / model / booster /
  HMM / Beam / GPU 0と一致する。

### Scientific attribution

以下をANDで満たす場合だけ
`target_free_regime_attribution_supported`とする。

1. `recovery_pressure_score`とwell等重みgainのSpearman相関が`>=0.10`、
   正方向が`>=4/5 folds`、fold内4096回置換の片側p値が`<=0.025`。
2. `damage_exposure_score`とwell等重みgainのSpearman相関が`<=-0.10`、
   負方向が`>=4/5 folds`、fold内4096回置換の片側p値が`<=0.025`。
3. 固定target cellのrow-weighted RMSE gainが`>=0.05 ft`で、
   `>=4/5 folds`を改善する。
4. target cellと残りのwell等重みmean gain差が`>=0.25 ft`で、
   target cellの改善well率が`>=0.50`。
5. target cellにpersistent episodeが`>=4 episodes / >=3 wells`存在し、
   同セルのepisode SSE reductionが`>=5%`、かつ全improved episodesの正の
   SSE reductionの`>=50%`を説明する。

1つでもFAILなら`no_reproducible_target_free_regime_close_attribution_branch`で閉じる。
PASSしてもadaptive rougheningの有効性や推論可能性は確定せず、別実験の
事前登録を検討できるだけとする。

## 実装後に生成する予定のもの

- Jupytext percent形式compact self-contained train候補と専用contract tests
- target-free `regime_feature_freeze.csv` / `regime_assignment.csv`
- `row_scope_freeze.csv.gz`
- fold / regime / position / individual-diagnostic readout
- persistent-episode regime readout
- technical / scientific gate、summary、artifact manifest、各SHA

## 次のアクション

実装の明示承認を待つ。承認前にコード、正規Notebook、package、push、実行を行わない。
