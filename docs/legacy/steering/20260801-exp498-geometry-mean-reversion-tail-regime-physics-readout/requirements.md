# 要件

## 依頼

初回依頼では`geometry_mean_reversion_tail_regime_physics_readout`の実験ディレクトリと
steeringを作成し、設計だけを確定した。2026-08-01の追加依頼で、固定済み契約を変えず
compact self-contained diagnostic train notebookと契約testを実装する。Kaggle package、
scientific readout実行、推論、提出は追加依頼にも含めない。

2026-08-01の「実行してください」により、固定済み実行量のKaggle private CPU packageと
scientific readout 1回が追加承認された。推論、提出、親control再実行は引き続き含めない。

## 目的

exp490はfull OOF RMSEをexp357から`1.257040 ft`改善し、persistent episode SSEを
`41.409965%`削減した一方、by-well delta RMSE p95が`+7.257814 ft`、worst wellが
`+49.602560 ft`となった。この二極化が、truthやerrorを使わず観測できる単一の
物理regimeへ集中するかを、保存済みfull OOFとSHA固定raw inputだけで説明する。

## 実験識別

- 実験: `exp498_geometry_mean_reversion_tail_regime_physics_readout`
- Route: `pf_beam`
- 親: `exp490_geometry_centered_mean_reverting_offset_hmm`
- 比較対象: exp490に保存された`exp357_parent_prediction`
- 役割: saved-full-OOF physics diagnostic。CV候補、selector、予測モデルではない。

## 固定入力

exp490 merge v1の次の生成物を正とし、SHA不一致時はfail-closedする。

| 入力 | 行数 | 固定SHA |
| --- | ---: | --- |
| full OOF prediction（展開後content） | 3,783,989 | `e020e82e748a7836085657c4058070ff7853ed285639f2c2555cab721f9e9a07` |
| by-well metrics | 773 | `65abf0131d06473980d71410ae931aa21e4c5b4a90d7975024c32713ecdba076` |
| persistent episode metrics | 638 | `c34bb3d088a2dfea381ec72bb790eab61eb51f731d8d7863f72bad43d07e6d7b` |
| K16 segment contract | 12,368 | `53f6412c7a5489b33d2ca22bd47ca60b3f1e844b2e289228825659e6c1a50a30` |
| well manifest | 773 | `cac549f53ef4a98fce8e3fbf7381c0313f0f28f65409639a0e7d36cd89be7f5f` |
| exp490 scientific contract | - | `6398bbac380d3eca3a6255681b22c44c26de268ce6d4fad9dd242c066f2b9a35` |

known-prefix GR物理量だけは、exp490 shard decoder manifestに記録されたhorizontal / typewell
ファイルSHAと一致するraw trainを読む。horizontalのsuffix truth列`TVT`は読み込まず、
visible prefixの`TVT_input`、`MD`、`Z`、`GR`とtypewell `TVT` / `GR`だけを許可する。

## truth-late契約

1. 入力SHAとrow / well identityを検証する。
2. full OOF predictionからtruth / error / parent prediction / foldを除いたsafe columns、
   K16 contract、SHA固定raw/typewellだけでwell-level物理量を作る。
3. bucket境界、primary regime flag、feature table、feature contractのSHAを保存する。
4. freeze完了後にだけfold、by-well delta RMSE、episode SSEをjoinする。
5. outcomeを見たbucket変更、best bucket / interaction探索、閾値調整を禁止する。

## 固定物理量とbucket

- suffix horizon MD: `[0,4000]`, `(4000,6000]`, `(6000,inf)` ft。
- K16 median segment span: `[0,240]`, `(240,360]`, `(360,inf)` ft。
- known-prefix GR residual sigma: exp490 `prefix_stats`と同じstdを`[10,60]`へclipし、
  `[10,20)`, `[20,40)`, `[40,60]`。
- known-prefix GR information ratio:
  `(p95(typewell_GR_at_visible_TVT)-p05(...))/prefix_sigma`とし、
  `[0,1)`, `[1,2)`, `[2,inf)`。
- geometry disagreement proxy: suffix内`abs(exp226_pred-tvt_geop)`のmedianとし、
  `[0,2)`, `[2,10)`, `[10,inf)` ft。これはgeometry真値の不確実性とは呼ばず、
  geometry面とGR-corrected exp226面の観測可能な不一致proxyと呼ぶ。
- early offset evidence: `suffix_offset=0..31`の
  `abs(median(geometry_mean_reverting_delta_mean))`とし、
  `[0,1)`, `[1,5)`, `[5,inf)` ft。
- state uncertainty audit: suffix内`geometry_mean_reverting_hmm_std`のmedianを
  `[0,2)`, `[2,5)`, `[5,inf)` ftで記述するが、primary regimeには使わない。

## 事前固定primary regime

単一primary regimeを`weak_gr_geometry_conflict`とする。次をすべて満たすwellだけを1とする。

- `prefix_gr_sigma >= 40` または `prefix_gr_information_ratio < 1`
- `geometry_disagreement_median >= 10 ft`
- `early_abs_offset >= 5 ft`

suffix horizon、segment span、state uncertaintyのbucketはmechanism解釈用のsecondary表に
限定し、primary regimeの救済や再定義には使わない。

## outcomeと判定

- `harmful_well`: exp490 minus exp357 by-well RMSE `> +0.25 ft`。
- `catastrophic_tail_well`: 同差分`> +5.0 ft`。
- episode outcome: exp490 candidate SSE minus exp357 parent SSE。
- 判定はclassification modelや学習済みselectorを使わず、固定regime対complementの
  coverage、harm rate、delta RMSE、episode SSEをpooled / fold別に集計する。

物理regime支持は次のall-ANDとする。

1. regime coverageが全体20 wells以上、かつ4/5 foldsで各3 wells以上。
2. harmful-well rate ratio（regime / complement）がpooledで`>=1.5`。
3. harmful-well rateがregime側で高いfoldが4/5以上。
4. regime minus complementのmean delta RMSEがpooledで`>=+1.0 ft`。
5. mean delta RMSE差が正のfoldが4/5以上。
6. catastrophic tail wellの30%以上を、全wellの30%以下のregime coverageで捕捉する。

PASSしてもexp490を昇格・救済しない。観測可能な不確実性で復元力を弱める単一式を
次の別exp / 別steering / 別承認で設計してよい、という根拠だけにする。FAIL時は
mean-reversion tail regime原因追跡を終了する。

## 実行量契約

- scientific readout: 1
- target-free well feature aggregation: 773 wells
- truth-late readout folds: 5
- 新規HMM / prediction / model / LightGBM config / fold training / booster: 0
- PF / Beam / blend / selector: 0
- GPU: 0
- inference / submission: 0

## 禁止事項

- exp490のterminal fail-close再分類、gate緩和、rerun。
- truth / errorを入力とするwell / row selectorまたはregime作成。
- same-OOF gate、blend、conditional prediction、postprocess。
- half-life、Huber、state、noise、grid、bucket、primary regime閾値の探索。
- worst-well IDやfold outcomeを見たrule追加。
- inference、submission、Public LB確認。

## 受け入れ基準

- steering、実験config、README、SESSION_NOTES、metrics scaffold、result scaffoldが
  上記契約と一致する。
- `experiment.route=pf_beam`、状態はimplemented / not-runである。
- planned execution countがreadout 1、HMM / model / prediction / GPU各0である。
- input SHA、truth-late順序、fixed buckets、single primary regime、all-AND gate、
  terminal条件が機械可読configにも記録されている。
- 正規train notebookはJupytext percent形式のcompact self-contained実装を正とし、
  source解決、SHA guard、chunked Phase A、feature freeze、truth-late Phase B、固定all-AND、
  planned生成物保存をセル上で追える。
- 正規inference notebookは範囲外のplaceholderを維持する。
- 合成fixtureの契約test、py_compile、Ruff F821、Jupytext round-trip、strict experiment
  validationがPASSする。
- `execution.run_readout=false`、Kaggle package / run未承認を維持し、実データreadoutは
  この実装依頼では起動しない。

上記は実装依頼時点の受け入れ基準であり、後続の2026-08-01実行承認後は
`execution.run_readout=true`として固定済みreadoutを1回だけ実行する。

実行はKaggle private CPU version 2で完了した。technical gateはPASSしたがprimary
regimeは0 wellsでphysics gateをFAILし、固定terminal decisionに従って原因追跡を終了した。
