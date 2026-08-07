# 要件

## 依頼

exp490をstandalone予測として採用せず、long suffixで強くなる補正だけを固定fadeした
13番目のselector候補として全体MLへ渡す。バックログ、実験ディレクトリ、steeringを作成し、
設計を確定する。今回は実装、Notebookロジック作成、学習、Kaggle package/run、推論、提出を
行わない。

## 2026-08-02 実装承認

ユーザーの `exp505を実装してください` により、Stage Cのcompact self-contained候補と
contract testの実装を承認済みとする。既存の正規train Notebook placeholderの上書き、
canonical採用、Kaggle package / CPU run、Stage D実装 / GPU run、inference、submissionは
この承認に含めない。

## 2026-08-03 Stage C実行承認

ユーザーの `実行してください` により、compact train候補の正規train Notebookへの採用、
Kaggle CPU package、push/runを承認済みとする。実行範囲は固定済みの
1 variant × 2 objectives × outer 5 × inner 4 = 40 CPU boostersだけで、parent/control再学習、
HMM / PF / Beam再実行、GPU、Stage D、inference、submissionは含めない。

## 2026-08-03 Stage C実行結果

Kaggle private CPU version 1で40 modelsを完走し、technical checksは全PASSした。hard OOFは
`8.243315437`でraw exp501 `8.264890209`から`0.021574771 ft`改善し、4/5 folds、固定7 scope、
fade利用条件も通過した。しかしby-well p95縮小`0.000036536 ft < 0.10 ft`、worst縮小
`0.173168079 ft < 1.0 ft`でtail 2条件をFAILした。契約どおりStage Dを許可せず、
same-OOF rescue、inference、submissionなしで終端閉鎖する。

## 仮説

exp490はprefix直後512 rowsではexp357より悪化する一方、512--1024 rowsから改善し、
long suffixほど改善幅が大きい。exp503で固定式
`1 - exp(-md_since / 500)`をexp490補正へ掛けると、always-exp490
`8.480155260`から`8.447032560`へ改善し、5/5 foldsで非劣化だった。

raw exp490をそのままfixed13 bankへ入れたexp501はselector pooled OOFを
`8.652531956 -> 8.264890209`へ改善したため候補価値はあるが、by-well p95 / worstは
`+2.904594 / +18.394664 ft`でtailを壊した。raw候補をtau=500 fade候補へ置換すれば、
候補価値を残しつつprefix直後の不要な補正とcandidate-bank rerankingを弱め、最終的に
exp413 downstream TVT MLへ安全なselector表現として渡せる可能性がある。

ただしraw exp501 compact77をそのままexp413へ置換したexp502は、pooled gainが
`0.002658891 ft`に留まり、fold 3 / 4を`+0.116027 / +0.234686 ft`、hidden-like 2面を
約`+0.14 ft`悪化させてFAILした。exp505はこのFAILを再分類せず、fadeによるselector-tail
改善をStage Dの必須先行条件にする。

## 実験範囲

- 実験: `exp505_exp490_tau500_fade_fixed13_on_exp413`
- Route: `ensemble`
- selector親: `exp501_exp490_mean_reverting_hmm_fixed13_selector_on_exp264`
- fixed12親: `exp264_exp263_candidate_confidence_dual_selector`
- fade根拠: `exp503_exp490_strength_weakness_prefix_policy_readout`
- downstream親: `exp413_scale5_likpf_full_replacement_on_exp335`
- raw downstream比較: `exp502_exp501_fixed13_selector_replacement_on_exp413`
- Stage C: raw exp490候補をtau=500 fade候補へ1対1で置換し、strict-nested fixed13
  selectorを評価する。
- Stage D: Stage C gateを通った場合だけ、exp413 nested compact74を新compact77へ置換する。
- inference / submission: 対象外。PASS後も別途承認を必要とする。

## 固定要件

- fade式は
  `p_fade = p_exp357 + (1 - exp(-md_since / 500)) * (p_exp490 - p_exp357)`、
  `alpha=1`、`tau=500 ft`に固定する。
- hard cutoff、tau / alpha grid、相対深度gate、well gateを試さない。
- fixed12 bank、fixed7 fallback、candidate順、outer 5 / inner 4 folds、2 objectives、
  sampling、LightGBM設定、評価scopeをexp501から変更しない。
- 13番目はfade候補だけとし、raw exp490とfade exp490を同時に保持しない。
- exp490 HMM、exp357、fixed12候補、exp413 control、exp413 signed selectorを再生成・再学習しない。
- exp490 sourceからfeature freeze前に読める列をkey、`md_since`、exp490 prediction、
  exp357 parent prediction、既存2 native confidenceだけに限定する。
- source fold、truth、error、role、episode、scope、by-well outcomeをfeature / split / gateへ
  使用しない。
- exp503のtau=500は同一full OOF上の29 profileから得た探索的根拠であり、exp505を
  独立なclean validationまたはdeterministic submission anchorと表記しない。
- Stage C FAIL時はStage Dを実装・実行せず、tau / alpha / threshold / featureの救済を行わない。

## 将来の実行量

- Stage C: 1 variant × 2 objectives × outer 5 × inner 4 = 40 CPU selector boosters。
- Stage D: Stage C PASSかつ別承認後のみ、1 treatment × 3 configs × outer 5 = 15 GPU boosters。
- 全段階を実行した場合の新規booster上限: 55。
- parent/control selector retraining: 0。
- exp490 / exp357 / PF / HMM / Beam再実行: 0。
- Stage DのGPU 15 boostersは、実行前に改めてユーザー承認を得る。

## 設計完了の受け入れ基準

- steering 3文書、experiment scaffold、`config.yaml`、README、SESSION_NOTES、result、
  metricsを設計のみの状態で作成する。
- train / inference Notebookはcode cell 0のmarkdown-only placeholderにする。
- fade式、入力allowlist、candidate置換、fold-safe selector、Stage C / D gate、実行量、
  禁止事項、再現性を曖昧さなく固定する。
- `KAGGLE_DIRECTION.md`へP2 backlogとして追加し、`experiment_summary.md`へ記録する。
- implementation / canonical Notebook / package / run / inference / submissionの承認をfalseにする。

## 将来のStage C科学的PASS

次を全ANDで満たす。

- technical / leakage / fixed-fallback parityを全PASSする。
- fixed fade direct predictionをRMSE `8.447032560`、許容誤差`1e-6 ft`以内で再現する。
- fade fixed13 hard OOFがraw exp501 `8.264890209`を悪化させない。
- raw exp501比で4/5 folds以上が非劣化する。
- exp501と同じ固定7 scopeがraw exp501比`+0.02 ft`以内である。
- fade候補top1利用が全体0.5%以上かつ4/5 folds以上で正である。
- fixed12比by-well p95 deltaをraw exp501より`0.10 ft`以上縮小する。
- fixed12比worst-well deltaをraw exp501より`1.0 ft`以上縮小する。

このtail gateはexp501のabsolute tail FAILをPASSへ再分類するものではなく、raw候補より
materialに安全化した場合だけdownstream評価を許可するprogression gateである。

## 将来のStage D科学的PASS

Stage C PASSかつ別承認後にのみ判定し、次を全ANDとする。

- saved exp413 control `7.884802794`からpooled RMSEを`0.03 ft`以上改善する。
- 3/5 folds以上でexp413に非劣化する。
- `md_since 0--250 / 250--1000 / 1000+`、hidden-like spatial、
  hidden-like typewell-purgedの5 scopeがexp413比`+0.02 ft`以内である。
- by-well RMSE deltaのp95 / worstがexp413比それぞれ`+0.25 ft`以内である。
- technical / fold / role / feature-surface guardを全PASSする。

PASSでもinference実装、current-test生成、submissionは自動承認しない。
