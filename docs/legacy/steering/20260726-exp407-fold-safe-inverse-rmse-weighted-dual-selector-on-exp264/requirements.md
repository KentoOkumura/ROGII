# 要件

## 依頼

`exp264_exp263_candidate_confidence_dual_selector`の共有selectorでは、12候補が
candidate-long形式で同数の学習行を持つ。`beam_mean`のように全体RMSEが高い候補が
同じ重みで学習されることが、共有LightGBMのcapacityを消費し、より有力な候補の
誤差・確率推定にnegative transferを起こしているかを検証する。

候補を削除せず、selectorの各fit partition内で計算した候補別TVT RMSEの逆数に比例する
task weightを、`pred_abs_error`と`p_within10`の両目的の学習時だけに適用する。
今回はbacklog、実験scaffold、steering、config、評価契約を作成して設計を確定する。
実装、学習、推論、提出は行わない。

## 2026-07-26 実装承認追記

ユーザーの「exp407を実装してください」により、固定済みStage B契約の
implementation-onlyを承認済みとする。別名のcompact self-contained train候補、
fit-partition限定weight、共有Stage Bの後方互換hook、truth-read / sampling / SHA監査、
親v5比較の全AND gate、synthetic contract testまでを実装範囲とする。
Kaggle Stage B 10 CPU boostersの実行、正規Notebook採用、Stage C/D、inference、
submissionはこの承認に含めない。

## 仮説

- `beam_mean`は保存済みOOF上でlast-known TVTからの差が小さい行が多い一方、
  候補単体RMSEは15.774327 ftで、12候補中の弱い側にある。
- ただし、学習後にBeamだけをscore対象から除外した既存readoutはhard RMSEを悪化させたため、
  Beam削除は行わない。そのreadoutは、Beam行が学習時に共有modelへ与えるnoiseを検証していない。
- 候補別fit RMSEの逆数を穏やかに正規化・clipしたsample weightなら、候補bankと
  推論契約を変えずに共有modelの学習配分だけを変え、negative transfer仮説を一因子で検証できる。

## 制約

- Route: `ml_model`
- 親実験: `exp264_exp263_candidate_confidence_dual_selector`
- 12候補、宣言順、candidate値、88列raw-test-safe feature schema、2つのlegal domain、
  2 objectives、outer 5 folds、deterministic sampling、LightGBM設定を固定する。
- `beam_mean`を含む候補を削除・追加・置換しない。
- weightは各modelの正確なfit partition内だけでtrue TVTをjoinして計算する。
  outer-valid、inner-valid、hidden-like、current-test、全OOF集計のtruthは使わない。
- Stage Bでは各outer-trainのdeterministic sampled base rowsだけから候補別RMSEを計算する。
- 将来Stage Cへ進む場合は、各outer × inner modelのsampled inner-train rowsだけから
  weightを再計算する。outer-validとinner-validのtruthを使わない。
- 同じcandidate weightを`pred_abs_error`と`p_within10`のfitへ適用する。
  validation、early stopping、OOF metric、gate metricはすべてunweightedで計算する。
- 全OOFで計算した候補別RMSEとillustrative weightはsanity check専用で、
  fit時のweightとして使用しない。
- inverse-square、指数・clip・候補subset・candidate別手調整・grid searchは行わない。
- Stage B実装は2026-07-26に承認済み。Kaggle実行は別のユーザー明示承認を必要とする。
- 親Stage B controlは保存済みv5を使い、再学習しない。Stage B承認時の新規学習は
  1 variant × 2 objectives × 5 folds = 10 CPU boostersだけとする。
- Stage C、Stage D、inference、submissionはStage Bの全gate通過後も自動実行せず、
  それぞれ別承認を必要とする。
- 再現性は`docs/06_reproducibility.md`に従い、input、sampling、feature schema/content、
  weight table、model manifest、OOF prediction、Kaggle package/kernel versionのSHAを記録する。

## weight契約

候補`c`、fit partition`p`について次を固定する。

1. `rmse[p,c] = sqrt(mean((candidate_tvt - true_tvt)^2))`
2. `raw[p,c] = 1 / max(rmse[p,c], 1e-6)`
3. `normalized[p,c] = raw[p,c] / mean_c(raw[p,c])`
4. `clipped[p,c] = clip(normalized[p,c], 0.5, 1.5)`
5. `weight[p,c] = clipped[p,c] / mean_c(clipped[p,c])`

`weight`はfinite、候補数12、平均1.0、最終範囲`[0.5, 1.5]`を必須とする。
最終再正規化で範囲外になる場合はtechnical errorとして停止し、clipを事後変更しない。
各candidate-long行は、その行のcandidate IDに対応する同一partitionの`weight[p,c]`を持つ。

## Stage B受け入れ基準

### Technical gate

- 親v5と同じ3,783,989 base rows、45,407,868 candidate-long OOF rows、
  12候補、候補ごと3,783,989 OOF rowsである。
- candidate IDと順序、2 legal domain、88列feature schema logical SHA
  `aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`
  が親と一致する。
- outer-train / outer-valid well overlap、valid truth read before fit、weight欠損・非finite、
  candidate coverage errorがすべて0である。
- 各outer foldのfit row IDsとdeterministic sampling manifestが親v5 contractと一致する。
- fold別にraw RMSE、raw inverse、normalized、clipped、final weightを12候補すべて保存し、
  mean weightは`abs(mean-1) <= 1e-12`、最終weightは`[0.5, 1.5]`である。
- validation datasetにはsample weightを渡さず、primary/secondary metricはunweightedである。
- 10/10 modelを生成し、model manifestとcandidate-score OOFのSHAを検証する。

### Scientific gate

保存済み修正版Stage B v5を唯一のcontrolとし、次をすべて満たす。

- unweighted expected-error MAEが`<= 3.7858011626240818`
  （親3.7958011626240817から0.010 ft以上改善）。
- expected-error MAEが5 folds中4 folds以上で親より改善する。
- unweighted within10 loglossが`<= 0.3604715948027694`、
  Brierが`<= 0.11295097922004564`である。
- within10 loglossとBrierが、それぞれ5 folds中3 folds以上で親以下である。
- hard primary top1 RMSEが親`8.587004386703422`以下である。
- hard primaryが5 folds中3 folds以上で親以下である。
- near、1000+、hidden-like spatial、hidden-like typewell-purgedの
  hard RMSE deltaが親に対してそれぞれ`<= +0.02 ft`である。
- worst-well RMSE regressionが親に対して`<= +0.25 ft`である。
- 候補別unweighted score/calibrationと候補別選択率をすべて保存する。
  これらは診断であり、同じrun内のweight、gate、candidate subset変更には使わない。

### 判定

- technical gateとscientific gateがすべてPASSした場合だけ、Stage Cの実装・40 CPU
  booster実行を別承認候補にできる。
- scientific gateが1つでもFAILした場合はbranchを閉じ、weight exponent、clip、
  candidate subset、Beam削除、objective別weightで救済しない。
- technical errorの場合だけ、同じ固定契約の技術修正を行い、再実行は別承認を得る。
- deterministic anchorとして扱うのは同一packageの再実行でmodel/OOF content SHA parityを
  確認した後だけとする。
- gzip生成物を比較する場合はraw `.csv.gz` SHAではなくdecompressed content SHAを主証拠にする。
