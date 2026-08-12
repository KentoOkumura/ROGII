# 設計

## 仮説

target-freeなexact-group availabilityとglobal/identity fallbackを先に固定すれば、
exp311のsame-group平均gainとworst-well悪化を、prior値の再推定なしに分離できる。

## アプローチ

`exp311`で凍結済みのType Well群統計を変更せず、各outer foldでpeer数、effective rows、
group availability、fallback sourceを先に決定する。評価面は
`same-group held-out well`、`leave-one-typewell-group-out`、
`spatial+typewell-group-purged`の3面に固定する。

exact groupが利用不能ならglobal prior、それも利用不能ならidentity/no-correctionへ落とす。
global priorは同じfold/surfaceのexp311 exact-group priorのうちexp352 support条件を満たす
群から、対象wellの群を除外して各prior値のequal-group medianを取る。利用可能な他群が
1群もなければidentityへ落とす。この集約方法は固定し、同一readoutで重みや閾値を変えない。
この決定をtruth-free manifestとしてSHA固定した後だけsuffix truthを結合し、平均gainと
negative transfer、worst-well safetyを評価する。`exp311`で失敗したgroup fit-RMSE予測を
作り直さず、`exp312`のconditional emissionも使わない。

## 実験範囲

- 対象実験: `exp352_typewell_transfer_safety_guard_readout`
- Route: `pf_beam`
- 科学的参照: `exp311_typewell_group_prefix_suffix_gr_calibration_readout`
- 履歴参照: `exp313_typewell_group_unseen_transfer_guard`
- 変更する変数: Type Well群priorの利用可否と固定fallbackだけ。
- 固定する変数: exp311 fold/group/prior値、peer/support閾値、3 split surfaces、score、fallback順。
- 実行量: 1 diagnostic / 3 audit surfaces / 5 folds / model・booster・decoder・HMM各0。
- PASS後の範囲: guard readoutの採用可否まで。補正、ML feature、candidate rank、whiteningは別実験・別承認。
- score unit: exp311保存値と同じ`horizontal_gr_api`。事前登録した閾値の数値は維持し、
  TVT予測を生成しない本実験では誤っていた`ft`表記だけを訂正する。

## 再現性設計

- seed policy: RNGなし。保存済みfoldとgroup/wellの辞書順を固定する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: RNGを使わず、集計順とthread数を固定する。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU/internet off、上限30分。
- train cache / test feature regeneration の SHA 記録方針: exp311 input、fold/group manifest、
  availability/fallback table、3面score tableのschema/content SHAを保存する。
- model manifest / prediction / submission SHA 記録方針: model/prediction/submissionを生成しないため非該当。
- Kaggle package bootstrap 確認方針: package承認後にcanonical configとbootstrap内configのSHA一致を確認する。

## リスク

- リークリスク: truth/errorを見てavailabilityやfallbackを選ぶとguardが無効になる。manifestを先に凍結する。
- CV/LB 不一致リスク: trainのType Well群coverageがhidden testを代表しない可能性があるため、
  leave-group-outとspatial+typewell purgeを必須にする。
- ランタイム/メモリリスク: 0-model集計のみで低い。3面のpair tableはstreaming集計可能にする。
- 再現性リスク: exp311大容量出力の取得元混同を防ぐためkernel versionとdecompressed SHAをhard preflightする。
- 解釈リスク: PASSしてもexp311/312の科学FAILを取り消さず、旧downstreamを自動解禁しない。

## 実装境界

- compact self-contained train候補はexp311 summary、fold、membership、group prior、
  score population、pair tableのSHAをhard preflightする。
- availability/fallback manifestはsuffix pairをmaterializeする前にcontent SHAを固定する。
- pair tableはmanifest freeze後だけ読み、well単位のhorizontal-GR RMSEを計算する。
- 正規Notebookは初回scaffoldのまま維持し、compact候補の採用とKaggle実行は別承認とする。
- inferenceはfail-closedで、model、booster、decoder、HMM、prediction、submissionを生成しない。

## 次のアクション

compact候補の採用承認後に正規train notebookへ反映し、別の実行承認後だけKaggle CPU
Stage 0を実行する。PASSしても補正や後続実験へ自動進行しない。
