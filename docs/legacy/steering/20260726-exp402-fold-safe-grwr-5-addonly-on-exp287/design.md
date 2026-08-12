# 設計

## 結論

GRWR 6列を一括復旧しない。formation依存の5列だけをexp287のfold-safe
formation roleとclean-273内のtarget-free成分から再計算し、
exp287の421特徴へadd-onlyする。

親は`exp287_fold_safe_formation_74_addonly_on_exp264`とする。
exp396は親にせず、entropy依存の6列目も本実験へ含めない。

## 仮説

PF/Beam 5候補とdense-formation 3候補の値を個別特徴として与えるだけでは、
「候補bank全体がどの程度広がっているか」をLightGBMが一様に復元しにくい。
8候補の標準偏差・rangeと、既存のGR/DWT/FFT信頼度成分との固定interactionは、
候補bankの局所的不確実性を表す低次元の補助情報になり得る。

exp287ではdense-formation 3候補がfold-safe化され、いずれもformation blockの
gain上位だったため、旧GRWR 5列の無効理由を解消できる。一方、元になる候補値と
GR成分は既に親surfaceにあるので、冗長性により追加価値がない可能性も高い。

## 実験範囲

- 対象実験: `exp402_fold_safe_grwr_5_addonly_on_exp287`
- Route: `ml_model`
- 親実験: `exp287_fold_safe_formation_74_addonly_on_exp264`
- clean tail control: `exp264_exp263_candidate_confidence_dual_selector`
- historical formula source:
  `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- 変更する変数: fold-safeに再計算したGRWR 5列だけ
- 固定する変数:
  - clean 273 + nested compact 74 + fold-safe formation 74 = 421特徴
  - exp287 outer 5-fold GroupKFold、row identity、score rows、target residual
  - exp287 LightGBM configs `[0,1,2]`、early stopping 250、seed 42
  - score rowsの非加重RMSE、distance scopes、hidden-like 2面、by-well評価
- 最終特徴数: `421 + 5 = 426`
- 現在のscope: Stage 0 implementation-only。別名compact self-contained候補を
  実装済みで、正規Notebookと`settings.py`はtemplate placeholderのまま。

PF/Beam候補は補助特徴の入力で、最終TVTはdownstream LightGBMが生成するため
routeは`ml_model`とする。

## 比較対象と固定証拠

| 役割 | 実験 | 固定値 |
| --- | --- | --- |
| 主control | exp287 | RMSE `8.136708220359452` |
| clean tail control | corrected exp264 | RMSE `8.460811237612477` |
| formula source | exp218 | GRWR 5列の名前と演算順だけ |
| availability audit | exp264 | 5列の依存理由とclean allowlist |

- exp287 OOF SHA:
  `8f026c5c5f6508fb142981832994c6ba9cded4940168c648a9df9f3e698c3913`
- exp287 model manifest SHA:
  `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`
- exp287 feature schema SHA:
  `c1327324d6e0719eab45b9f8841033dd6cf09dd09228b044e6e8cc85f0fa8413`
- exp287 formation fold manifest SHA:
  `25611e281299991d626f1caca48673aee6225a890ad47ecdcd28a117ae827772`
- exp287 by-well SHA:
  `3562cec13abe3c3df496e57d71b46aeb592ea2022c7bf0b9b5df1e062c21024d`
- corrected exp264 OOF SHA:
  `b11c5005ca566f76588f4e1735386c15b8f016b874701a82e1c0741c8b839ae2`
- availability audit SHA:
  `6f93a502c9b58301e49da6effbf47b36d4635d4045157681749a762f08c89c67`

いずれかが一致しない場合はfit前に停止する。control boosterは再学習しない。

## 固定候補TVT

候補順と導出を次の8本に固定する。

1. `pf_ancc = float32(pf_ancc)`
2. `beam_mean = float32(last_known_tvt + beam_mean_d)`
3. `likpf_mean = float32(last_known_tvt + likpf_mean_d)`
4. `sc_ens = float32(last_known_tvt + sc_ens_d)`
5. `hyb = float32(last_known_tvt + hyb_d)`
6. `tvt_dense = float32(last_known_tvt + tvt_dense_d)`
7. `tvt_densew = float32(last_known_tvt + tvt_densew_d)`
8. `tvt_dense50 = float32(last_known_tvt + tvt_dense50_d)`

最初の5候補と`last_known_tvt`はclean 273から取得する。
最後の3候補は必ずmatching exp287 fold-safe formation roleから取得する。
全候補をfloat32でstackし、欠損補完や候補削除を行わない。1値でも非finiteなら
そのpartitionをFAILにする。

## 固定GRWR 5列

上記8候補のrow-wise stackを`C`とする。演算後は毎回float32化する。

1. `grwr_candidate_tvt_std`
   - `std(C, axis=0, ddof=0)`
2. `grwr_candidate_tvt_range`
   - `max(C, axis=0) - min(C, axis=0)`
3. `grwr_dwt_energy_ratio_w065_x_candidate_std`
   - `grwr_dwt_detail_energy_ratio_w065 * grwr_candidate_tvt_std`
4. `grwr_fft_rotation_ratio_x_candidate_range`
   - `grwr_fft_rotation_energy_ratio * grwr_candidate_tvt_range`
5. `grwr_dwt_minus_raw_ncc_gap_x_candidate_range`
   - `grwr_dwt_approx_minus_raw_default_candidate_ncc *
     grwr_candidate_tvt_range`

3つのGR/DWT/FFT成分はclean-273 allowlistに残る既存target-free列と同じ
名前・式に固定する。保存済み値を要求せず、raw GR / MD / TVT_input /
typewell GR・TVTから必要な3成分だけを最小再生成する。GRWR generator全体は
実行せず、horizontal truth TVTや旧exp218の5列値も参照しない。

## Fold-safe生成契約

### Outer-train

downstream outer fold `o`のmodel-fit rowsには、exp287が同じouter fold用に作った
outer-train self-excluded formation roleを使う。各train well自身をplane/dense
referenceから除外した`tvt_dense*` 3列を使い、全trainまたは別outer foldの
formation値を混ぜない。

### Outer-valid

outer fold `o`のvalid wellsには、そのouter-train wellsだけをreferenceにした
exp287 outer-valid formation roleを使う。valid wellおよび同じvalid foldの
他wellのformation/TVTを参照しない。

### Current-test

promotion PASSと別承認後のinferenceでは、exp287と同じraw-test flowで全train
wellsをformation referenceにし、target wellのformation列は読まない。
静的なpublic current-test feature fileをhidden inference入力として使わない。
5列はraw regenerationされた421列surfaceから同じ式で作る。

## 0-booster preflight

実装と実行は別承認とし、全technical gateをAND条件にする。

- pinned exp287 / exp264 / availability audit SHAが一致
- rows / wells / foldsが`3,783,989 / 773 / 5`
- parentのrow、well、fold、score-row、target、421列順が一致
- outer-train / outer-validのformation reference境界がexp287 manifestと一致
- 8候補と3 interaction sourceが全件finite
- 5列が宣言順、float32、全件finite、重複なし、parent 421列に不在
- historical exp218 GRWR値、exp111/exp396 score、entropy interactionのloadが0
- current-test raw generationが14,151 rows / 3 wells / 5列finite /
  target formation read 0
- outer-roleとcurrent-testごとにschema SHAとid-sorted float32
  logical-content SHAを保存
- model / booster / prediction / submissionが`0 / 0 / 0 / 0`
- 後続実行量が1 variant / 3 configs / 5 folds / 15 GPU boosters /
  control 0へ解決

preflight結果を見て候補subset、std/range、interaction、dtypeを変更しない。

### version 1終了後の分割設計

version 1はtrain-side 10 rolesとcurrent-test replayを一つのCPU runで直列実行し、
`CANCEL_ACKNOWLEDGED`、公開output 0件で終わった。保持ログから厳密な停止地点は
分からないため、科学処理を変更せず、次の3 runへ責務を分ける。

1. Stage 0A `train_roles`
   - 入力: exp072 train cache、exp287 artifacts、corrected exp264 OOF、
     competition raw train。
   - 出力: train source components 3列、outer fold×train/validの
     GRWR-5 10 partitions、partition manifest、input/SHA evidence。
   - current-test PF/Beamは実行しない。
2. Stage 0B `current_test`
   - 入力: competition raw train/test、bootstrap内のSHA固定exp072/exp218 source。
   - 出力: current-test 14,151 rowsのGRWR-5、replay/formation/content SHA manifest。
   - exp287 role partitionsとexp264 OOFは読まない。
3. Stage 0C `aggregate`
   - 入力: Stage 0A/0Bのprivate immutable Notebook outputs。
   - A/Bのimplementation source SHA、config SHA、scientific contract SHA一致を必須にする。
   - 10 partition file SHA、partition manifest SHA、source component file SHA、
     current-test file SHAを再計算する。logical-content SHAは生成phaseのledgerを
     immutable file SHAへ結び付ける。
   - 重いfeature、PF、Beam、formation生成は再実行せず、最終preflight /
     reproducibility manifestだけを保存する。

各runはCPU、internet off、model / booster / prediction / submission
`0 / 0 / 0 / 0`。A/Bは独立に実行でき、Cだけが両方の完了を要求する。
分割はruntime境界だけの変更であり、候補、式、fold、dtype、seed、logical SHA定義、
後続426列surfaceを変更しない。

## Downstream TVT add-only

preflight PASS、実装承認、Kaggle train承認後だけ実行する。

- source surface: exp287 421特徴
- added surface: fold-safe GRWR 5列
- final surface: 426特徴
- variant: `fold_safe_grwr_5_addonly`
- LightGBM configs: 3
- folds: 5
- 合計: 15 GPU boosters
- exp287 / exp264 control再学習: 0
- runtime: Nvidia Tesla T4、internet off、`gpu_use_dp=true`、
  `deterministic=true`、`force_col_wise=true`、threads 8

特徴量重要度はfold別gain/splitと平均を保存し、5列groupの利用率をreportする。
重要度を見て列を削除・追加することはしない。

## Promotion gate

全項目をAND条件とする。

1. exp287比pooled OOF delta RMSE `<= -0.02 ft`
2. 5 folds中4 folds以上でexp287以下
3. near / mid / 1000+ / hidden-like spatial /
   hidden-like typewell-purgedが各exp287比`<= +0.02 ft`
4. by-well delta RMSE p95がexp287比`<= 0.00 ft`
5. corrected exp264比worst-well delta RMSE `<= +0.25 ft`
6. corrected exp264比`+1/+3/+5 ft`悪化well数が`135/39/14`以下

FAIL時はbranchを閉じる。列subset、5候補だけのspread、dense候補だけのspread、
interaction削除、weight、threshold、gate緩和による同一OOF救済を行わず、
inference、submissionへ進めない。

## 本実験に含めないもの

- historical exp218 GRWR 5列の値
- `grwr_ll_entropy_x_dwt_energy_ratio_w065`
- exp396 score-27またはexp111 score
- full-train formation referenceを使ったOOF
- candidate subset、spread、interaction、feature、model、thresholdのgrid
- sample weight、error-segment weight、hard row/well gate
- candidate hard selection、TVT direct correction、blend
- exp287 / exp264 control再学習
- gate緩和、same-OOF rescue
- promotion PASSと別承認前のinference、submission

## 再現性設計

- seed policy: feature生成にRNGなし。foldとdownstream seed 42はexp287を継承。
- stochastic処理: 別承認後のGPU LightGBMだけ。
- train側PF/Beam / HMM: 保存済みtarget-free値のload-only、再実行0。
- current-test側: exp072 source SHAを固定し、feature family / split / wellから作る
  stable seedでPF ANCC 3、PF Z 3、Beam path 21、likelihood-PF 3 well-runs /
  384 seed-well trajectories / 192,000 particle startsを再生成する。
- 並列処理: feature式は決定論的でglobal RNGなし。row orderをidで固定して
  logical-content SHAを計算する。
- GPU: exp287のDP/deterministic/force_col_wise/thread設定を継承するが、
  bitwise deterministic anchorとは主張しない。
- train/current-test: outer-roleごととraw current-testについてrow count、
  well count、schema、float32 logical-content SHAを保存する。
- gzip: raw gzip SHAではなくdecompressed content SHAを主証拠とする。
- model/prediction: 学習承認後は15 model manifest SHA、OOF SHA、
  fold/scope/by-well metrics SHAを保存する。
- inference/submission: promotion PASSと別承認後だけtest prediction content SHA、
  submission SHA、submit-check、kernel versionを記録する。
- Kaggle bootstrap: package作成時にmetadata、embedded config、source、
  input SHA、internet/GPU設定を照合する。

## リスク

- リークリスク: 旧`tvt_dense*`はfull-train fitなので禁止。必ずmatching exp287
  fold roleを使う。
- 冗長性: 親が8候補値と3 interaction sourceを既に持つため、5列が不要な可能性がある。
- CV/LB不一致: exp287はPublic LBを改善したがtrain-side tail guardはFAILしている。
  global改善とtail安全性を分離して判定する。
- attribution: exp218 GRWR 86列全体の改善を5列の根拠として扱わない。
- runtime/memory: 新しいfitは15 boostersだけだが、421+5列のfold-role wide
  matrixを扱う。chunked matrix assemblyとpeak RSS表示を実装時に必須とする。
- 再現性: float32演算順、candidate order、row orderの差でSHAが変わるため、
  formula fixtureとlogical-content SHAを固定する。

## 現在の承認境界

2026-07-26の「exp402を実装してください」「実行してください」により実装、
正規Notebook採用、version 1 Stage 0 runを承認済み。version 1未完了後の
「設計変更と再実行を進めてください」により、上記Stage 0A/0B/0Cの実装、
private CPU package/push/runも承認済み。15 GPU booster train、inference、
submissionは未承認・未実行。

## 2026-07-28 aggregate path alias hotfix

aggregate version 1はfold 4の旧canonical input pathが存在せず、同名sentinelを
持つfold 0–4の5 rootをfallback候補にして一意性guardで停止した。
upstream 7 runのconfig SHAとcompact implementation source SHAを維持するため、
両ファイルは変更しない。

version 2ではaggregate wrapperがconfigを読みscientific contractを検証した後、
in-memoryのfold 4 `artifact_root_patterns`先頭へ実在する
`/kaggle/input/notebooks/kentookumura/exp402-foldsafe-grwr5-train-fold4-v2/artifacts`
を追加する。config file SHAの計算対象はdisk上の未変更fileなので、
upstream manifestとのexecution identity比較は不変となる。

修正範囲はaggregate wrapper、対応Notebook、push package、回帰testだけ。
候補、式、fold、dtype、seed、partition、logical SHA、promotion gate、
model / booster / prediction / submission数は変更しない。
