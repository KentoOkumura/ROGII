# 設計

## アプローチ

exp287の保存済みOOF予測を一切再学習せず、各wellの既知prefix末端と未知suffix先頭の`U = TVT + Z`だけを接続するdeterministic postprocessを1回評価する。

公開Notebookは6.594系の複数layerを含むが、本実験が参照するのはU境界fadeの式だけである。公開Notebook本体やpretrained packageを実行せず、同じ式をexp287の固定OOF surfaceに独立実装する。これにより、public同名well依存のcontact overrideとU補正の効果を分離する。

## 数式契約

well `w`について、raw horizontalの最後のfinite `TVT_input`行を`k`、最初の未知suffix行を`k+1`とする。

```text
U_last(w) = TVT_input[w, k] + Z[w, k]
gap_U(w)  = parent_pred[w, k+1] + Z[w, k+1] - U_last(w)
d[w, i]  = MD[w, i] - MD[w, k]
move[w, i] = -clip(gap_U(w), -8.0, 8.0) * exp(-d[w, i] / 240.0)
candidate[w, i] = parent_pred[w, i] + move[w, i]
```

- 計算はfloat64で行い、保存時dtypeとroundingをmanifestへ記録する。
- 最初のunknown行も`d > 0`なので補正率は厳密には1未満である。`d=0`へ置換しない。
- correctionは全suffix行に同じ境界gapから適用し、途中でgapを再推定しない。
- `clip`はgapにだけ適用し、candidate TVT、row move、Uへ追加clipを行わない。
- gap thresholdやconfidence gateを設けない。gapが0なら式の結果としてno-opになる。

## 入力とfreeze境界

### Generation phase（truth禁止）

1. exp287 train v5の`fold_safe_formation_oof_predictions.parquet`をSHA検証し、親予測列`fold_safe_formation_74_addonly__lgb_mean__pred_tvt`だけを読む。
2. raw train horizontalを`MD/Z/TVT_input`だけで読み、finite prefix、contiguous NaN suffix、ID=`{well}_{raw_row_index}`を構築する。
3. OOF ID／well／row順とraw suffixを完全一致させる。欠損well、重複ID、非連続suffix、非単調MD、`d<=0`は全run ERRORとする。
4. 上記固定式でcandidate、gap、move、fixed gap bucketを作る。
5. `target_free_candidate.parquet`、`u_boundary_diagnostics_pretruth.csv`、input/schema/config manifestのcontent SHAを保存する。
6. predictionとdiagnosticを再読込してSHA一致を確認するまでtruthをロードしない。

### Evaluation phase（late join）

1. freeze済みIDへexp287 OOFの`actual_tvt`と`outer_fold`をjoinする。
2. exp115固定assignment SHA `5f9ac9fac6bb3725a7c613f09856a85bdf73b8206fd2edf1b79e8eaa9bca6597`をjoinし、hidden-like spatial／typewell-purgedを評価する。
3. pooled、fold、distance、gap、by-well metricsを親とcandidateのpaired比較で保存する。
4. 全technical gateとscientific gateのANDで`PASS_FOR_INFERENCE_REVIEW`または`FAIL_CLOSE_NO_RESCUE`を決める。PASSしてもinferenceへ自動移行しない。

## 固定bucket

- distance: `0--64`, `64--128`, `128--240`, `240--480`, `480--1000`, `1000+` MD-ft。primary boundary集計として`0--240`も出す。
- absolute gap: `[0,1)`, `[1,2)`, `[2,4)`, `[4,8)`, `[8,+inf)` ft。
- gap sign: negative / zero / positive。
- すべてのbucket edgeとlabelをtarget/error join前にconfigからfreezeする。

## Promotion gate

### Technical AND gate

- 親OOF SHA、3,783,989 rows、773 wells、unique ID、親CV `8.136708220359452`の`1e-6` parity。
- raw suffixとOOFのID／well／rowの完全一致、全well coverage 1.0。
- generation phaseでtarget／actual TVT／error／oracle access 0。
- finite prediction、finite gap/move、`max(abs(move)) <= 8 + 1e-10`。
- 各wellで`abs(move)`がMD順に非増加。
- 各well最初のunknown行で`abs(gap_after) <= abs(gap_before) + 1e-10`。
- correction signがnonzero gapと常に逆、`move`と式のmax abs parity `<=1e-10`。
- prediction／diagnostic／schema／config／input manifest SHAの保存と再読込一致。

### Scientific AND gate

- pooled RMSE gain `parent - candidate >= 0.020 ft`。
- candidateがparentを改善するouter fold `>=4/5`。
- 0--240 ft RMSE gain `>=0.050 ft`。
- RMSE delta `candidate - parent`が240--480 ft `<=+0.020 ft`、480--1000 ft `<=+0.010 ft`、1000+ `<=+0.005 ft`。
- hidden-like spatial／typewell-purgedのdeltaが各`<=+0.020 ft`。
- by-well RMSE delta median `<=0.0 ft`、p95 `<=+0.10 ft`、maximum `<=+0.50 ft`。
- `delta_rmse > +0.25 ft`のwell数が`delta_rmse < -0.25 ft`のwell数以下。

## Failure policy

どれか1つでもFAILした場合は`FAIL_CLOSE_NO_RESCUE`とする。同じOOFでcap、tau、gap threshold、sign、fade形、適用distance、well subset、blend weight、親予測を変更しない。診断上の特定bucketだけが良くても、そのbucketを使ったhard gateやfar/near-only variantをexp349内で追加しない。独立した新しい仮説を作る場合は、exp349の結果を記録して別途ユーザー確認を得る。

## 実験範囲

- 対象実験: `exp349_exp287_u_boundary_continuity_fade`
- Route: `ml_model`
- 親実験: `exp287_fold_safe_formation_74_addonly_on_exp264`
- 変更する変数: 親TVT予測へ固定`cap=8.0 / tau=240.0`のU境界fadeを1回だけ加える。
- 固定する変数: exp287 OOF／fold／parent model、raw prefix/suffix、formula、全bucket、metric、gate、hidden-like assignment。
- 実行量: 1 postprocess variant、5 reporting folds、trained fold/model/config/booster/PF/Beam/HMM/control再学習/GPU `0/0/0/0/0/0/0/0`。
- 優先順位: 中・P1。GPU不要で現行ML submitted anchorを直接監査できるため、未実装GPU P2/P3候補より先に置く。ただし進行中runを中断しない。

## 再現性設計

- seed policy: `no_rng_fixed_saved_oof_postprocess`。乱数処理なし。
- parallel RNG: 該当なし。初回実装は`num_workers=1`固定とし、well順は文字列sortで固定する。
- PF/Beam/likelihood-PF/HMM/LightGBM/GPU: すべて0。
- runtime: Kaggle private CPU、internet off、目標runtime 30分以内、peak RSS 6 GB以内。
- SHA: public reference identity、親OOF file/content、親model manifest、raw horizontal ordered manifest、schema、config、pretruth candidate、pretruth diagnostic、metrics、package、kernelを記録する。
- deterministic anchor: false。補正式自体は固定入力に対しdeterministicだが、親exp287はGPU rerun bitwise parityを主張していないため、本実験単独で新anchorとは呼ばない。
- model SHA: 新規model 0を記録し、親model manifest SHA `419dbdf83dd6bc343f0265aca56dd690ba1f231ee419e7cf0ff456ffdb797590`を入力証拠として固定する。
- gzipを保存する場合はdecompressed content SHAを主証拠にする。

## 外部参照

- Kaggle Notebook: `phuongncn/kdrill-f594-ucont8`、id_no `128179543`、2026-07-22取得。
- 取得Notebook SHA256: `a98e718dc8cc3b8e94b74bf76b9d91d3e5af50b59b917d0fef31eb636f46f8ab`。
- 外部Notebookに記載された`selected on 495 wells / re-audited on 773 wells`は未検証の参考情報であり、promotion gateへ使用しない。

## リスク

- 実際の局所層序変化によるU jumpを誤って消す可能性がある。
- exp287 OOFのpseudo/current-like境界とhidden rerunの境界分布が一致せず、CV/LBが乖離する可能性がある。
- 773 wellのpooled改善が少数の大gap wellへ集中し、long-tailを悪化させる可能性がある。
- exp125、exp299など過去のcontinuity/handoff系はworst-well悪化を示しており、単純な「滑らかさ」を改善根拠にしてはならない。
- 公開Notebook全体のPublic scoreにはsame-well contact等が混在するため、公開scoreを本式の単独効果と解釈できない。

## 実装状態

- compact self-contained train候補へ、親OOF/raw suffix preflight、固定U-fade、pretruth SHA freeze、late-truth評価、全gate、生成物manifestを実装済み。
- compact self-contained inference候補はStage 0 pooled gain FAIL後のprediction/submissionをfail-closedで拒否する。
- synthetic contract tests 10件を追加した。
- canonical train Notebookへ採用し、Kaggle CPU version 2で完了した。canonical inference Notebookは未採用のまま、hidden-test predictionとsubmissionを生成していない。

## 次のアクション

pooled改善`0.001611295 ft < 0.020 ft`のため`FAIL_CLOSE_NO_RESCUE`で閉じる。continuity再訪は独立したtarget-free feature／selector仮説を事前設計できた場合だけ別途判断する。
