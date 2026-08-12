# exp257_nested_selector_output_replacement_only_on_exp218 セッションノート

## 2026-07-15 実装

### ユーザー訂正

exp255で実装したgated/bounded direct readoutは依頼意図と異なっていた。求められていたのは、
HMM・self-GR HMM・exp226をselector候補へ追加した後も、selector出力をLightGBMの新規特徴として
add-onlyせず、既存selector特徴を置き換える構成だった。

exp255は既に異なる仮説としてKaggle実行・評価済みなので履歴を改変しない。仮説とモデル構造が
変わる場合は新expを切るリポジトリルールに従い、正しいreplacement-only構成をexp257とした。

### 既存特徴の分類

exp218 feature schema 380列を監査し、`ll_*` learned-likelihood blockが54列あることを確認した。

- 29列: rank、predicted error、probability、margin、entropy、candidate spread、legacy candidate別score、weighted TVT。新nested selector出力で上書きする。
- 25列: legacy 5候補のmulti-observation 15列とanchor/likPF差10列。selector入力診断なのでexp218値を維持する。

29 + 25 = 54をコードでfail-fastし、380列の列名・順序を変えない。exp238の35 `nsel_*`は
1列も作らない。新候補専用one-hotも作らず、HMM/exp226は11候補全体のtop index、rank、
margin、spread、weighted TVTを通じて既存slotへ反映される。

### Nested stacking契約

- selector source: exp238 selector train v4、outer 5 × inner 4の保存済み20 models / 5 score artifacts。
- selector再学習: 0。
- outer-train: inner OOF predicted-error score。
- outer-valid: outer-trainだけで学習したinner 4 modelsの平均score。
- outer-valid true TVT / true candidate errorを特徴生成へ使わない。
- selector score 5 artifactはexp238 v3と一致したdecompressed SHAをconfigへ固定する。
- final LightGBMはexp238 score artifactのroleを正のouter fold contractとして使う。

### Selector output adapter

predicted absolute error scoreを`max(score, 1e-3)`でclipし、逆数を行内正規化して既存
probability slotへ入れる。error slotには元scoreを入れる。全11候補でrank、entropy、spread、
weighted TVTを作り、legacy candidate別の既存slotは共通5候補だけを更新する。

probabilityとerrorのweighted TVTは同じnormalized inverse-error重みになるため同値だが、
既存380 schemaを維持するため両方の既存slotへ同じ値を上書きする。

### Kaggle train実行前確認

- active variant: 1 (`nested_selector_output_replacement_only`)
- selector configs / folds / boosters: 0 / 0 / 0（保存済みscoreを使用）
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- outer folds: 5
- planned final boosters: 15
- total new boosters: 15
- parent/control retraining: なし
- feature schema: exp218と同じ380列
- selector output replaced: 29列
- selector input diagnostic preserved: 25列
- selector added / `nsel_*`: 0列
- GPU: Kaggle T4、internet disabled

ユーザーは実装を依頼しているが、この新しい15-booster GPU trainのpushはまだ明示承認していない。
実装・静的検証まで進め、push前に承認を確認する。

### Inference

CV未実行のためfail-closed notebookだけを置く。OOF、fold、worst-well、near、1000+、
hidden-like、380 schemaを監査して採用判断後に、exp238 copcf parity current-test generatorと
保存済み20 selectorを使うhidden-safe inferenceを同じexp257内へ実装する。

### 静的検証・Kaggle package

- 実exp218 train v1 feature schemaを使ったcontract test: 380列、selector出力上書き29列、入力診断維持25列、`nsel_*` 0列を確認。
- 合成7行 × 11候補でselector output adapterを実行し、29列、全finite、probability row sum誤差`5.96e-08`以下を確認。
- `py_compile`: pass。
- `ruff --select F821,F401`: pass。
- Jupytext train/inference `--test`: pass。
- `validate_experiment.py`: strict pass。
- CV後監査として同一outer foldのexp238 OOFを必須入力にし、global、near `000_050`、`1000_plus`、fold、worst-wellを保存する。
- guardはglobal/near/1000+非悪化、3/5 folds改善、worst-well最大回帰0.25以下をすべて要求する。
- train package: `kaggle/train`。kernel `kentookumura/exp257-nested-selector-output-replacement-train`、T4、internet off、`run_on_push=false`。
- kernel sources: exp238 final OOF、exp238 selector、exp218/exp237および既存upstreamを含む13 sources。bootstrap内config/source SHA manifestを生成済み。
- package prepareのみ実施し、Kaggle pushは行っていない。
- SHA: config `b4bcbfb8a5bb574b06646594c74228ac3967f33a4bb4c570a394c0c1ba8f0a20`、replacement engine `aa334fb7d346e8e1d2cae7ad209e14d992866a8c2a54e57ff4977c968ed5d51f`、Jupytext train source `ca2f629fb7929a9a5e39fbe28e82e110de85b87975ec317ccb44818794cbd862`、kernel metadata `208a34260c4749960cf61cdb4c032e5a0a2461675c7f78815ca1b24cd2287f07`、packaged notebook `f88bc024719c7387a8bda82c79fe159aa7fa543001ad45059a47c6d65845c4cb`。

### Kaggle GPU train実行承認

- 2026-07-15、ユーザーが「学習を実行してください」と明示承認した。
- 実行対象: active variant 1 (`nested_selector_output_replacement_only`)。
- LightGBM configs: 3、outer folds: 5、新規final boosters: 15。
- selector configs / folds / boosters: 0 / 0 / 0。exp238保存済みnested scoreを読むだけでselectorは再学習しない。
- parent/control再学習: 0。exp218 historical OOFと同一fold exp238 OOFを参照baselineとして読む。
- feature contract: exp218と同じ380列、既存selector出力29列上書き、入力診断25列維持、`nsel_*`追加0。
- Kaggle T4、internet disabled、canonical kernel `kentookumura/exp257-nested-selector-output-replacement-train`へpushする。

### Kaggle GPU train v1 push

- Kernel: `kentookumura/exp257-nested-selector-output-replacement-train` v1、id_no `127325011`。
- URL: `https://www.kaggle.com/code/kentookumura/exp257-nested-selector-output-replacement-train`。
- `run_on_push=true`、`NvidiaTeslaT4`、internet disabledでpush成功。push直後statusは`RUNNING`。
- 実行対象は1 replacement-only variant、3 LightGBM configs、5 folds、合計15 boosters。selector/parent/control再学習は0。
- Kaggle側pull metadataでcompetition source 1、kernel sources 13、T4、internet offを確認した。
- pull後notebookはlocal packageと16 cellsすべてのsourceが一致。source結合SHAは`487670f04ee17bee72f8a74ec28b3d7eb7972186c1fb2cd210984ff1ef89fbdd`。
- push時config SHA `b6e40aa2cd2cebabedc20af16ea254383c73f143252b26bf1b97d1b8867312ac`、replacement engine SHA `aa334fb7d346e8e1d2cae7ad209e14d992866a8c2a54e57ff4977c968ed5d51f`、packaged notebook SHA `d2c49a11a0006b035907d60f7af23e5aea981d39944d7463d1145992f2d73756`。
- 実行中CLI logsの継続取得は行わない。ユーザーの完了/失敗連絡後に同じkernel v1の通常logsを取得して監査する。

## 2026-07-16 Kaggle GPU train v1完了・監査

- ユーザーの完了連絡後に、継続監視ではなく同じcanonical kernel v1の通常logsを1回取得した。
- Kernelは`COMPLETE`。1 replacement-only variant、3 LightGBM configs、5 folds、15 boostersを完走し、selector / parent / controlの再学習は0だった。
- rows / wells / candidatesは3,783,989 / 773 / 11。
- feature契約はexp218 schema 380列、既存selector出力29列上書き、入力診断25列維持、追加`nsel_*` 0列でPASSした。
- pooled RMSE TVTは`lgb0` 8.172206879、`lgb1` 8.142499924、`lgb2` 8.142779350、`lgb_mean` 8.101330757。
- 因果比較の正は同一fold exp238 add-only `lgb_mean` 7.936689854。global差は+0.164640903で悪化した。historical exp218 8.475793752との差-0.374462995はfoldが異なるため参考値に限る。
- guardはglobal +0.164640903、near `000_050` +0.068730295、`1000_plus` +0.184078217、改善fold 1/5、worst-well最大回帰+13.291302681で全条件を満たさず`pass=false`。
- `inference_allowed=false`。これはnotebookや学習の技術的失敗ではなく、完走したreplacement-onlyモデルのCV不採用判定である。
- selector sourceはexp238 selector train v4の保存済み20 models。5 score面のdecompressed SHA契約はすべて一致し、exp257内selector refitは0。
- artifact SHA: metrics `c13960751976f40465835a6ba14215ddcb243133fb97eff92cb6f3742c8989a6`、OOF decompressed `6d9c1aa4f4d6de15a405922b48287854f8d9e4d24b826fcb309b332f96ca104c`、feature schema `2ec950058f6c8cb169cd07bd19661c937abc9c78239db2a2af1e2b383b737097`、replacement contract `9f79c7d5b20d5cf5c40074edd21be740be3ba3d952dd0e5f8ff37810f2a615de`、model manifest `92552370dd5c2decb2a109ae43d96e5bb26a03ef1932ebd125d3bc495f488e6b`。
- by-well / bucket / guard SHAもsummaryに記録され、採用判断に必要なtrain-side情報はlogsで確認できたため、リポジトリ運用ルールどおりKaggle output archive全体は取得していない。
- 結論: replacement-only仮説は不採用。guardを緩和せず、hidden-safe inference実装・Kaggle inference・competition submitは行わない。

## 2026-07-16 ユーザー指示による推論

- ユーザーが「推論に進んでください」と明示したため、CV採用guardとは分離したoverrideとしてKaggle inferenceを実行する。
- guardは`pass=false`のままであり、exp257を採用・昇格したとは扱わない。同一fold exp238比はglobal `+0.164641`、near `+0.068730`、1000+ `+0.184078`、改善fold `1/5`、worst-well最大回帰`+13.291303`である。
- 推論ではexp238 selector train v4の保存済みouter 5 × inner 4 = 20 selectorと、exp257 train v1の保存済み3 configs × 5 folds = 15 final LightGBMだけを読む。selector / final LightGBMの学習はともに0。
- current testごとにexp238 COPCF parityのhidden-safe contextを生成し、outer foldごとのselector scoreから既存29 `ll_*`出力slotを上書きする。25 selector入力診断を維持し、`nsel_*`は0、最終schemaは380列のままとする。
- test-test neighborとpublic test行artifactへの依存を禁止する。visible testとtrainのwell重複はhidden固有の入力契約として利用せず、hidden testのwell数に依存しない。
- notebookは`submission.csv`を生成するが、competition submit APIは呼ばない。推論完了後に`kaggle-submit-check`で出力を検証し、提出は別の明示指示を待つ。

### Kaggle inference v1 push

- 最初のslugは51文字でKaggle APIのkernel登録に失敗した。短縮したtitle/slugを一致させ、`kentookumura/exp257-selector-output-replacement-inference` v1、id_no `127434953`としてpush成功した。
- URL: `https://www.kaggle.com/code/kentookumura/exp257-selector-output-replacement-inference`。push直後statusは`RUNNING`。
- Kaggle pull metadataでT4、internet off、competition source 1、kernel sources 12を確認した。入力にはexp257 final train、exp238 saved selector、exp072/218 feature sourcesとcurrent-test COPCF生成に必要なtrain-only referenceを含む。
- pulled notebookは22 cellsすべてlocal packageとsource一致。selector / final modelの`fit`およびcompetition submit APIはnotebookに存在しない。
- push package SHA: config `a56cc73ef4893d1d439c46c8be615a91d2cb7d96ebcc21cd8b78cc558778753a`、replacement engine `aa334fb7d346e8e1d2cae7ad209e14d992866a8c2a54e57ff4977c968ed5d51f`、Jupytext inference source `ce37e45224e3c2f18ebd51e0f53665a47847481091f2149bc22f920c74f4024b`、kernel metadata `57052c32f9266fec381e23458de5075faadfae4a532d93005a4ffa53f5d91ca6`、packaged notebook `e0bcc69844db8853ced95678d827fbd0c0b5c721ccf7a2bfb3419047e1c7cf29`。
- 継続監視は行わない。ユーザーの完了/失敗連絡後に通常logs/outputを取得し、`submission.csv`を提出前検証する。

### Kaggle inference v2 manifest修正

- v1 push後のtrain artifact契約突合で、exp257 final manifestの実名が`..._model_manifest.json`であるのに、v1がexp238由来の`..._final_model_manifest.json`を探していることを検出した。実行時停止になるため、正しいexp257名へ修正した。
- 同じcanonical kernelへv2をpushし、v1をsupersedeした。v2 push直後statusは`RUNNING`。
- Kaggle pull後のv2はlocal packageと22 cellsのsourceが一致し、正しい`{OUTPUT_PREFIX}_model_manifest.json`だけを含み、誤った`_final_model_manifest.json`参照は0。
- v2 package SHA: config `f5d03c641fbbe9599e2ec27f4731d1c45a76b31e323f811014ede913db12b261`、Jupytext inference source `d0cbb8a2d5f348cbfe70b5cb672aabb4c072e68cab92773a99017508fd5c3fea`、kernel metadata `57052c32f9266fec381e23458de5075faadfae4a532d93005a4ffa53f5d91ca6`、packaged notebook `caad5a735e034554743c07fc7d9203bf9316ec70490119debf1276af53d9f9cb`。

### Kaggle inference v2完了・提出前検証

- ユーザーの完了連絡後、canonical kernel v2の通常logsとoutputを取得した。最終statusは`COMPLETE`、notebook summary runtimeは`463.951435`秒。
- 14,151 rows / 3 wells、context 184列、COPCF 41列すべてに有限値があり、missing context 0、all-nonfinite context 0、required diagnostic failure 0、context parityはPASS。
- 保存済みselector 20 modelsとfinal LightGBM 15 modelsをロードした。selector / final LightGBMの推論内学習は0、test-test neighbor利用0、competition submit実行0。
- 最終特徴契約は380列、selector出力29列上書き、selector入力診断25列維持、`nsel_*` 0列。outer 0..4の各replacement auditが14,151行 × 11候補 × 29列をカバーした。
- `kaggle-submit-check`: FAIL 0 / WARN 0 / PASS。`submission.csv`は`id,tvt` 14,151行、sampleとheader・行数・ID内容/順序が完全一致し、重複ID 0、missing / NaN / Inf 0。prediction artifactとも数値一致。
- submission SHA256は`02e2a0311a99df52b25ae98f6a888f1c872d815b2c1015a8b4656578bac78c69`、prediction decompressed SHA256は`2283742221ca3ec20ff5c8c68e426752feda8946f2a8fe6767c4cd86a1928f71`で、summary内SHAと一致。
- 品質guardは引き続き`pass=false`、exp257は不採用のまま。生成物は形式上提出可能だが、competition submitは未実施。次アクションは、CV悪化を受け入れて比較提出するかをユーザーが判断すること。

### Code submission scoring完了

- ユーザーのスコアリング完了連絡後、Kaggle submission historyを取得した。最新ref `54753824`、submitted `2026-07-16 06:21:04.920000`、status `COMPLETE`、Public LB `7.718`、private score未公開。
- 対象は`kentookumura/exp257-selector-output-replacement-inference` v2のcode submissionとして記録する。submission descriptionは空だが、ユーザーの直前exp257提出コンテキストと最新時刻によりattributionした。
- exp238 hidden-safe ref `54662073` / 7.775から`-0.057`、exp238 COPCF parity ref `54725625` / 7.842から`-0.124`改善。ensemble route anchor exp082 / 7.601よりは`+0.117`悪い。
- exp257 OOF 8.101331は同一fold exp238 7.936690より`+0.164641`悪化しており、CVとPublic LBの方向が反転した。したがってPublic-LB上のML route submitted anchorはexp257へ更新するが、train-side guard falseと不採用判断は維持し、CV/LB divergenceの証拠として扱う。
- submission SHAはvisible outputの`02e2a0311a99df52b25ae98f6a888f1c872d815b2c1015a8b4656578bac78c69`、submit-check PASS。
