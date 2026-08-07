# exp257 nested selector output replacement-only on exp218 結果

## 結論

Kaggle GPU train v1は15 boosterまで正常完走したが、採用guardは不通過だった。同一outer foldのexp238 add-only OOFと比べて`lgb_mean`は`7.936690`から`8.101331`へ`+0.164641`悪化した。near、1000+、fold、worst-wellも基準を満たさないため、replacement-onlyモデルは採用せず、inference / submitには進まない。

## 実行契約

- kernel: `kentookumura/exp257-nested-selector-output-replacement-train` v1、id_no `127325011`
- runtime: Kaggle T4、internet off、ログ最大時刻約12,803秒
- rows / wells: 3,783,989 / 773
- selector source: exp238 selector train v4、保存済みouter 5 × inner 4の20モデル / 5 score面
- selector再学習: 0
- final LightGBM: 1 variant × 3 configs × 5 folds = 15 boosters
- parent/control再学習: 0

## Feature契約

- exp218 schema: 380列を同じ列名・順序で維持
- 既存`ll_*`: 54列
- nested selector出力で上書き: 29列
- selector入力診断として維持: 25列
- 新規`nsel_*`: 0列
- 最終feature: 380列

HMM、self-GR HMM、exp226を含む11候補のpredicted-error scoreから、既存29 slotのrank、probability、error、margin、spread、weighted TVTを作り直した。exp238の35個のadd-only特徴は追加していない。

## OOF結果

| model | pooled RMSE TVT |
| --- | ---: |
| lgb0 | 8.172207 |
| lgb1 | 8.142500 |
| lgb2 | 8.142779 |
| lgb_mean | **8.101331** |

比較の正は同一foldのexp238 `lgb_mean` `7.936690`で、差は`+0.164641`である。historical exp218 `8.475794`よりは名目`-0.374463`良いが、exp218はfold割当が異なるためreplacement-only効果の因果比較には使わない。

## Guard

| 指標 | exp257 - same-fold exp238 | 基準 | 判定 |
| --- | ---: | ---: | --- |
| global RMSE | +0.164641 | 0以下 | fail |
| near `000_050` RMSE | +0.068730 | 0以下 | fail |
| `1000_plus` RMSE | +0.184078 | 0以下 | fail |
| 改善outer fold | 1/5 | 3/5以上 | fail |
| worst-well最大回帰 | +13.291303 | +0.25以下 | fail |

`inference_allowed=false`は技術的な学習失敗ではなく、完走したCVが採用基準を満たさなかったことを示す。

## 解釈

exp238で追加した35個の`nsel_*`特徴をやめ、既存29 slotだけへ圧縮すると精度が落ちた。今回のadapterでは11候補の情報を既存5候補中心のschemaへ写像するため、HMM・exp226固有のidentityやtop1/top2 path値など、exp238 add-only側が持っていた情報の一部が失われる。また、既存slotの意味と分布も5候補selector時代から変わり、exp218の残り351特徴との組み合わせが悪化した可能性が高い。

## 信頼性と判断

380列schema、29列上書き、25列維持、`nsel_*` 0列、15 model、selector score 5 fold SHA、OOF / schema / replacement contract / model manifest / by-well / bucket / guard SHAはKaggle完了ログで確認した。train-side評価に必要な情報はlogsにあるため、Kaggle output archive全体は取得していない。

結果は実装失敗ではなく、今回のinverse-error adapterによるreplacement-only仮説の否定結果として扱う。guardを緩和せず、exp257のinference・submissionは実施しない。

## 2026-07-16 推論override

その後、ユーザーから推論実行の明示指示を受けた。上記の不採用判断と`guard pass=false`は変更せず、比較用artifactを得るためのoverrideとしてhidden-safe inferenceを実行する。保存済み20 selectorと15 final LightGBMを適用するだけで、推論中のselector / LightGBM学習は行わない。出力`submission.csv`は検証対象として生成するが、この指示ではcompetition submitは行わない。

Kaggle inferenceは`kentookumura/exp257-selector-output-replacement-inference`（id_no `127434953`）へT4・internet off・run-on-pushで開始した。v1のexp257 final manifestファイル名に誤りをpush後監査で検出し、正しい`..._model_manifest.json`へ直したv2でsupersedeした。

v2は463.95秒で`COMPLETE`。14,151行 / 3 well、context 184列、COPCF 41/41、missing / all-nonfinite context 0、test-test neighbor 0でcontext parityを通過した。保存済み20 selector / 15 final LightGBMだけをロードし、推論内学習は0。最終特徴は29列上書き、25列維持、`nsel_*` 0、380列で契約どおりだった。

生成した`submission.csv`はsampleとheader・14,151行・ID順が完全一致し、重複、missing、NaN、Infは0。`kaggle-submit-check`はFAIL 0 / WARN 0でPASSした。submission SHA256は`02e2a0311a99df52b25ae98f6a888f1c872d815b2c1015a8b4656578bac78c69`。ただしCV guard不通過と不採用判断は変更せず、competition submitは未実施である。

## 次アクション

提出形式とhidden-safe実装は検証済みだが、same-fold exp238よりCVが悪いため自動提出はしない。比較目的で提出するかは、CV悪化リスクを踏まえてユーザー判断とする。

## Public LB

ユーザー提出のcode submission ref `54753824`は`COMPLETE`、Public LB `7.718`だった。exp238 hidden-safe ref `54662073` / `7.775`を`-0.057`、COPCF parity ref `54725625` / `7.842`を`-0.124`改善し、Public-LB上のML route submitted anchorを更新した。ensemble route anchor exp082 / `7.601`よりは`+0.117`悪い。

一方、same-fold OOFはexp257 `8.101331`、exp238 `7.936690`でexp257が`+0.164641`悪い。CVとPublic LBの方向が反転しているため、LB anchor更新をtrain-side仮説の採用とはみなさない。replacement-onlyはhidden test構成には合った可能性があるが、単一Public LBだけでguardを撤回せず、CV/LB divergenceとして記録する。
