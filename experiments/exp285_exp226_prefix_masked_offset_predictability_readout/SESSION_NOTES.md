# exp285_exp226_prefix_masked_offset_predictability_readout セッションノート

## 目的

known prefix内の固定masked backtestから得るexp226 geometry-only offset summaryが、official suffixの
exp226 residualを予測できるかを、補正前に0-boosterで監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle version 2完了、scientific guard FAIL、branch closed
- CV / LB: 対象外 / なし
- active readout variants: 1
- LightGBM config / trained fold / booster: 0 / 0 / 0
- HMM / PF regeneration: 0 / 0
- parent/control retraining: なし
- implementation approval: 2026-07-19ユーザー依頼「実装してください」
- Kaggle push/run approval: 2026-07-19ユーザー依頼「実行してください」

## コマンドログ

- `task new-steering ...`: `task`未導入のため実行不可。
- `make new-steering EXP=exp285_exp226_prefix_masked_offset_predictability_readout`: steering作成。
- `make new-exp EXP=exp285_exp226_prefix_masked_offset_predictability_readout`: template scaffold作成。
- raw trainの`TVT_input` finite lengthだけをread-only集計し、mask 640 + visible 512条件で
  766 / 773 wellsがeligible見込みであることを確認した。予測生成やtruth readoutは行っていない。
- `make update-summary`: `experiment_summary.md`へexp285のnode / lineage / status rowを追加。
- `make validate-exp EXP=exp285_exp226_prefix_masked_offset_predictability_readout`: strict PASS。
- `review_exp_docs.py exp285 --root .`: core evidence categoriesが揃っていることを確認。
- 別名compact self-contained train/inference Jupytext sourceを実装した。
- Jupytextでcompact train/inference `.ipynb`を生成し、round-trip testを通した。
- 専用合成test 7件でmask、well-end replay、summary、freeze、official truth、permutation、guard、
  fail-closed inferenceを確認した。
- 初回実装時にexp284/285関連test 15件、全repository test 218件を実行し全PASSした。
  version 1修正test追加後の最終状態は専用8件、全repository 219件PASS。
- strict `make validate-exp`を実装後configでも再実行しPASSした。
- Kaggle実行前契約を1 variant / 0 LightGBM config / 0 trained fold / 0 booster、CPU、
  parent/control retrainingなしとして再確認した。
- `kaggle-platform` credential checkerでKaggle CLI OAuthとlegacy credentialが利用可能であることを確認した。
- compact self-contained trainを正規train notebookへ採用した。canonical notebook SHAは
  `41c4d614...eac95`。
- `make prepare-kaggle-notebooks ... --notebook train --kernel-id
  kentookumura/exp285-prefix-masked-offset-readout-train --title 'exp285 prefix masked offset readout train'
  --run-on-push --strict`でKaggle packageを生成した。
- package metadataはprivate / CPU / internet off / run-on-push、competition source 1件、
  kernel source 2件。packaged notebook SHAは`5d53c45a...cd334`、metadata SHAは`8f07a61a...c928b`。
- push前の同一kernel pullは403 Forbiddenで、既存kernelを確認できなかった。初回作成として同じ
  canonical IDのままpushする。
- Kaggle kernel `kentookumura/exp285-prefix-masked-offset-readout-train` version 1をpushし、
  kernel id_no `127855223`を確認した。
- version 1はfold 0 donor field構築後、raw horizontal CSVに存在しない`id`列をtarget-safe loaderが
  `usecols`指定したため停止した。geometry/maskは`id`値を科学入力に使わず、exp284と同様に
  `<well>:<row_idx>`をloader内で生成する最小修正とする。mask、summary、guardは変更しない。
- loader修正を専用testへ追加し8/8 PASS、compact/正規notebookを再生成し、同一kernel IDへversion 2をpushした。
- version 2は766 eligible / 7 ineligible wells、5 foldsを77.492秒で完走した。technical guardは全PASS。
- primary offset-median pooled Spearman `-0.004135`、fold `0.055222 / 0.036900 / -0.091083 /
  -0.033034 / -0.026692`、positive fold 2/5、`rho>=0.20` fold 0/5、balanced sign accuracy
  `0.488567`、256 permutation p `0.599222`でprimary guardは全FAILした。
- supporting slope / drift Spearmanは`-0.009074 / -0.013928`。scopeはH256/H512/H640
  `0.186915 / 0.153020 / 0.131063`、near `0.189776`、1000+ `-0.006022`、hidden-like
  spatial/typewell `0.118237 / 0.119217`。scope/supporting guardもFAILした。
- outputを`/tmp/kaggle-output/exp285_exp226_prefix_masked_offset_predictability_readout/train_v2`へ取得し、
  summaryとmetricsのSHAがKaggle log記録と一致することを確認した。

## 固定scientific contract

- mask 640 rows、visible minimum 512 rows、one cut / well。
- exp226 geometry-only、fold外donor field、saved fold kappa、validation well全体除外。
- pseudo replay horizonは640で切らず、pseudo cutからwell末尾まで。
- pseudo summaryは5 x 128 blockのmedian、slope、drift-rate。
- pseudo path -> masked TVT_input -> prefix summary -> official truthのfreeze順序を固定。
- primary Spearman `>=0.30`、5/5 fold positive、4/5 fold `>=0.20`、balanced sign accuracy
  `>=0.60`、256 permutation p `<=0.01`。
- slope/driftのsupportとnear / 1000+ / hidden-like正相関もpromotion guardに含める。
- cut/mask/block/summary/clip/threshold/guard gridは禁止。

## 変更点

- exp226のfull predictionではなくgeometry-only `tvt_geop`を対象に固定した。
- official cutのOOF再利用だけでなく、known prefixを短縮したfold-safe pseudo replayを追加する設計にした。
- 補正値を生成せず、prefix summaryとofficial target summaryのpredictability readoutだけに限定した。

## 実装状態

- `config.yaml`、README、SESSION_NOTES、result、metrics: 設計確定へ更新。
- steering requirements / design / tasklist: 設計確定へ更新。
- 正規train `.ipynb`: 実行承認後にcompact self-contained版を採用済み。
- 正規inference `.ipynb`: template stubのまま。
- compact train: 1,902行 / 9章 / 20 cells。source SHA `25d6a8d4...4c49c8`、
  notebook SHA `1b037365...616cc`。
- compact inference: 135行 / 4章 / 10 cells。source SHA `eff6e31b...7b66d`、
  notebook SHA `0e9de3a2...999d4`。
- same-exp helper import: なし。`__file__`参照: なし。
- 専用test: 8/8 PASS。
- repository test: 219/219 PASS。
- `py_compile`、ruff、Jupytext round-trip、strict experiment validation: PASS。
- Kaggle version 1: input-schema failure。version 2: completed / scientific guard FAIL。
- local notebook run、current-test生成、inference、submission: 未実行。

## 再現性メモ

- real replay / summary / readout: RNGなし。
- stochastic component: 256回fold内permutation negative controlのみ。
- seed policy: experiment / fold / permutation indexからstable SHA256 local RNG。
- process: Kaggle CPU / single process / fold -> sorted well順、GPU/TPU/internet off。
- exp226 OOF decompressed SHA、fold kappa SHA、hidden-like SHAをhard guardする設計。
- pseudo path / prefix summary / official target summaryのschema/content SHAを段階別に保存する。
- model / prediction / submissionは生成しないため各SHAは対象外。
- fixed-input diagnosticでありdeterministic prediction anchorとは呼ばない。

## 次のアクション

固定契約どおりparameter rescueを行わずbranchを閉じる。prefix-calibrated correction、exp281のblend/
selector救済、current-test生成、inference、submissionへ進まない。後続exp284も固定horizon self-GR
incremental recovery / safety guardをFAILしたため、短距離の弱い正相関を救済根拠にせず、新規backlogも
追加しない。
