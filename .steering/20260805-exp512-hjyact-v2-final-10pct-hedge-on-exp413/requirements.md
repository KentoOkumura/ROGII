# 要件

## 依頼

hjyact公開Notebook `Ultimate PF-Config Strategy | A Reproducible Score` version 2の最終予測をhidden sample上で
動的再生成し、`exp413_scale5_likpf_full_replacement_on_exp335`と`0.50 / 0.50`でアンサンブルする。

実験ディレクトリ名の`10pct_hedge`は初期設計の履歴名として維持するが、2026-08-05のユーザー変更指示を
最終仕様とし、予測式・設定・テスト・文書はすべて等率blendへ固定する。

## 制約

- Route: `ensemble`。
- 親: `exp413_scale5_likpf_full_replacement_on_exp335`。比較対象: `exp510_exp413_exact_public_preoverride_hedge`。
- sourceはhjyact version 2 / run `337064157` / profile `vp_balanced_modelpkg_005`へ固定する。
- sourceのSP45、learned、guarded overlap、balanced visible-prefix、PF seed-branch hedgeを
  最終writeまで順序どおり実行する。model-package correctionは2026-08-05のユーザー指示により無効化し、
  入力dataset・model load・推論を行わない。
- SP45のtest-well loop、exp413のHMM/PF/K16 test-well loopは各4並列とする。well入力順、well内row順、
  per-well seed、数値dtypeを維持し、親processで入力順に再結合する。
- 最終式はfloat64で`0.50 * exp413 + 0.50 * hjyact_v2_final`。weight fit/grid、well router、LB後変更は禁止する。
- visible output CSVやstatic exp413 prediction sidecarをruntime入力に使わない。dynamic sample ID上で両成分を生成する。
- fingerprintが完全一致する決定論的候補だけを共有する。likelihood-PF、route-specific PF、HMM/K16、
  final overlayは共有しない。
- 特定well ID、visible row/well数、Public LB値を予測分岐に使わない。既知SHAはdynamic sample identity確定後の
  post-hoc assertionに限る。
- 新規boosterと親/control再学習は0。source Ridgeだけを1 config × 5 foldsで実行する。
- source/input/model/component/final SHA、seed、image、kernel version、shared-node generation/hit countを記録する。
- exp413 visible parityは、exact reference content SHA、またはv3--v5で再現して参照差を監査済みの
  witness content SHAだけを受け入れる。後者はmax absolute `0.02 ft`以下かつRMSE `0.001 ft`以下とする。
- 2026-08-05のユーザー指示により、全wellを対象とするKaggle package/push/runを実行する。
  正規Notebook採用、output archiveの全量取得、competition submitは別承認を必要とする。

## 受け入れ基準

## 2026-08-05 v6構成のlatest-version再実行

ユーザー指示により、速度最適化v7ではなく、Kaggle version 6で実行済みの科学packageを同じcanonical
kernelへ再pushし、最新kernel versionとしてもう一度実行する。このrunに限り、上記の4並列化・model-package
無効化というv7 runtime overrideを適用しない。

- canonical kernelは`kentookumura/exp512-hjyact-v2-equal-blend-inference`のまま変更しない。
- sourceはv7 push直前に同kernelからpullして保存したv6 Notebookを使う。historical `/6` API pullは403だったが、
  pre-v7 pullのcode-cell SHAは`9b5a4cae3d1943472316f8b6a15640c2e0380f4b3f83de693ade8756fbdabee6`、
  embedded candidate SHAは`66ed4f78e0c3525ab7f7f52d99f2b2a1cb100e36223c6465bd49c2b3c18f804c`で
  当時のv6記録と一致する。
- v6 metadataどおり`pilkwang/rogii-model-package`を含む7 dataset sourcesを使い、model-package correctionを
  有効にする。SP45とexp413 HMM/PF/K16はv6の逐次実装を使う。
- scientific variant 1、LightGBM train config 0、新規booster 0、親/control再学習0、runtime Ridge 1 config ×
  5 folds、保存model 88ファイル / 108推定器とする。
- 同一canonical idへ追加される最新versionのhjyact SHA、exp413 witness SHA、reuse manifest、fixed formula、
  submission SHAをv6実績と照合する。期待submission SHAは`b960c2b16f01e4224850a5c644a04b792a471b3a0def08018a8d184fea713e23`。
- competition submit、weight/profile変更、正規Notebook採用は行わない。

### 実装・静的検証

- Jupytext percent形式の別名compact self-contained inference候補が存在し、正規Notebookを上書きしていない。
- source pull Notebook SHA、code-cell SHA、active cell順、13 hjyact保存modelファイルと75 exp413保存modelを固定する。
- shared DAGがgeneration=1/cache hit=1をfail-closeで監査し、exp413固有PFを別生成する。
- dynamic ID one-to-one、finite、duplicate 0、CSV boundary、固定0.50/0.50式、external submit falseを実装する。
- `py_compile`、Ruff F821、Jupytext `--test`、専用pytest、`validate-exp`を通す。
- 4並列設定とmodel-package無効化を専用pytestで固定し、Kaggle実行logに実効well数・並列数・runtimeを残す。

### 将来のKaggle実行

- visible sampleではhjyact final SHA
  `b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a`と一致する。
- exp413 predictionはexact reference decompressed content SHA
  `875a1334ae3c90f841414f8f98d8877fb06234e17e0fd0b8d46385170a584dc4`、または監査済みwitness SHA
  `3a9bbd1f7e6ab93189c90b4c9c0da9d6a2858746028e93b25fe2a10c7be68d87`と一致する。
- witnessを使う場合、reference差はmax absolute `0.02 ft`以下かつRMSE `0.001 ft`以下である。
- shared nodeがwellごとに1回生成され、exp413 consumer hitが1で、fallback 0となる。
- final formula max absolute errorが`1e-9 ft`以下となる。
- 同一Kaggle条件2回でcomponent/final/submission SHAが一致する。
- submit-checkがPASSし、competition submitは別承認まで行わない。
