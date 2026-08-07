# exp512_hjyact_v2_final_10pct_hedge_on_exp413

## 状態

- Route: `ensemble`
- 実装: compact self-contained推論候補を作成済み
- 静的検証: 構文・Ruff F821・専用契約テスト・Jupytext・`validate-exp`の対象
- Kaggle実行: exact v6 source再実行のversion 8 COMPLETE、canonical kernelのlatest、current-test submit-check PASS
- CV: honest OOFなし
- code submission: ユーザー実施ref `55255459`がCOMPLETE、Public LB `6.541`、Private LB未表示
- Public LB位置づけ: exp413 / exp510の`7.201`を`0.660`改善した新しい全体・アンサンブル基準
- 正規Notebook: placeholderのまま（候補を別名で保持）

ディレクトリ名の`10pct_hedge`は初期設計時の履歴名である。ユーザーの2026-08-05の変更指示により、
実装済みの最終式は次の等率blendへ固定した。

```text
final = 0.50 * exp413 + 0.50 * hjyact_v2_final
```

## 仮説

動的に再現したexp413とhjyact-v2 finalの誤差が補完的なら、ユーザー指定の等率blendは両単体とは異なる
public/private挙動を示す。ただしhonest OOFがないため、実装完了だけでは性能改善を主張しない。

## 実装

- hjyact kernel version 2 / run `337064157`のactive final pathを、pullしたNotebook SHAとcode-cell SHAで固定した。
- sourceのSP45、learned trajectory、guarded overlap、balanced visible-prefix、
  PF seed-branch hedgeを順序どおり動的に再生成する。visible output CSVは入力に使わない。
- local候補version 7では速度優先指示によりmodel-package correctionを無効化し、SP45とexp413
  HMM/PF/K16を最大4並列化した。この変更は速度実験の履歴として保持する。
- canonical kernelのlatestであるversion 8は、pre-v7 pullに保存されていたexact v6 sourceをそのまま再実行した。
  したがってSP45/exp413はv6の逐次実装で、model-package datasetと5 modelsを含む。
- exp413はexp510で作成したhidden-safe runtimeを埋め込み、75保存modelから動的に再生成する。
- hjyact learned側で作った7-beam、NCC、formation/dense、決定論的GR幾何特徴をprocess-localでexp413へ渡す。
  PF/likelihood-PFはseedとsigma設定が異なるため経路別に再生成する。
- 両成分を一度CSV境界で読み戻し、dynamic `sample_submission.csv`へone-to-one整列してfloat64でblendする。
- visible sampleとID順SHAが一致した場合だけ既知component SHAをpost-hoc assertionとして使う。exp413はexact SHA、
  またはユーザー承認のmax `0.02 ft` / RMSE `0.001 ft`以内と監査済みのwitness SHAだけを許可する。

## 実行量

- scientific variant: 1
- LightGBM train config / 新規booster / 親control再学習: `0 / 0 / 0`
- source Ridge: 1 config × 5 folds = 5 runtime fits
- latest version 8保存model: exp413 75 + hjyact 13 = 88ファイル
- latest version 8のwrapper内部を含む推定器: exp413 75 + hjyact 33 = 108

詳細は`model_manifest.yaml`と`ensemble_contract.yaml`を参照する。

## 検証方針

- 静的段階ではsource/model SHA、Jupytext、構文、F821、専用契約、experiment schemaを検証する。
- Kaggle version 6でhjyact exact SHA、exp413 numerical witness、shared-node generation/hit、固定式を照合してPASSした。
- Kaggle version 7は同じcomponent/final SHAを維持しつつ`1,197.667秒`で完走し、version 6比`23.771%`短縮した。
- exp413は`38.529%`短縮した一方、SP45 threadingはvisible 3 wellで`38.712%`遅く、次の最適化対象として残る。
- Kaggle version 8はexact v6 sourceで`1,193.477秒`（total log `1,205.751秒`）で完走した。hjyact、exp413、
  submissionはversion 6とbyte-identicalで、same-v6 visible output gateは2/2 PASS。
- latest v6 contractの200 well工程別外挿は`約14–18時間`で、visible 3 well runtimeだけを9時間制限へ直接比較せず
  planning gateをFAILのまま保持する。後のcode submission COMPLETEは実際に提出されたhidden rerunのplatform完走証拠だが、
  APIはscript version、hidden well数、正確な実行時間を返さないためversion 8の9時間保証には使わない。
- v6/v8ではdeterministic-GR intermediate content SHAが異なるため、visible最終出力2/2 PASSと全中間・hidden RNG再現を区別する。
- dynamic sample契約を使い、known visible cardinalityは予測分岐へ使わない。

## 実行入口

- 候補source: `exp512_hjyact_v2_final_10pct_hedge_on_exp413_compact_selfcontained_inference.py`
- 候補Notebook: `exp512_hjyact_v2_final_10pct_hedge_on_exp413_compact_selfcontained_inference.ipynb`
- generator: `../../scripts/prepare_exp512_hjyact_v2_candidate.py`
- 正規inference Notebook: placeholder、未採用

Kaggle package/push/runは承認済みで、exact v6 sourceのversion 8を全well対象で完走しlatest versionとした。正規Notebook採用と
Codexによるcompetition submitは未承認で行っていない。後にユーザーがexp512として提出したref `55255459`は
`2026-08-05 02:08:11.450000 UTC`に受理され、Kaggle CLIでCOMPLETE / Public LB `6.541`を確認した。

## 実行時生成物

- `submission.csv`
- `hjyact_v2_final_submission.csv`
- `exp413_component_submission.csv`
- `exp512_component_readout.csv`
- `candidate_reuse_manifest.json`
- `exp512_model_manifest.json`
- `exp512_reproducibility_manifest.json`
- `metrics.json`

## 所見

version 8の0.50/0.50 outputはsource parity、numerical witness、reuse manifest、current-test submit-checkをPASSし、
hjyact/exp413 componentとsubmissionはversion 6とbyte-identicalだった。model-packageは実行されたがp95差
`26.700659 ft > 25 ft`のguardにより最終weight 0となった。ユーザー提出ref `55255459`のPublic LBは`6.541`で、exp413 / exp510
`7.201`から`-0.660`、source公開値`6.568`から`-0.027`。両component scoreが同じPublic rowsでexactに再現された
という条件では、三角上限`6.8845`も`0.3435`下回り、等率blendで誤差が一部相殺された結果になる。
ただしhonest OOF、Private LB、全中間生成物のbyte再現、hidden-well stochastic determinism、private一般化は未証明のままである。
