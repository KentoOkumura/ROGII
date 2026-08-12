# exp513_hjyact_v2_final_standalone_public_lb_audit

## 状態

- Route: `ensemble`
- 状態: Kaggle同一package 2回完走、visible final source parity PASS、hidden RNG決定性は未確定
- CV / Public LB / Private LB: `- / - / -`
- Submit ID: なし
- 作成日: 2026-08-05
- 親実験: `exp512_hjyact_v2_final_10pct_hedge_on_exp413`
- 公開source: hjyact version 2 / run `337064157` / 公開記載 Public LB `6.568`

正規train / inference Notebookはtemplate placeholderのままである。実行専用candidateだけをKaggleで2回実行し、
必要outputを取得した。正規Notebook採用とcompetition submissionは未承認・未実行である。

## 仮説

exp512で50%成分として定義した完全な`hjyact_v2_final`を単独でhidden-safe再生成して提出すれば、
公開sourceのPublic LB `6.568`を再現でき、exp413とのblend前に公開成分そのものの寄与と実装parityを
切り分けられる。

## 実装

- source Notebook SHAとexp512 v3 generator SHAを固定する専用generatorを作成した。
- source version 2のactive 37 code cellsを抽出し、診断/CV-only cellを除外した。
- SP45 / learned `0.60/0.40`、guarded overlap、balanced visible-prefix、model-package guard、
  PF seed-branch hedge、最終write順を保持した。
- exp413 runtime、50/50 blend、cross-consumer candidate reuse DAGを完全に除外した。
- learned trajectoryのprecomputed visible CSV探索とinference-time training fallbackは含めない。
- 最終`submission.csv`を変更せず、dynamic sampleとのschema / row / ID order / duplicate / finiteを確認する。
- dynamic sample ID-orderが既知visible sampleと一致した場合だけ、source final SHA
  `b192d3f3...b9ded4a`をpost-hoc assertionに使う。
- input/model/prediction/submission SHAとruntimeをKaggle output manifestへ保存する。

## exp512失敗への対策

exp512 v1はcompetition dataの旧固定mountから空well listとなり、`FormationPlaneKNN`で
`KeyError: wid`になった。exp513は旧/current候補のうち`train`、`test`、`sample_submission.csv`を
持つrootが一意であることを確認してから全source CFGへ渡す。

exp512 v2はRidge特徴量tableだけが旧dataset mountを直接参照し、読込前に停止した。exp513は
Ridgeの`data/train.csv`と5 trainer wrapperをSHA監査してrootを一意解決し、その同じrootを
`RIDGE_ARTIFACT_ROOT`と`CFG.artifacts_path`へ明示的に渡す。旧pathの直接assignmentが生成sourceに
残らないことを専用testで固定した。

exp512の初回SaveKernelは1 MiB source制限にも抵触した。exp513候補sourceはexp413と共有DAGを除いて
236,961 bytes、bootstrap済みpackageは437,126 bytesまで縮小し、上限をPASSした。

## 実行量

- scientific variant: 1
- LightGBM train config / new booster / parent-control retraining: `0 / 0 / 0`
- source Ridge: 1 config × 5 folds = 5 runtime fits
- hjyact saved model files / contained estimators: `13 / 33`
- exp413 saved models: 0（親から75 filesを除外）
- PF/Beam/likelihood-PF: 公開source finalの動的再生成のみ
- accelerator contract: GPU / internet off

## 検証方針

静的段階ではsource境界、mount resolver、model inventory、禁止経路、生成物SHAを契約testで確認する。
Kaggle段階ではvisible source exact parity、同一条件2 rerun、submit-checkの順にfail-closeで判定する。
Public LBは最後の単独code submissionでのみ確認し、結果後の設定変更は行わない。

## 検証

- candidate source: 5,097行 / 236,961 bytes / SHA `542f0947...9e9baf43`
- candidate Notebook: 48 cells / SHA `1e919d36...35826c5e`
- 専用契約test: `7 passed`
- exp512依存test込み: `13 passed`
- `py_compile` / Ruff F821 / Jupytext round-trip: PASS
- strict `validate-exp` / template validation: PASS
- 親compact比較: exp512 6,879行 / 8章に対し、exp513 5,097行 / 7章。
  削除した章はexp413 runtimeとshared-DAG/fixed-blend出力であり、公開finalの6章は保持した。

Kaggle version 1 / 2はいずれもCOMPLETEし、visible final SHAはsourceと一致した。ただしpre-override
PF/Ridge blend統計が2 runで異なるため、hidden-well deterministic anchorは未成立である。
submit-checkとLBは未評価である。

## 実行入口

- generator: `prepare_exp513_hjyact_v2_standalone_candidate.py`
- candidate source: `exp513_hjyact_v2_final_standalone_public_lb_audit_compact_selfcontained_inference.py`
- candidate Notebook: `exp513_hjyact_v2_final_standalone_public_lb_audit_compact_selfcontained_inference.ipynb`
- 正規inference Notebook: template placeholder、未採用
- train Notebook: 学習なし、template placeholder、実行禁止
- source/model契約: `standalone_contract.yaml` / `model_manifest.yaml`
- steering: `../../docs/legacy/steering/20260805-exp513-hjyact-v2-final-standalone-public-lb-audit/`

## 所見

exp512で失敗した2つのmount境界をsource実行前にfail-fast監査し、Ridgeだけ別rootへ戻る経路をなくした。
両Kaggle runでRidge 5 foldと全保存model推論まで完走し、visible source parityを再現した。
一方、source由来のunseeded Numba PFをthread並列生成する中間経路にはrun差があり、未知wellへそのまま
一般化した場合の決定性は主張しない。

## 次

source RNG semanticsを維持して提出へ進むか、well別明示seedを入れた別candidateを再検証するかを決める。
正規Notebook採用、submit-check、competition submissionは別承認とし、現状のcandidateはsubmitしない。
