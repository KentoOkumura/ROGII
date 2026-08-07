# exp509_exp413_strict_public_core_final_slot セッションノート

## 目的

最終提出2枠の第1枠として、exp413へexp497 strict public-coreを固定13.716%だけ混ぜる
private一般化優先のreference candidateを設計する。

## 現在の状態

- Route: `ensemble`
- 状態: candidate implementation complete / canonical notebook not adopted
- CV / LB: 新規なし / なし
- implementation / Kaggle package / run: `1 / 0 / 0`
- inference output / submission: `0 / 0`
- 正規train / inference notebook: template placeholder、未変更
- Jupytext候補inference: `.py/.ipynb`実装済み
- blocker: 正規notebook採用、Kaggle package/run、output取得、提出の各別承認

## 2026-08-04 設計確定

- ユーザーは最終提出枠2つを使い、exp413単独を最終案にしない方針を明示した。
- exp497の科学的gate FAILは変更せず、final portfolioのreference overrideとして別実験化した。
- public-core weightをexp497 meta-fold係数中央値`0.13716473330712417`へ固定した。
- final式を`0.8628352666928758 * exp413 + 0.13716473330712417 * strict_public_core`に固定した。
- weight再fit/grid、Public LB tuning、router、Gold/contact、最終SG/warmup/projectionを禁止した。
- exp413とexp497を再学習・再生成せず、保存predictionの0-model CPU blendとした。
- technical gateだけを新規評価し、exp497 CV promotionを再判定しない。

## 根拠

- exp413: CV `7.884802794404715`、Public LB `7.201`。
- exp497 candidate: CV `7.87448814999802`、gain `0.010314644 ft`。
- exp497 meta-fold public-core weightsは5/5で正、中央値`0.13716473330712417`。
- ただしnonworse fold `3/5`、hidden-like `+0.105138/+0.097410 ft`、by-well p95/worst
  `+0.700720/+7.541588 ft`のため科学的promotion gateはFAIL。この判定を保持する。

## 実行予定inventory

| scientific variant | model config | fold | booster | PF | Beam | GPU | parent retraining |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Kaggle push前に再確認する。今回の実装では新規booster 0、保存exp497 booster 40 + Ridge 2、
保存exp413 booster 75を読む。今回はpushしない。

## コマンドログ

- `task new-steering ...`は環境に`task`がなく失敗した。代替の`make new-steering`でsteeringを作成した。
- `make new-exp EXP=exp509_exp413_strict_public_core_final_slot`でtemplate scaffoldを作成した。
- `make validate-exp EXP=exp509_exp413_strict_public_core_final_slot`はstrict PASS。
- `make validate-template`はPASS。
- `make update-summary`で`experiment_summary.md`へdesign-only実験を反映した。
- notebook実行、Kaggle API、学習、推論、提出は行っていない。

## 再現性メモ

- seed policy: `no_rng_saved_prediction_fixed_float64_id_order`
- stochastic components: 本実験内なし。上流はprediction SHAで固定する。
- CPU/GPU runtime: 将来のKaggle private CPU、GPU 0、internet off。
- input SHA: exp497 Stage I v4のprediction/model/Ridge/artifact SHAをconfigへ固定した。dynamic
  exp413はruntime/source SHAを固定し、prediction content SHAは実行時に記録する。
- model manifest: exp497 model-set SHAと6 artifact SHAを固定。新規model 0。
- prediction / submission SHA: 未生成。
- deterministic anchor: false。rerun一致前は昇格しない。

## 次のアクション

1. 別承認後に候補を正規inference notebookへ採用する。
2. Kaggle packageの76相当support file、embedded config/source、kernel sourceをreadbackする。
3. Kaggle run、output取得、submit-check、外部提出はそれぞれ別承認で進める。

## 2026-08-04 prediction-only候補実装

### 承認範囲

- ユーザーの「exp509を実装してください」を、候補Jupytext source/notebook、契約test、config、
  実験記録更新までの承認として扱った。
- 既存steeringどおり、正規`*_inference.ipynb`は上書きせずtemplate placeholderを維持した。
- Kaggle package/push/run、output取得、submit-check、competition submitは行っていない。

### upstream preflight

- exp497 Stage I version 4は14,151 rows / 3 wellsで完了し、LightGBM 24 + CatBoost 16、
  Ridge 2を保存済み。model-set SHAは`dcc2166f...626`、strict prediction SHAは
  `27641aa6...885`、blend prediction SHAは`c939c9f8...6f72`。
- 保存model inference v1は全model推論まで完了したが、一律`0.001 ft` parityでstrict
  `0.0012812500`、blend `0.0141953125`となり停止した。OOM、artifact欠落、model SHA不一致ではない。
- exp497側で実装済みの修正契約どおり、strict toleranceを`0.002 ft`、dynamic exp413とblendを
  `0.02 ft`へ分離した。既知v1差は通るが、旧一律`0.001 ft`なら落ちる専用testを追加した。
- bootstrap dependency 24 filesは全件ローカル存在を確認した。

### 実装内容

- `*_compact_selfcontained_inference.py/.ipynb`をJupytext起点で実装した。
- raw hidden sampleからexp510検証済みruntimeでexp413を動的生成し、公開test固定sidecarを使わない。
- dynamic exp413が先に作る`/kaggle/working/submission.csv`は
  `artifacts/exp413_intermediate_submission.csv`へ隔離する。
- exp497 Stage I v4の40 booster + Ridge 2をSHA検証して読み、strict public-core特徴とpredictionを
  hidden sample上で動的生成する。artifact欠落時の学習fallbackは持たない。
- exp497 coreの診断blend出力は`artifacts/strict_public_core_runtime/`へ隔離し、exp509 final名にしない。
- sample順へID one-to-oneで戻した`exp413_tvt`と`strict_public_core_tvt`をCSV境界で読み戻し、
  `0.8628352666928758 / 0.13716473330712417`のfloat64式を1回だけ評価する。
- Gold、guarded contact、same-well lookup、public output copy、weight refit/grid、row/well router、
  final SG/warmup/projection、外部submit処理を含めていない。
- component prediction、pooled/well/horizon/start-row差分readout、input manifest、float64 logical
  prediction SHA、gzip decompressed SHA、submission SHA、technical gateを保存する。

### 実行inventory

| scientific variant | new model config | fold | new booster | loaded exp497 booster | Ridge | loaded exp413 booster | weight refit | GPU training |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 40 | 2 | 75 | 0 | 0 |

- PF/Beamはraw hidden componentの再生成にだけ使用し、探索・候補選択・再fitは行わない。
- hidden well数が実行前に不明なためPF/Beam run数は動的。Kaggle push前にsampleから記録する。

### 検証ログ

- candidate source: 694行、8章、SHA `cc4dab6a...af5c`。
- candidate notebook: 18 cells（code 8 / markdown 10）、SHA `f24c796b...bee`。
- 親exp497 compact inferenceは343行/7章。exp509はcomponent inferenceとfinal blend、
  technical/reproducibility出力を分離した8章で、同一exp helperを呼ぶ薄いnotebookではない。
- dedicated tests: `6 passed`。
- dependency tests `tests/test_exp497_strict_public_core_fold_safe_ensemble.py`: `30 passed`。
- Jupytext round-trip、`py_compile`、Ruff `F821/F401/F811/E501`: PASS。
- `make validate-exp EXP=exp509_exp413_strict_public_core_final_slot`: strict PASS。
- `make validate-template`: PASS。
- repository-wide `make test`: `1861 passed, 8 skipped, 4 failed`。失敗は既知の対象外で、
  exp293 contract SHA不一致2件とexp296完了後status/run approval期待不一致2件。exp509専用testと
  依存exp497 testに失敗はない。
- notebookローカル実行、Kaggle API、学習、推論、提出は行っていない。
