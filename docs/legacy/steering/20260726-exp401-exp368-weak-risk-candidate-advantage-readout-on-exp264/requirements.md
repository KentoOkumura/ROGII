# 要件

## 依頼

exp368で「誤差の大きい区間」は比較的よく識別できたため、そのtarget-freeな
weak posteriorが後段のexp264 dual selectorに役立つかを、別実験として設計する。
今回はbacklog、steering、実験ディレクトリと設計だけを確定し、コード実装、
Notebook採用、Kaggle実行は行わない。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親実験はcorrected Stage C v6が確定している
  `exp264_exp263_candidate_confidence_dual_selector`とする。
- 補助入力は`exp368_marginalized_reliability_pf`の保存済みtarget-free
  block ledger / weak posteriorだけとし、exp368 Stage 1 PFは再開しない。
- exp368のknown-prefix NLL / weak-mass FAILを再分類せず、「PFのGR尤度を弱める」
  仮説から「selector用のrisk context」仮説を分離する。
- Stage 0は0 model / 0 booster / 0 predictionの決定論的readoutとする。
- `weak_posterior_mean`は連続値のまま使う。weak判定threshold、hard router、
  quantile別候補切替、threshold/grid探索を行わない。
- exp264の12候補を単一hard-select domainへ統合しない。
  `primitive_pair_bank` 11候補をprimary、`primitive_fixed_bank` 7候補を
  secondaryとして別々に読む。
- feature、fold、候補domain、合否条件をfreezeした後にだけsuffix truthを読む。
- Stage 1 selector学習はStage 0全gate PASSかつ別の実装・実行承認後だけ許可する。
- downstream TVT Stage D、inference、submissionは本実験の現時点の範囲外とする。

## 受け入れ基準

- `docs/legacy/steering/20260726-exp401-exp368-weak-risk-candidate-advantage-readout-on-exp264/`
  に仮説、特徴定義、Stage 0 / Stage 1、合否条件、禁止事項が固定されている。
- `experiments/exp401_exp368_weak_risk_candidate_advantage_readout_on_exp264/`
  にdesign-onlyの`config.yaml`、`README.md`、`SESSION_NOTES.md`、
  `result.md`、`metrics.json`がある。
- `KAGGLE_DIRECTION.md`の未着手backlogと`experiment_summary.md`にexp401が
  記録されている。
- Stage 0の入力SHA、期待行数3,783,989、well数773、block数15,174、
  candidate-long行数45,407,868、fold数5、候補domainが明示されている。
- Stage 0のtechnical gateとscientific gateが数値で事前固定されている。
- 将来のStage 1実行量が1 variant / 2 objectives / outer 5 × inner 4 /
  40 CPU LightGBM boosters、parent/control再学習0と明記されている。
- gzip生成物はraw gzip SHAではなくdecompressed logical-content SHAを主証拠とする。
- 現時点では実装済みコード、採用済みNotebook、Kaggle package/run、
  model、prediction、submissionがすべて0である。

## 2026-07-26 implementation-only追加承認

ユーザーの「exp401を実装してください」により、固定済みStage 0設計の
implementation-onlyを承認済みとする。承認範囲は次に限定する。

- 別名compact self-contained train Jupytext source / Notebook候補
- fail-closed inference Jupytext source / Notebook候補
- Stage 0 technical / scientific contractの専用test
- config / README / SESSION_NOTES / result / metrics /
  experiment_summary / KAGGLE_DIRECTIONの実装状態更新

正規Notebook上書き、Kaggle package / push / run、Stage 1の40 CPU
selector boosters、downstream TVT、inference、submissionは承認に含めない。
