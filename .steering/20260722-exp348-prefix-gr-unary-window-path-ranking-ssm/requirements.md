# 要件

## 依頼

2026-07-22は、exp332の「固定長window内で経路全体を学習する」という着眼点を維持し、全状態のpartition functionを計算するexact structured NLLの代わりに、正例経路が固定負例経路より高得点になるpath-ranking loss案をdesign-onlyで固定した。2026-07-24のユーザー依頼で、exp347のterminal decisionを先行条件として満たしたためimplementation-onlyへ進める。別名compact self-contained train/inference候補とcontract testsを実装するが、正規Notebook編集、Kaggle package/push/run、推論、提出は行わない。

## 仮説

256-row windowごとにtruthから作る1本のlegal positive pathと、truth/errorによる事後選択をしない固定negative path bankを事前生成し、path全体のunary + fixed transition scoreをranking lossで比較すれば、exp331のpointwise local CEよりtransition-awareな学習を保ちながら、exp332のnormal/label-conditioned 4-sweep exact DPを学習loopから除去できる。

## 制約

- Route: `ensemble`。
- 親実験はterminal closedの`exp332_prefix_gr_unary_fixed_window_structured_ssm`。exp332の結果、記録、Notebook、configを変更しない。
- exp347 batched exact DPを優先する。exp348の実装・Stage 0は、exp347がterminal decisionに到達した後、またはユーザーが明示的にranking branchを先行指定した場合だけ検討する。両案を同時GPU実行しない。
- 256 rows、3 scheduled slots、最大3 active windows/well/epoch、8 epochs、architecture、preprocessing、teacher boundary、local CE重み`0.25`、fixed exp209 grammar、full-well exact decode、controls、fold/gateはexp332を維持する。
- structured NLL `1.0`だけをpath-ranking loss `1.0`へ置換する。margin、negative bank、path score normalizationは固定し、同一実験内でgrid探索しない。
- positive/negative path bankはwindow schedule/boundary確定後、model fit前に生成・SHA freezeする。outer-valid/current-test truth、error、oracleでnegativeを選ばない。
- Stage 0は1 benchmark variant、固定16 windows、一時neural model 1、永続model 0、LightGBM config/booster/PF/Beam/control再学習0。実装は2026-07-24に承認済み、GPU実行は別承認なしに行わない。
- Stage AはStage 0全gate PASSと別承認後だけfold 0 / model 1。Stage B/C、推論、提出はさらに別承認とする。
- 再現性は`docs/06_reproducibility.md`に従い、window/boundary/positive/negative/dedup manifest、input/model/prediction/package/kernel SHAを記録する。

## 受け入れ基準

- `.steering/20260722-exp348-prefix-gr-unary-window-path-ranking-ssm/`と`experiments/exp348_prefix_gr_unary_window_path_ranking_ssm/`が存在し、仮説、path bank、loss、段階gate、禁止事項、実行量が記録されている。
- `config.yaml`は`implemented_waiting_stage0_approval`、implementation approvalのみtrue、Kaggle/Stage A/inference/submission承認falseでfail-closedになっている。
- positive pathはGaussian sigma`0.35 ft`のlabel emission + fixed exp209 grammarによるtraining-only label-conditioned Viterbi 1本とする。
- negative bankは最大16本。position offsets 6、constant rate-index offsets 4、midpoint rate pulses 4、保存済みexp209 path 1、geometry-only path 1を固定し、positiveと重複するものだけ削除する。最低12 unique negatives未満はfail-closedとする。
- path scoreはvalid rowあたりの`neural unary + fixed transition/boundary log potential`平均。lossはmargin`0.05`のpairwise softplus ranking平均`1.0` + local CE`0.25`とする。
- Stage 0 gateはpath bank technical checks、early-holdout positive top-1 rate`>=0.80`、positive-negative mean score margin`>=0.02`、保守的fold runtime`<=8.5 h`、peak`<=14 GB`を要求する。
- Stage Aではfull-well exact decodeでexp332と同じreal-vs-shuffle/geometry/exp209/tail safety gateを要求し、ranking accuracyだけで昇格しない。
- compact source候補、別名Notebook候補、contract testsを作成し、正規Notebook採用、Kaggle package/output、model、prediction、submissionが未作成であることを明記する。

## Assumption

固定negative bankが実際の誤経路を十分に代表するかは未検証であり、設計時点では科学的改善を主張しない。

## 次のアクション

正規Notebook採用とKaggle T4固定16-window Stage 0は別承認後だけ行う。実行前に1 variant / temporary neural model 1 / persisted model・trained fold・LightGBM・booster・PF/Beam・control再学習各0を再確認する。
