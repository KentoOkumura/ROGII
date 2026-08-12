# 要件

## 依頼

`exp512_hjyact_v2_final_10pct_hedge_on_exp413`でアンサンブルした公開Notebook実装の完成版finalを
単独で提出し、Public LBを確認する実験を設計する。初回依頼ではバックログ、実験ディレクトリ、steeringを
作成して設計を確定した。2026-08-05の追加依頼でexp512の実行失敗原因を反映した候補実装と静的検証まで
承認された。正規Notebook採用、Kaggle package / run、output取得、提出は行わない。

## 仮説と変更点

完全な`hjyact_v2_final`単独が公開source scoreを再現するかを測る。exp512からexp413 runtime、50/50 blend、
cross-consumer reuseだけを除き、公開final内部は固定する。exp512 v1/v2で判明した旧competition/Ridge mount
参照を、内容検査とSHA監査に基づく一意root解決へ置換する。

## 制約

- 対象実験: `exp513_hjyact_v2_final_standalone_public_lb_audit`
- Route: `ensemble`。公開実装内でPF/Beamと保存済みMLの両方が最終予測へ本質的に寄与する。
- 親実験: `exp512_hjyact_v2_final_10pct_hedge_on_exp413`
- 公開source: hjyact `ultimate-pf-config-strategy-a-reproducible-score` version 2 / run `337064157`。
- final境界: exp512の`after_complete_hjyact_v2_final_stack_and_pf_seed_branch_hedge`。
- exp413再生成、`0.50/0.50` blend、cross-consumer candidate reuseを含めない。
- SP45 / learned `0.60/0.40`、guarded overlap、balanced visible-prefix、model-package guard、
  PF seed-branch hedge、write順をsourceどおり固定する。
- visible output CSV、static prediction sidecar、特定well ID、14,151 rows / 3 wellsを使う予測分岐は禁止する。
- profile、weight、threshold、seed count、particle count、後処理を探索しない。LB結果後も救済調整しない。
- honest OOFはない。公開sourceのPublic LB `6.568`は参照値であってexp513の実績値ではない。
- 学習は行わない。新規booster 0、親/control再学習0。
- 再現性は`docs/06_reproducibility.md`に従い、stochastic PF/Beam、Kaggle bootstrap、model / feature /
  prediction / submission SHAを記録する。
- 現在の正規train / inference Notebookはtemplate placeholderであり、別名candidateを実装しても別承認まで
  採用・実行・提出しない。

## 受け入れ基準

### 今回の設計段階

- `docs/legacy/steering/20260805-exp513-hjyact-v2-final-standalone-public-lb-audit/`に要件、設計、tasklistがある。
- `experiments/exp513_hjyact_v2_final_standalone_public_lb_audit/`にroute、lineage、固定source identity、
  実行量、検証gate、禁止事項、未承認状態が記録されている。
- `KAGGLE_DIRECTION.md`にP0候補として追加し、exp512の追加rerun / submission判断との関係を記録している。
- 初回設計段階では実装・Kaggle操作・提出を行っていない。

### 実装・静的検証段階

- Jupytext percent形式の別名compact self-contained inference候補を作り、正規Notebookは別承認まで上書きしない。
- exp512候補の公開成分境界だけを抽出し、exp413 / downstream blend / shared-DAG経路がないことをtestで固定する。
- exp512 v1の旧competition rootとv2の旧Ridge root直接参照を、内容検査による一意root解決とSHA監査済みroot注入へ置換する。
- 候補sourceをKaggle 1 MiB source制限より十分小さくし、最終package sizeはpackage作成時に再確認する。

### 将来のKaggle実行段階

- dynamic sampleから全予測を生成し、ID one-to-one、sample order、finite、duplicate 0を満たす。
  static/precomputed predictionとinference-time training fallbackは0とし、source-defined defensive fallbackは
  source parityのため保持して実行logへ記録する。
- visible sample identity確認後だけ、source final SHA
  `b192d3f348ae00680dc4df942b95cef5fd708c636a741f77dfb6b6e89b9ded4a`とのparityを判定する。
- 同一Kaggle GPU / internet-off条件2回でprediction / submission SHAが一致している。
- `submit-check`を通し、別承認後にcode submissionを1回だけ行ってPublic LBを記録する。
- Public LBがsource `6.568`と表示精度で一致すればreproduction PASS、異なればno-retuneで原因を記録する。
- deterministic anchorと呼ぶ場合は、feature content SHA、model SHA、prediction SHA、submission SHA、
  Kaggle kernel versionを記録する。
- gzip生成物はraw `.csv.gz` SHAではなくdecompressed content SHAを主証拠として記録する。

## 次

別承認後に正規Notebook採用と最小bootstrap package作成へ進み、1 MiB制限とembedded source/configを
readbackしてからKaggle GPU / internet-off parity runを行う。competition submitはさらに別承認とする。
