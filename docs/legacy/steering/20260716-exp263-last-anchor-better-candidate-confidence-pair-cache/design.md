# 設計

## アプローチ

Stage 0 で canonical exp072 row index を一度固定し、source resolver が解決した保存済み OOF artifact を ID / well-row で join する。core 12 は candidate-major、outer-fold partition の Parquet として保存し、common shape/context と source に存在する family-specific confidence を分離する。pair / triple は catalog と formula DAG だけを保存し、loader が primitive partition から chunk ごとに再構成する。

Stage 1 inference notebook はdeployability manifestでraw-test-readyとした6 primitiveをraw competition testから再生成する。likPF / PF-ANCC / Beamはexp073 stable-seed replay source、exact HMMはexp209 source、self-GR HMMはexp223 source、K16はexp226 sourceを固定する。6 primitiveから5 pairと固定named formulaを再構成し、current public testでは保存済みexp237 candidate frameとの数値parityも確認する。hidden testではID集合が異なる静的referenceを予測入力に使わず、比較をskipする。最終`submission.csv`は固定`exp226_w500_50_50`列だけからsample順に生成する。

confidenceは候補値と同一generator callから回収する。exact/self-GR HMMは`std_eval`とwell-level
`loglik`を保存し、後者をそのwellのunknown row数で割った`loglik_per_row`もOOFと同じ規則で作る。
self-GRはquality/peak/gap/typewell agreement/valid、PFは`pf_ancc_std`、Beamは`beam_std_d`を使う。
exp226は別proxyを発明せず、同じ`predict_well()`が返す`PredictionResult.delta`をOOF
`gr_delta`と同じ`geometry_gr_delta`へ写す。likPFはnative scalarなしを明示する。formulaは親confidenceを
平均せず、後段exp264がnamespaced parent fieldを決定的に展開する。

## 実験範囲

- 対象実験: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- Route: `pf_beam`
- 親実験: exp072 canonical cache。candidate provenance として exp103/104/192/209/223/225/226/231/232/233/243 を参照する。
- 変更する変数: candidate inventory/cache role、confidence schema、deployability tier、pair shortlist、named formula、virtual loader、raw-test regeneration orchestration、固定formula submission出力、Stage 1 namespaced confidence出力。
- 固定する変数: exp072 row identity、anchor 15.909866082、reference 33/core 12/raw-test 6、pair 8、named triple 3、window 32/128/512、outer 5 folds、audit metrics。

## 再現性設計

- seed policy: Stage 0は`not_applicable_deterministic_artifact_transform`。Stage 1 PF/Beamはexp073の`stable_sha256_per_well`を固定する。K16/HMMは乱数を使わない。
- stochastic 処理の有無: Stage 1 PF/Beam/likelihood-PFのみ。global RNGではなくwell/split/family由来stable seedを使う。
- PF/Beam / likelihood-PF / seed bagging の有無: Stage 0は保存済み値を読むだけ。Stage 1はexp073 sourceからraw test replayを再実行し、128 seeds / 500 particlesの固定設定を使う。
- 並列処理と乱数の関係: 乱数なし。candidate 順と partition 順を catalog / fold / row index でソートし、thread scheduling に依存しない。
- CPU/GPU runtime と deterministic flags: CPU、GPU 不要、0 booster。Parquetと JSON/CSV の deterministic content hash を主証拠にする。
- train cache / test feature regeneration の SHA 記録方針: source file SHA256、generation config SHA、schema SHA、candidate partition content SHA、ID index SHAを `cache_manifest.json` に記録。gzip sourceは decompressed content SHAも記録する。
- model manifest / prediction / submission SHA 記録方針: 学習・modelはnot_applicable。Stage 1は6 primitive content、formula parity、prediction content、submission file SHAを記録する。
- Kaggle package bootstrap 確認方針: strict prepare 後に bootstrap ZIP 内 `config.yaml`、loader、schema/catalog の SHA と loose file を比較する。

## リスク

- リークリスク: target-derived audit metricsは catalog/pair manifest に隔離し、row feature/confidence に入れない。outer eligibilityは outer-valid fold を除外して計算する。
- CV/LB 不一致リスク: train-only core 6本を raw-test-ready に暗黙昇格させない。提出は固定OOF RMSE 8.238331の1式だけとし、train-only 8.209225とfold別fit 8.231651は使わない。
- hidden testリスク: current-test静的artifactはparity referenceだけに限定し、prediction sourceには使わない。sample IDがreferenceと異なるhidden runではreference比較をskipし、raw test再生成値だけで提出する。
- ランタイム/メモリリスク: 3,783,989 rows × 12 candidates をwideに全展開せず、candidate-major/fold/chunk で読み書きする。pair/triple full tensorは保存しない。
- 再現性リスク: 外部 artifact path と gzip metadata の差は resolver + decompressed content SHA で検出する。ローカルにない confidence は推測せず unavailable にする。
