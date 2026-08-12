# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `last_anchor_better_candidate_confidence_pair_cache` 実装契約に従い、exp072 `last_anchor` より OOF RMSE が良い known 33 path を catalog として固定する。後続 selector / ML / fixed blend が共通利用するのは family 圧縮した core 12 primitive、target-free confidence、有望 8 pair、3 named combination に限定し、candidate-major cache と chunked virtual loader を実装する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- HMM+LGB `exp221/234/240`、selector / TVT model output、target / error / oracle confidence は除外する。
- reference 33 本の全 row 値は複製せず、core 12 本だけを float32、ID 付き、outer-fold partition で保存する。
- confidence は source artifact に存在する target-free 列だけを保存し、未提供値は NaN + valid flag / manifest `unavailable` とする。
- 378 pair の full row tensor、exp104 5本の追加 pair sweep、pair / triple の再帰 closure は作らない。
- outer-fold eligibility は対象 outer-valid fold を除いた 4 folds だけの RMSE で決める。
- Stage 0 は既存 OOF artifact の cache 生成とする。
- Stage 1 は raw-test-ready primitive 6本・pair 5本・`exp226_w500_50_50` の current-test parityを確認する。
- Stage 1は6 primitiveの候補値に加え、同じraw-test生成処理から取得したsource-native confidenceを
  `confidence__<candidate_id>__<field>`で同じParquetへ保存する。exp264 Stage A採用schemaが必要とする
  exp226 GR delta、HMM std/loglik/self-GR診断、PF std、Beam family stdを欠損時fail-closedで保証する。
- `likpf_mean`にはsource-native scalarがないため推測値を作らず、明示的な
  `confidence__likpf_mean__confidence_valid=False`だけを保存する。
- ユーザーの提出指示により、Stage 1 inferenceは静的current-test予測を提出値として読まず、exp073 deterministic PF/Beam replay、exp209 exact HMM、exp223 self-GR HMM、exp226 K16の固定sourceをraw competition testへ適用して6 primitiveを再生成する。
- 提出対象は追加weight fitを伴わない固定`exp226_w500_50_50 = 0.50*exp226_k16 + 0.25*likpf_mean + 0.25*exact_hmm`だけとする。train-only最良、fold別outer-convex係数、selector outputは提出しない。
- 学習 0 variant、LightGBM config 0、fold training 0、booster 0、parent/control 再学習 0。

## 受け入れ基準

- `candidate_catalog.json` が 33 reference / 12 core、raw-test-ready core 6本、cache role と source provenance / SHA を固定する。
- core 12 の value/confidence partition、`outer_fold_eligibility.csv`、8 pair の shortlist/readout、3 named combination、DAG cycle guard、`cache_manifest.json` を生成できる。
- loader が primitive / pair / named combination を名前と fold/chunk で要求し、pair/triple を要求時に再構成する。w500 alias と `exp226_w500_50_50` の式を精密に再現する。
- canonical exp072 と `id/well/well_row_idx/outer_fold/md_since` が完全一致し、coverage、finite、dtype、schema、content SHA を manifest に記録する。
- pair / combination の small parity sample で direct formula と virtual loader が float32 許容差内で一致する。
- cache artifact 自体は deterministic として扱い、feature content SHA、source SHA、schema SHA、Kaggle kernel version を記録する。モデル学習・提出は行わないため model / prediction / submission SHA は not_applicable と明記する。
- Stage 1 inferenceではraw-test primitive content SHA、prediction content SHA、submission SHA、Kaggle kernel versionを記録し、`submission.csv`をsample submissionの行順・ID・列・finite契約に対して検証する。
- current-test confidenceはOOFと同じfield名・well-level loglik normalizationを使い、全required列、finite、
  validity、namespace数を検証する。formula confidenceは保存時に平均せず、後段がparent namespaceから再構成する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
