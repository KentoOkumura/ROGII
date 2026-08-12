# 設計

## アプローチ

1. raw train と visible test の horizontal GR availability から、well ごとの欠損率、連続 missing-run 長、評価 suffix 内の欠損 run を集計する。
2. exp221 train v3 の固定 OOF control cache と exp148 `lgb_mean` OOF を読み、ID coverage と SHA を検証する。
3. exp221 の exact HMM 実装を複製し、GR emission 行列の構築直後に `eval_raw_gr_missing` rowだけ `emission_ll[row, :] = 0.0` とする。LGB unary はその後に加算するため、欠損 row は「GR更新なし、LGB unaryあり」になる。
4. mask-only 1 variant を全 773 wells に生成し、固定 control と同一 ID で比較する。
5. row metrics、group metrics、by-well metrics、missing run / post-gap bucket、finite coverage、control からの連続分岐 segment を保存する。

## 実験範囲

- 対象実験: `exp247_missing_gr_masking`
- Route: `ensemble`
- 親実験: `exp221_lgb_oof_gaussian_emission_hmm_on_exp148`
- 比較入力: exp221 train v3 `hmm_lgb_exp148_lgb_mean_s2000_l0500`
- 変更する変数: evaluation suffix の raw horizontal GR 欠損 rowにおける GR emission contribution のみ。
- 固定する変数: exp221 grammar、grid step/band、rate lattice、transition、prefix calibration/sigma、GR補間値、LGB OOF center、LGB sigma/lambda、score rows、hidden-like role definition。
- 学習コスト: active variant 1、LightGBM config 0、fold 0、booster 0、control再学習なし。
- inference: disabled。train-side readoutのみ。

## 診断定義

- missing-run bucket: `observed`, `1_4`, `5_31`, `32_127`, `128_255`, `256_999`, `1000_plus`。
- post-gap bucket: missing run 終端後の observed rowsについて `post_gap_001_128`, `post_gap_129_256`, `post_gap_257_plus`。missing run 内は別 bucket とする。
- distance bucket: exp221 と同じ `000_050` から `1000_plus`。
- divergence: `abs(mask-control) > 1e-6 ft` を changed row とし、well内の連続 changed segment 長、missing runとの重なり、最大絶対差を記録する。
- finite-state coverage: mask/control prediction、posterior std、log-likelihood の finite row/well coverageを記録する。

## 再現性設計

- seed policy: `no_new_rng_exact_hmm_deterministic_ablation`。
- stochastic 処理の有無: 新規処理には乱数なし。固定 OOF control / LGB unary の upstream provenanceはSHAで固定する。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: well順をsortし、乱数なし。Numba thread数とouter worker数をconfig・summaryに記録する。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU false、internet false。モデル学習なし。
- train cache / test feature regeneration の SHA 記録方針: raw train file mapping SHA、control cache SHA、LGB OOF SHA、mask output raw gzip SHAとdecompressed content SHA、schema/group/by-well SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: model再学習・submissionなし。固定 LGB source manifestは親実験の記録を参照し、今回の mask prediction content SHA を記録する。
- Kaggle package bootstrap 確認方針: prepare後にkernel metadata、bootstrap内config、canonical experiment名、GPU/internet、kernel_sourcesを照合する。

## リスク

- リークリスク: missing mask、emission、gateはraw GR availabilityだけで作る。true TVTとhidden-like roleは集計専用とし、関数境界で渡さない。
- CV/LB 不一致リスク: exp221はCV改善に対しPublic LB転移が弱かったため、train-side positiveでもraw-test inferenceやsubmitへ自動昇格しない。
- ランタイム/メモリリスク: exact HMMは約5時間級。controlを保存済みcacheから読み、mask-only 1 variantに限定する。row cacheはgzipで保存する。
- 再現性リスク: pandas/NumPy/Numba版差が浮動小数点末尾に影響しうる。input/output content SHA、runtime version、thread設定を記録する。
- 解釈リスク: missing runが少ない場合、overall差が小さくbucket分散が大きい。run-length gateの探索へ進まず、初回結果をそのまま閉じる。
