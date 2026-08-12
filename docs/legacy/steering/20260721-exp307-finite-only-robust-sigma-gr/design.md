# 設計

## 仮説

known-prefix欠損GRを0補完した現行scaleは欠損率に引かれる。有限pairだけのpopulation stdで0補完除去の効果を分離し、有限pairのMADをprimaryにすれば、外れ値に頑健なGR尤度校正としてexact-HMMを改善できる。

## アプローチ

exp209 exact-HMMの観測モデル、state grammar、欠損区間GR補間を固定し、既知prefixから`σ_GR`を作る関数だけを置換する。`finite_std`で0補完除去の効果を読み、事前primary `finite_mad`で重い裾に頑健なscaleを評価する。truthはprediction freeze後のreadoutにだけ結合する。

## 実験範囲

- 対象: `exp307_finite_only_robust_sigma_gr`
- Route: `pf_beam`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更: 既知prefix `σ_GR`推定だけ。
- 固定: `step=0.35`, `n_rates=41`, `rate_span=0.10`, `sig_r=0.002`, `sig_p=0.02`, `mom=0.998`, Gaussian emission、evaluation GR補間、typewell処理、raw GR比較、start prior、posterior mean。
- 除外: LikPF/PF/Beam再生成、補間GR重み、遷移noise、affine較正、Student-t、temperature、inference、submission。

## 固定scale契約

既知prefixで`TVT_input`、horizontal GR、そのTVTに線形補間したtypewell GRがすべて有限な行だけをpairとし、`e_i=GR_h,i-GR_tw(TVT_input_i)`を作る。

- `finite_std`: `std(e, ddof=0)`。
- `finite_mad`: `1.4826 * median(|e - median(e)|)`。
- 有効pairが20未満、scaleがnonfiniteの場合は30。
- 最終scaleは`clip(scale,10,60)`。
- affine係数は`a=1,b=0`、残差中心をemissionから引かない。
- `finite_mad`を唯一のpromotion candidate、`finite_std`をdiagnosticに固定する。

## 検証方法

1. raw train file identity、exp209 scientific config、saved HMM control、saved LikPFのSHAをpreflightする。
2. TVT truth列を読み込まず、各wellのprefixから2 scaleとaudit列をfreezeする。
3. 2 variantsを同一exp209 decoderへ個別に渡し、well_id/row_id順でpredictionとposterior diagnosticsをfreezeする。
4. freeze後だけunknown suffix truth、5 folds、exp115 hidden-like assignment、saved LikPF predictionをjoinする。
5. primary gateをoverall/fold/distance/hidden-like/by-well/50:50 blendで判定する。
6. diagnosticの結果にかかわらずprimary FAIL後の推定量、clip、fallback、blend weight gridを行わない。

## 実行量

- active scientific variants: 2 (`finite_std`, `finite_mad`)
- HMM well-runs: 1,546
- model / LightGBM config / trained fold / PF / Beam / booster: `0 / 0 / 0 / 0 / 0 / 0`
- control再実行: 0
- GPU: 0、Kaggle CPU予定、internet off、runtime上限8.5時間/1 run

## 生成物契約

- scientific contract JSON
- input/control manifest JSON
- well別scale audit CSV.gz
- variant prediction/posterior diagnostics CSV.gz
- overall/fold/distance/hidden-like/by-well metrics CSV/JSON
- fixed LikPF 50:50 readout
- gate summary JSON

scale/predictionをfreezeする前にunknown suffix truth/error/oracle列を読まない。CSV.gzはdecompressed content SHAを主証拠にする。

## 再現性設計

- RNGなし。well ID昇順、固定2 variants順で処理する。
- outer workersとNumba threadsはexp209 v5の`2 x 2`を開始点にし、実装時に固定して記録する。thread変更はparityなしに許可しない。
- raw identity、exp209 control、saved LikPF、scale audit、variant prediction、metricsのcontent SHAを保存する。
- model/submission SHAは非該当。Kaggle kernel versionとbootstrap内config/source SHAを実行時に記録する。
- deterministic submission anchorではなく、train-side deterministic candidate auditとする。

## リスク

- 0補完除去とMADを同時に解釈すると原因が曖昧になるため、finite std diagnosticを必須にする。
- MADは周期的misregistrationを観測noiseから除きすぎ、尤度を過信する可能性がある。worst/p95 guardを必須にする。
- prefix残差がsuffixを代表しないwellがある。P3縮約はexp310へ分離し、本実験内で救済しない。
- exp209 full decodeは長時間。control再実行を禁止し、1 run 8.5時間でfail-closeする。

## 次のアクション

self-contained実装とKaggle CPU version 2の評価を完了した。finite-only scaleはGR emissionを過度に鋭くして全主要gateをFAILしたため、同一結果上の救済を行わず、exp307 PASSを固定依存にする後続も未実行で閉鎖する。
