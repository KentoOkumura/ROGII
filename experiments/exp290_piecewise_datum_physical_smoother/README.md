# exp290_piecewise_datum_physical_smoother

## 状態

- ルート: `pf_beam`
- 状態: Stage 0 scientific guard FAIL / branch closed
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-19
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`

## 仮説

exp226の誤差にはpersistentなwell datum成分が大きいが、prefix/group/neighborのbiasを一回だけ
推定してsuffixへ足す方法は不安定だった。exp226のgroup-safe geometryを固定し、known prefixで
well固有の信頼度を校正しながら、Type Well GR evidenceでbounded piecewise datumの事後分布を
平滑化すれば、exp281の中心誤差改善を残しつつcatastrophic wrong offsetを制限できる。

## 単一モデル境界

- 物理式: `TVT_w(t) = g_w(t) + delta_w(t)`
- 固定geometry: exp226と同じgroup-safe `tvt_geop`生成契約
- latent datum: absolute `[-15,+15] ft`、0.5 ft grid、minimum duration 256 rows
- observation: known-prefix residual + known-prefix-calibrated robust Type Well GR likelihood
- hierarchy: well固有calibration、Type Well/近隣はscale・hazard・noise事前だけ
- solver: exact log-space semi-Markov forward-backward
- output: 一つのposterior mean TVT prediction
- 使用しないもの: ML、candidate bank、Viterbi選択、blend、well selector、posthoc offset、oracle

## 段階とguard

1. Stage 0: known prefix末尾から512/256/128 rows戻した3 cutの直後128-row windowで、過去cutだけからreliabilityを作り、RMSE 0.20 ft以上改善、large-error correction sign 0.58以上、4/5 folds改善、well p95非悪化を要求する。
2. Stage 1 shadow guard: direct OOF 8.0以下、4/5 folds改善、1000+とhidden-like 2面非悪化、well p95 15以下、worst 45以下。
3. 単独inference guard: direct OOF 7.0以下、5/5 folds改善、hidden-like 2面改善、well p95 13以下、worst 40以下。

不通過時はparameter/group/neighbor/likelihood救済、inference、submissionへ進まない。

## exp289との違い

exp289はformationからfault cut付き2D共通surfaceを解く空間topology仮説である。exp290はexp226
geometry上のwell内datumをMD方向に解く状態空間仮説であり、相互predictionを入力・blendしない。

## 検証方針

- Fold: 既存5-fold GroupKFoldを固定
- Group: well単位
- Score rows: `TVT_input.isna()`
- Stage 0 rows: known prefix内の3 fixed cut直後の128-row pseudo-tailだけ
- Leakage check: outer-valid formation 6列、official suffix true TVT/error、pseudo-cut後のheld-known TVTをprediction freeze前に除外
- 比較: 保存済みexp226 OOF。control再生成なし
- oracle: 全粒度で禁止。診断値を達成可能下限として扱わない

## 実行入口

- Stage 0 source: `exp290_piecewise_datum_physical_smoother_compact_selfcontained_train.py`
- canonical Stage 0 notebook: `exp290_piecewise_datum_physical_smoother_train.ipynb`
- compact source parity notebook: `exp290_piecewise_datum_physical_smoother_compact_selfcontained_train.ipynb`
- disabled inference source/notebook: `exp290_piecewise_datum_physical_smoother_compact_selfcontained_inference.py` / `.ipynb`
- Kaggle kernel: `kentookumura/exp290-piecewise-datum-physical-smoother-train` version 1、id_no `127881061`
- Kaggle runtime: CPU、GPU/internet off、587.042秒、peak RSS 1,215.824 MB
- notebook実行: 初回実行はKaggle CPU kernelを正とする

## 実装済み Stage 0

- exp226 fold map / fold別 kappa のSHAを検証し、outer-valid wellをdonor geometryから除外する。
- outer-valid horizontal readerは`MD/X/Y/Z/GR/TVT_input`だけをmaterializeし、`TVT`とformation 6列を除外する。
- 3 pseudo-cutそれぞれでexp226 geometryをcutの`TVT_input`へ再anchorし、datum prior meanは0に固定する。
- outer-train masked-prefix backtestからscale/hazard/noiseだけを階層化し、exact Type Well groupはscale location、spatial k=16はvarianceだけを更新する。
- 61 offset states × 5 duration phasesのexact log-space forward-backwardからposterior meanだけを出す。
- windowごとのprediction SHAをfreezeしてからheld-known `TVT_input`を結合し、Stage 0 guardを判定する。
- 専用tests 11件でleakage、mask、stable neighbor、transition、state bound、truth-after-freeze、符号禁止、fail-closed inferenceを確認する。

## 結果

| メトリック | 値 |
| --- | --- |
| Stage 0 base pseudo-tail RMSE | 1.436926 |
| Stage 0 model pseudo-tail RMSE | 1.403407 |
| 改善 | 0.033519 ft（guard 0.20未達） |
| large-error correction sign | 0.483111（guard 0.58未達） |
| fold改善 | 5/5（PASS） |
| well RMSE p95 | 2.437183 → 2.440010（FAIL） |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- per-well offsetの自由度、tail暴走を防ぐbound、group情報の用途、単一posterior mean出力を実装前に固定した。
- exp281のnegative resultを踏まえ、自由random walk、always-on shift、hard top-1を禁止した。
- compact self-contained notebook、SHA manifest、truth-after-freeze guard、disabled inferenceを設計どおり実装した。
- 296,832 rows / 773 wells / 2,319 windowsでtechnical guardを全通過し、全5 foldsで小幅改善した。

### 悪かった点

- RMSE改善は0.033519 ftに留まり、固定0.20 ft guardへ届かなかった。
- large-error correction signは0.483111でchance未満、well p95も0.002826 ft悪化した。
- scientific guard FAILのため、bounded datum / GR識別性をStage 1へ昇格できない。

### リスク / 注意

- pseudo-cut reliabilityがofficial suffixの信頼度へ移らない可能性がある。
- Type Well GR evidenceが弱いwellではposteriorが誤ったbounded offsetへ張り付く可能性がある。
- Stage 0 runtimeは587.042秒、peak RSSは1,215.824 MBでtechnical runtime guardはPASSした。
- deterministic anchorではない。Kaggle rerun、raw-test regeneration parityは未確認である。
- Stage 0の128-row windowはminimum duration 256 rowsより短くreset不能なので、ここではconstant bounded datumの識別性だけを評価する。

## 次

exp290 branchは固定failure policyどおり閉じる。parameter/group/likelihood救済、Stage 1、raw-test inference、submissionへ進まない。

## 表記

用語は`backlog/KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせ、実験名や設定名を除いて日本語優先で記録する。
