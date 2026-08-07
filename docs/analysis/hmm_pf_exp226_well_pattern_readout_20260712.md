# HMM / PF / exp226 well pattern readout 2026-07-12

## 目的

このメモは、2026-07-11 から 2026-07-12 のセッションで行った well 単位の診断を集約する。

調査対象は以下。

- PF は大外しだが HMM は比較的当たる well。
- HMM は大外しだが PF は比較的当たる well。
- HMM/PF と exp226 の差。特に GR 主体の HMM/PF と、z/geometry 主体の exp226 で何が違うか。

これは既存 OOF / by-well artifact を読む diagnostic であり、新規学習、提出候補、anchor 更新ではない。

## 参照元

- HMM: `/tmp/kaggle-output/exp223-selfgr-hmm-train-v1/artifacts/exp223_joint_typewell_self_gr_hmm_likelihood_probe_by_well_delta.csv`
- HMM feature / self-GR signal: `/tmp/kaggle-output/exp223-selfgr-hmm-train-v1/artifacts/exp223_joint_typewell_self_gr_hmm_likelihood_probe_joint_typewell_self_gr_hmm_likelihood_probe_train_features.csv.gz`
- HMM generation summary: `/tmp/kaggle-output/exp223-selfgr-hmm-train-v1/artifacts/exp223_joint_typewell_self_gr_hmm_likelihood_probe_by_well_generation_summary.csv`
- PF map: `artifacts/pf_beam_disagreement_error_map/pf_beam_disagreement_well_map.csv`
- typewell / XY: `artifacts/typewell_position_groups/native_overlap_1_well_position_typewell_summary.csv`
- exp209 fixed HMM blend: `/tmp/kaggle-output/exp209_exp072_exp205_joint_exact_parity_fast_cache_generation/train_v5_small/artifacts/exp209_vs_exp072_exp205_by_well_delta.csv`
- exp226 by-well metrics: `/tmp/kaggle-output/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/train_v1/artifacts/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_by_well_metrics.csv`
- exp226 OOF predictions: `/tmp/kaggle-output/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/train_v1/artifacts/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train_oof_predictions.csv.gz`
- train raw wells: `data/raw/train/*__horizontal_well.csv`

## 定義

主な比較対象。

- HMM: exp223 `hmm_selfgr_boost_only_a070_c100`
- PF primary: exp072 `likPF_mean`
- pure PF: exp072 `pf_ancc`
- exp226: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction` train OOF by-well `rmse`

基本閾値。

- 大外し: RMSE `>= 30`
- 比較的当たり: RMSE `<= 10`

exp226 についての注意:

- ユーザー観点では z-only 系として扱ったが、実装上の selected variant は `v6_k16_geometry_gr_u_projection`。
- K=16 segment spline、XY local-linear kNN、adaptive kappa、near-strike ANCC local theta、typewell GR correction、U-projection を含む。
- HMM/PF ほど GR 主体ではないが、完全な z-only ではなく、z/geometry 主体 + 軽い GR correction と見るのが正確。

## 1. PF bad / HMM good

定義:

- `likPF_mean RMSE >= 30`
- `HMM RMSE <= 10`

該当 well は 7 本。

| well | HMM RMSE | likPF RMSE | pure PF RMSE | 主な特徴 |
| --- | ---: | ---: | ---: | --- |
| `86454a6f` | good | bad | bad | GR 欠損 81% の例外的 worst PF。HMM は小さい bias。 |
| `7987f2f2` | good | bad | borderline | PF が whole-well offset。HMM は局所 GR/typewell に乗る。 |
| `5f4d2a52` | good | bad | bad | PF offset 系。 |
| `b3388334` | good | bad | bad | PF offset 系。 |
| `3417285d` | good | bad | bad | PF offset 系。 |
| `ba48188d` | good | bad | bad | PF offset 系。 |
| `efe96181` | good | bad | bad | PF offset 系。 |

集約特徴。

- PF の失敗は whole-well vertical offset が中心。
- HMM bias は小さい。
- nearest 8 wells に同じ strict 傾向はほぼ出ず、空間的に伝播する局所問題ではない。
- z / TVT range が大きい。strict 7 の TVT range 中央値は全体よりかなり高く、previous readout では約 96 percentile 相当。
- formation slope 自体は決定的ではない。
- GR 欠損は一般原因ではない。中央値では全体より低めで、例外的に `86454a6f` が GR 欠損 81%。
- GR trend change はやや高いが、PF bad の主因は GR 欠損よりも tail / TVT range / PF branch offset。

解釈:

- PF/likPF が長い tail や branch prior で大きく上下に外す一方、HMM は self-GR / typewell alignment で正しい branch に戻れている。
- ただし HMM 直接置換は危険。HMM が効く well は存在するが、別方向の worst-well regression も大きい。

改善方針:

- HMM を direct replacement ではなく、PF/HMM disagreement、HMM std、self-GR valid rate、GR 欠損率、TVT range を confidence feature として使う。
- PF が whole-well offset している疑いが強いときだけ、HMM 寄り candidate を弱く混ぜる gate を検討する。

## 2. HMM bad / PF good

定義:

- HMM RMSE `>= 30`
- `likPF_mean RMSE <= 10`

該当 well は 7 本。pure PF `pf_ancc <= 10` まで含めると `8a3da6d1` も追加候補。

| well | HMM RMSE | likPF RMSE | pure PF RMSE | HMM bias | GR 欠損率 | HMM std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `b19b0395` | 54.6 | 7.7 | 6.4 | +47.0 | 3.8% | 3.84 |
| `97cd5bf9` | 40.6 | 4.6 | 3.1 | -31.5 | 63.3% | 10.76 |
| `fa31da94` | 37.8 | 3.6 | 3.1 | +33.7 | 69.5% | 8.26 |
| `c9578d27` | 39.4 | 5.3 | 3.0 | -31.7 | 57.4% | 4.87 |
| `a9c9b150` | 35.5 | 5.0 | 1.6 | -32.9 | 54.7% | 10.97 |
| `a6b8ac67` | 32.0 | 4.2 | 7.8 | -29.2 | 7.4% | 2.63 |
| `729665c0` | 31.0 | 3.5 | 3.4 | +28.6 | 55.5% | 2.99 |

集約特徴。

- HMM は well 全体で ±29-47 ft の branch miss / vertical offset。
- PF/likPF の bias は概ね小さい。
- XY は一箇所に固まらない。やや西寄りもあるが、空間 hot spot ではない。
- nearest 8 wells に同じ strict 傾向は 0。typewell group 内でも group 全体は壊れていない。
- z 深さ、z range、TVT range は普通。PF bad / HMM good と違い、深い tail / TVT range 大は主因ではない。
- formation slope はやや高い。strict 7 の formation absolute slope は全体 78 percentile 付近。
- GR 欠損は強く関係する。strict 7 の評価区間 GR 欠損率中央値は 55.5%、全体中央値は約 30.3%。
- ただし `b19b0395` と `a6b8ac67` は GR 欠損が低く、欠損だけでは説明できない。これらは GR があるが HMM が misleading match に吸い込まれた可能性が高い。
- self-GR valid rate / quality が低い。strict 7 の self-GR valid rate 中央値は約 0.089、全体中央値は約 0.614。
- HMM std が高い。strict 7 の HMM std 中央値は全体 86 percentile 付近。

解釈:

- HMM は GR/self-GR 証拠が弱い、または misleading なときに typewell branch を大きく取り違える。
- PF は物理 trajectory / likelihood 側で安定し、HMM の branch miss を避けている。
- simple HMM-likPF blend では防ぎきれない。exp209 fixed blend でもこの群では RMSE 15-31 程度残る。

改善方針:

- HMM std 高、self-GR valid rate 低、GR 欠損率高、HMM-PF offset 大の条件では HMM 重みを強く下げる。
- HMM candidate は add-only confidence feature にし、direct replacement や単純 blend は避ける。

## 3. HMM/PF と exp226 の対比

定義:

- GR 系 bad: HMM / `likPF_mean` / `pf_ancc` のいずれかが RMSE `>= 30`
- exp226 good: exp226 RMSE `<= 10`
- 逆: exp226 RMSE `>= 30` かつ HMM / PF のいずれかが RMSE `<= 10`

strict count。

| 条件 | 本数 |
| --- | ---: |
| HMM bad / exp226 good | 9 |
| likPF bad / exp226 good | 4 |
| pure PF `pf_ancc` bad / exp226 good | 19 |
| HMM or PF bad / exp226 good | 28 |
| HMM and PF both bad / exp226 good | 0 |
| exp226 bad / HMM or PF good | 3 |

### 3.1 HMM bad / exp226 good

該当 well。

`b19b0395`, `8a3da6d1`, `4caa7289`, `97cd5bf9`, `fa31da94`, `4c2208f5`, `fc0d20b2`, `a9c9b150`, `729665c0`

特徴。

- GR 欠損率が高い。中央値は約 54.7%。
- HMM std が非常に高い。中央値は全体 95 percentile 付近。
- self-GR valid rate が低い。
- GR trend change / quarter jump も高め。
- z range / TVT range は普通。
- exp226 donor 距離は極端ではない。

解釈:

- GR/HMM branch lock が壊れたが、z/geometry donor field は普通に成立したケース。
- HMM には warning signal が多い。HMM std、self-GR valid rate、GR 欠損率で検知しやすい。

### 3.2 likPF bad / exp226 good

該当 well。

`7987f2f2`, `3417285d`, `57f05c51`, `81bf5923`

特徴。

- HMM bad / exp226 good とは違い、GR 欠損は低め。
- self-GR valid rate は高い。
- likPF bias が ±25-45 ft 級。
- XY は東寄りが目立つ。
- TVT range が高め。
- `delta_abs_median` が上限 4.0 に張り付きがち。

解釈:

- GR が欠けているのではなく、likPF の candidate branch / prior が丸ごと外れている。
- exp226 の z/geometry smoothing が branch miss を避けている。

### 3.3 pure PF bad / exp226 good

該当 19 本。代表:

`4caa7289`, `7e721392`, `2fd68f7b`, `3932faa6`, `3417285d`, `c1d046f4`, `43e16325`, `f4d12d23`, `367456ce`, `57f05c51`, `81bf5923`, `a783cc24`, `fbd68d27`

特徴。

- TVT range が高め。
- `likpf_abs_delta_mean` が高い。
- GR 欠損は決定打ではない。
- PF/ANCC が長い tail や候補 branch のズレに弱い。

解釈:

- pure PF/ANCC は oracle 的には有用な well があるが、単体 candidate としては branch drift に弱い。
- exp226 の z/geometry smoothing が PF branch drift を吸収できるケースがある。

### 3.4 exp226 bad / HMM or PF good

該当 well は 3 本。

| well | exp226 RMSE | HMM RMSE | likPF RMSE | pure PF RMSE | exp226 bias | donor min / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `389ae58f` | 52.3 | 6.7 | 4.8 | 18.9 | -50.0 | 1419 / 2474 |
| `70925e23` | 33.0 | 13.2 | 6.6 | 23.2 | +28.0 | 2298 / 6928 |
| `ae8959c3` | 30.0 | 2.6 | 8.0 | 19.2 | -27.7 | 3048 / 4989 |

特徴。

- exp226 の失敗は whole-well bias。
- donor 距離が非常に大きい。donor min は全体 96 percentile、donor max は 98 percentile 付近。
- GR 欠損率は低い。中央値は約 11.8%、全体中央値は約 30.3%。
- self-GR valid rate は高い。中央値は約 0.98。
- GR std / half mean diff は高めで、GR pattern 自体は強い。
- z/TVT range はやや大きい。
- nearest 8 wells に同じ strict 逆傾向は 0。

解釈:

- z/geometry donor が遠く、exp226 が類似 donor field を見つけられずに全体 offset。
- GR signal は欠けておらず、HMM / likPF は GR pattern で補正できた。
- これは GR 主体と z/geometry 主体の差が最も明確に出た群。

## 観点別まとめ

### XY 座標

- PF/HMM の片側 failure も exp226 との対比も、単一の XY hot spot ではない。
- `likPF bad / exp226 good` は東寄りが目立つ。
- `exp226 bad / GR 系 good` は donor 距離が大きく、特に `70925e23` は近傍が遠い。

### 周りの well

- strict pattern は近傍にあまり伝播しない。
- HMM bad / exp226 good は nearest 8 の同傾向 0。
- exp226 bad / GR 系 good も nearest 8 の同傾向 0。
- 同じ typewell group に属する近傍が多い場合でも、group 全体が壊れているわけではない。

### z の深さ / range

- PF bad / HMM good は tail / TVT range が大きい。
- HMM bad / PF good は z / TVT range が普通。
- exp226 bad / GR 系 good は z/TVT range がやや大きく、donor 距離の悪さと合わさって geometry failure になりやすい。

### 地層の傾き

- HMM bad / PF good では formation slope がやや高い。
- exp226 bad / GR 系 good では marker slope の max / std が極端に高い。geometry interpolation が不安定になる候補 signal。
- PF bad / HMM good では formation slope は決定打ではない。

### GR 欠損

- HMM bad / PF good、HMM bad / exp226 good では重要。
- PF bad / HMM good では一般原因ではない。
- exp226 bad / GR 系 good ではむしろ GR 欠損が少なく、GR 主体の HMM/likPF が勝つ理由になる。

### GR trend change

- HMM bad / exp226 good では GR trend change / quarter jump が高め。
- exp226 bad / GR 系 good では GR variation は強いが欠損が少なく、GR 主体の手法に有効な信号として働いた可能性がある。
- PF bad / HMM good では GR trend change は補助的で、主因は PF branch / tail offset。

## 改善方針

直接 blend / replacement より、confidence / selector feature として使うのが安全。

優先 feature。

- `hmm_std_mean`, `hmm_std_p90`
- `self_gr_valid_rate`, `self_gr_quality_mean`
- 評価区間 GR 欠損率、longest NaN、GR quarter jump、GR half mean diff
- `likpf_abs_delta_mean`
- PF/HMM/exp226 prediction disagreement
- exp226 `donor_dist_min`, `donor_dist_max`
- z range、TVT range、tail length
- formation marker slope max / std

想定 gate。

- HMM を下げる: HMM std 高、self-GR valid rate 低、GR 欠損率高、HMM-PF offset 大。
- PF を下げる: likPF / pf_ancc bias 方向が大きく、TVT range 大、`likpf_abs_delta_mean` 高、PF/HMM/exp226 のうち exp226 が安定。
- exp226 を下げる: donor 距離が大きい、formation marker slope max/std が極端、GR 欠損が少なく self-GR valid rate が高い、HMM/likPF が低 std で一致。

次に実験化するなら、`hmm_pf_exp226_disagreement_confidence_features_on_exp218` のような add-only feature 実験が妥当。

やらない方がよいこと。

- HMM 直接置換。
- PF / HMM / exp226 の global 固定重み blend。
- exp226 residual の直接補正。exp228 で direct residual correction は exp218 anchor に届かなかった。

## 正式 study artifact

2026-07-12 に同じ集計を `studies/` 配下へ再生成し、正式な diagnostic study artifact として保存した。

- script: `studies/hmm_pf_exp226_well_pattern_readout.py`
- output dir: `studies/hmm_pf_exp226_well_pattern_readout_20260712/`
- joined table: `studies/hmm_pf_exp226_well_pattern_readout_20260712/joined_well_summary.csv`
- category wells: `studies/hmm_pf_exp226_well_pattern_readout_20260712/category_wells.csv`
- category summary: `studies/hmm_pf_exp226_well_pattern_readout_20260712/category_summary.csv`
- feature summary: `studies/hmm_pf_exp226_well_pattern_readout_20260712/feature_summary.csv`
- typewell context: `studies/hmm_pf_exp226_well_pattern_readout_20260712/typewell_context.csv`
- source manifest: `studies/hmm_pf_exp226_well_pattern_readout_20260712/source_manifest.json`

元の `/tmp/exp226_hmm_pf_contrast_joined.csv` は作業用の一時ファイルとして扱い、今後は上記 study artifact を参照する。
