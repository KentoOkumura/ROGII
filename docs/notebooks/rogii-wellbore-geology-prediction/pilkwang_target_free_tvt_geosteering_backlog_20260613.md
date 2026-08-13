# Pilkwang target-free TVT geosteering backlog proposal

作成日: 2026-06-13

対象 notebook:

- https://www.kaggle.com/code/pilkwang/rogii-target-free-tvt-geosteering

## Context

- 最新 `scoreAscending` では `pilkwang/rogii-target-free-tvt-geosteering` が先頭。
- 現在の全体 / PF route Public LB 基準は `exp027_public_replay_needless090_sel15_spread3` の 8.781。
- 現在の ML route Public LB 基準は `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit` の 8.811。
- `exp069_pixiux_pf_beam_direct_submit_audit` で Pixiux likelihood-PF direct は Public LB 9.877 まで悪化したため、PF/Beam direct replacement は採用しない。
- Pilkwang notebook の有効 profile は `projected_ridge_pf_pretrained_lgbm_modelpkg_gated`。主な構成は projected ridge/PF、pretrained LGBM late blend、model-package tiny gated correction。

## Reading

この notebook は単体の新規アルゴリズムというより、強い公開部品を trajectory-level に再構成したものとして扱う。

評価すべき差分は次の 4 つ。

1. `TVT + Z - anchor` 空間の robust polynomial projection。
2. projected ridge/PF と pretrained LGBM の 0.55 / 0.45 late blend。
3. model package を最大 1% 程度の agreement-gated correction として使う設計。
4. final contract guard と中間 submission の保存による branch decomposition。

Exact-match recovery と guarded overlap override は notebook 内にあるが、現設定では無効。したがって同一 well override を主な改善根拠として扱わない。

## Proposed backlog

| Priority | Idea | Route | Parent / inputs | Validation | Success condition | Risk |
| --- | --- | --- | --- | --- | --- | --- |
| High | `public_pilkwang_integrity_replay`: Pilkwang notebook をそのまま replay し、外部生成物と final branch を監査する | pf_beam | target notebook, `pilkwang/rogii-model-package`, `fleongg/rogii-claude-models-pub`, `ravaghi/wellbore-geology-prediction-artifacts` | Kaggle code rerun output の `submission.csv`、中間 `projected_ridge_pf_projection_submission.csv`、`pretrained LGBM_pretrained_submission.csv`、modelpkg report、contract guard を取得。exp027 / exp063 との pairwise diff、予測範囲、runtime、input metadata を記録 | code competition rerun で同一 profile が完了し、submission contract が PASS。static visible CSV や disabled branch ではないことを確認できる | 外部 dataset version 依存、model package manifest 不一致、Kaggle package source の再現性 |
| High | `projection_only_on_exp063`: exp063 Pixiux LGBM prediction に `TVT + Z - anchor` projection を後処理として移植する | ml_model | exp063 OOF / inference predictions, tracker features, raw data | 再学習なし。OOF で degree 3/4/5、blend beta 0.25/0.50/0.75、robust C を小さく比較。distance bucket、tail length bucket、near-continuity、prediction range を確認 | exp063 raw OOF 9.630105 から fold-stable に改善し、近距離 row と short tail wells を壊さない。inference diff が過大でない | same-OOF 過適合、真の急変を平滑化、hidden distribution で悪化 |
| High | `pilkwang_branch_decomposition`: Pilkwang final score の寄与を branch 別に分解する | pf_beam | replay output | final、projected ridge/PF projection、pretrained LGBM、model-package-only、gated candidates 0.003/0.005/0.010 の pairwise diff を exp027 / exp063 と比較。可能なら public score を 1-2 candidate に限定して確認 | 改善が projection / pretrained LGBM / modelpkg gate のどこから来るか特定できる | 提出回数消費。branch ごとの public score は hidden-safe とは限らない |
| High | `modelpkg_tiny_gate_on_exp063`: model-package prediction を exp063 に agreement-gated correction として足す | ml_model | exp063 base submission, `pilkwang/rogii-model-package` | package feature builder と manifest を使い、`g = gmax / (1 + (abs(pkg-base)/scale)^2)` を gmax 0.003/0.005/0.010、scale 4/5/8 で OOF surrogate と submission diff を監査 | tiny gate が exp063 / exp027 との差を詰め、p95 diff guard で大外しを抑えられる | model package の target space / feature contract 依存。OOF がない場合は public-only tuning になりやすい |
| Medium | `pretrained_lgbm_branch_port`: fleongg pretrained LGBM branch を exp063 の比較対象として単独 port する | ml_model | `fleongg/rogii-claude-models-pub`, exp063 feature generation utilities | pretrained branch 単独 submission と exp063 / exp027 pairwise diff、feature schema、model count、runtime を監査。可能なら OOF 相当を train pseudo-test に構築 | branch が exp063 と十分に decorrelated で、late blend の根拠がある | pretrained artifacts の由来と feature parity が不透明。直接 port は重い |
| Medium | `late_blend_exp063_pilkwang_projection`: exp063 と projected ridge/PF 系予測を固定重みで late blend する | ml_model / pf_beam bridge | exp063 submission, Pilkwang projected ridge/PF intermediate | 0.45/0.55/0.65 など少数の固定重みだけ比較。pairwise diff、prediction range、near rows、known visible sample behavior を記録 | exp063 8.811 と exp027 8.781 の間を安定して詰める candidate になる | public-only blend になりやすい。hidden-safe 根拠が弱い場合は submit 候補にしない |
| Medium | `projection_confidence_error_map`: projection が効く well 条件を読む | ml_model | exp063 OOF, Pilkwang projected branch if available | raw vs projected の差分を tail length、Z span、GR missing、PF/Beam disagreement、prefix length、native typewell group で error map 化 | projection を全体適用するか、flat/long/high-disagreement wells に限定すべきか判断できる | 診断に留める。hard router へ急がない |
| Low | `exact_override_negative_control`: exact-match / guarded overlap override を negative control として確認する | pf_beam | Pilkwang notebook optional layers, exp064 assert result | hidden code submission で train/test same-well overlap がない前提を維持し、optional layer が hidden で発火しない、または発火しても guard report で説明可能かを見る | same-well override を改善根拠から明確に除外できる | exp064 で no-overlap 方向の結果が出ているため優先度は低い |

## Recommended next 3 experiments

1. `public_pilkwang_integrity_replay`
   - 目的: 最新トップ notebook の submission 生成が code rerun として成立しているか、外部生成物と branch 発火を記録する。
   - 成功条件: final / intermediate files / guard summary / input metadata / pairwise diff を取得できる。

2. `projection_only_on_exp063`
   - 目的: Pilkwang 由来の最も移植しやすい新規要素である `TVT + Z - anchor` projection を、exp063 の保存済み OOF に再学習なしで試す。
   - 成功条件: original-fold と well-hash の両方で exp063 raw を上回る固定 setting が見つかる。

3. `pilkwang_branch_decomposition`
   - 目的: projection、pretrained LGBM、modelpkg gate のどれが score を押しているかを分解し、次に port すべき branch を決める。
   - 成功条件: final と各 branch の pairwise diff、予測範囲、public score candidate の優先順位が出る。

## Do not do yet

- Pilkwang final を blind blend で exp027 / exp063 に混ぜる。
- exact-match recovery / guarded overlap override を改善要素として採用する。
- PF/Beam direct prediction を再度 submit する。
- model package gate の gmax / scale を public LB だけで細かく探索する。

## KAGGLE_DIRECTION update draft

`backlog/KAGGLE_DIRECTION.md` に反映する場合は、既存の `public_artifact_replay_integrity_audit` を次のように更新する。

- `public_artifact_replay_integrity_audit` の最初の対象を `pilkwang/rogii-target-free-tvt-geosteering` に変更する。
- 新規 High backlog として `projection_only_on_exp063`、`pilkwang_branch_decomposition`、`modelpkg_tiny_gate_on_exp063` を追加する。
- `public_artifact_replay_integrity_audit` の注意点に、Pilkwang notebook の external inputs、pretrained LGBM branch、model package gate、optional exact/override disabled を明記する。
