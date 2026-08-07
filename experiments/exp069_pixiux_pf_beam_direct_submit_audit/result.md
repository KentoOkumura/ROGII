# exp069_pixiux_pf_beam_direct_submit_audit 結果

## 仮説

Pixiux likelihood-PF / PF-Beam direct prediction を LightGBM booster に通さず提出すると、exp063 `lgb_mean` Public LB 8.811 より良い可能性がある。特に exp063 と exp027 の差 +0.030 が、LightGBM residual replay による過補正か、PF/Beam direct signal のほうが public evaluation に合っているためかを切り分ける。

## 設定

- 親: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- 検証: 直接提出監査。train-side CV は作らない。
- メトリック: RMSE / Public LB
- シード: 42。2026-06-15 patch 後は well id 由来の stable seed を PF 系に明示注入し、`n_jobs=1` で実行する。
- 提出候補: `likpf_mean`
- 除外: LightGBM booster prediction、CatBoost、Ridge stack、final public blend、projection postprocess、static visible override、hidden-specific branch

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | 9.721 (`ref=53706005`) |
| Private LB | - |
| Kaggle inference | v3 deterministic complete |
| Submit-check | PASS |
| Selected candidate | `likpf_mean` |
| Submission SHA256 | `57d5c55c5caa1d07b6691a054116b434d63dd9f8e03c73dfb6ef45753aa8fa01` |
| Prediction range | 11601.626953 - 12241.427734 |
| Fallback rows | 0 |
| Diff RMSE vs exp027 | 12.852195 |
| Diff RMSE vs exp063 | 8.700711 |

## 解釈

Kaggle inference v2 / v3 は CPU metadata で完了した。GPU は不要だった。

`likpf_mean` direct submission は exp027 / exp063 と高相関だが全行に近い差分があり、Public LB を見る価値がある別候補になっている。deterministic v3 output では exp063 との差分 RMSE 8.700711、exp027 との差分 RMSE 12.852195 なので、同一提出の再生成ではない。

pre-patch v2 の提出結果は Public LB 9.877 (`ref=53637978`) で悪かった。deterministic patch 後の v3 code submission は Public LB 9.721 (`ref=53706005`) で、exp063 v2/best `lgb_mean` 8.811 より `+0.910`、PF route anchor の exp027 8.781 より `+0.940` 悪い。近接する duplicate submission `ref=53705994` も Public LB 9.721 だった。

2026-06-15 に再現性担保のため deterministic patch を入れた。`n_jobs=1`、well id 由来 stable seed、`_pf_ancc` / `_pf_z` / `lik_pf` の明示 seed 化、JIT warm-up の固定配列化により、code-submit の hidden rerun でも同じ乱数経路を通す設計にした。ローカルの構文、lint、experiment validation、Kaggle package 再生成は PASS。

deterministic patch 後の Kaggle inference v3 は完了した。ログ上で `Test dir: .../test wells=3`、`building Pixiux/public base features from raw test files...`、`building Pixiux likelihood-PF replay features for test...`、`deterministic=true`、`n_jobs=1`、`test_rows=14151`、`test_likpf_rows=14151` を確認した。したがって PF/Beam と likelihood-PF は Kaggle inference 時に raw test files から再生成されており、exp063 output を予測値として読んでいない。`reference_submission_paths` は比較 artifact 用で、予測値生成には使っていない。

Public LB 9.877 は patch 前の v2 code submission に対応する。deterministic v3 output は Public LB 9.721 で、v2 よりは改善したが exp027 / exp063 v2/best には届かない。`ref=53710264` / `ref=53710105` の Public LB 8.766 は exp069 への紐づけ誤りとして扱い、exp069 の結果には採用しない。

## 次

`likpf_mean` direct PF/Beam は deterministic 化しても Public LB 9.721 で悪化したため採用しない。今後は直接提出ではなく、PF/Beam disagreement、confidence、error map の診断値として使う。
