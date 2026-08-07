# exp231_same_typewell_horizontal_gr_atlas_gated_hmm_emission

## 状態

Kaggle CPU train-side を完了し、不採用です。Route は `pf_beam`、主親は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation` です。raw-test inference と submission は行いません。

## 仮説

同じ native-overlap typewell group の training-fold horizontal wells から作る局所 GR patch atlas は、base exact HMM が曖昧・innovation大・GR change point のときだけ、候補TVT stateの尤度を弱く識別できる可能性があります。

## 検証方針

- fixed seed 42 の 5 well folds。validation well と同fold validation wells は peer atlas source から完全除外。
- `(group, TVT bin)` ごとに `64/128/256` rows GR patch distribution を作り、16点へ縮約して比較。
- `final_loglik = base_loglik + alpha * confidence * centered_peer_score`。v3ではruntime guard下の `alpha=0.025` 1 variantを全773 wellsで評価。
- score はstate方向に中心化・clipし、support / match / uniqueness / base ambiguity / innovation / change pointだけでgateする。
- saved exp072 likPF と比較し、global RMSE、distance bucket、hidden-like、worst-well、step delta、HMM std、true-state rank、persistent-offset onset AUC/q90 liftを確認する。

## 所見

gateは平均 `0.086781` で実際に発火し、saved exp072 `likpf_mean` 比でglobal RMSEは `11.594898 → 11.569950` と小改善した。一方、`1000_plus` は `+0.016570 RMSE` 悪化し、316 wellsが悪化、最大悪化は `b19b0395` の `+48.316178 RMSE`。persistent-offset onset AUCも `0.507654` と偶然水準で、hidden-like subgroupは未評価だった。したがってatlas emissionの直接加算は不採用とし、alpha再grid、raw-test port、inference、submitには進まない。

## 次の扱い

same-typewell peer TVTをHMM emissionへ直接加える枝はclosedとする。次のPF/Beam候補は、この直接補正を使わない既存バックログから選ぶ。

## 参照ファイル

- train notebook: `exp231_same_typewell_horizontal_gr_atlas_gated_hmm_emission_train.ipynb`
- inference policy notebook: `exp231_same_typewell_horizontal_gr_atlas_gated_hmm_emission_inference.ipynb`
- HMM/atlas implementation: `exact_hmm_smoother.py`
- 結果記録: `result.md` / `SESSION_NOTES.md` / `metrics.json`
