# HMM / PF / exp226 well pattern readout inputs and outputs

このディレクトリには、既存OOFとby-well artifactをwell単位で結合した生の表とsource manifestを置きます。人間向けの結果と使用制約は、[HMM・PF・exp226のwell別失敗パターン監査](../../docs/surveys/hmm_pf_exp226_well_pattern_readout_20260712.md)を正とします。

新規学習、提出候補、anchor更新ではありません。

## 入力条件

- HMM: exp223 `hmm_selfgr_boost_only_a070_c100`
- PF primary: exp072 `likPF_mean`
- pure PF: exp072 `pf_ancc`
- exp226: train OOF by-well metrics
- 大外し閾値: RMSE >= 30
- 当たり閾値: RMSE <= 10

## 再実行

```bash
uv run python studies/hmm_pf_exp226_well_pattern_readout.py
```

## 生の出力

`joined_well_summary.csv`、`category_wells.csv`、`category_summary.csv`、`feature_summary.csv`、`typewell_context.csv`、`source_manifest.json`を保存します。条件別件数と解釈はsurveyへ集約します。
