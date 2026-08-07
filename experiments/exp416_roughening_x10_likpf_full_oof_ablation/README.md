# exp416_roughening_x10_likpf_full_oof_ablation

## 状態

実装・正規Notebook採用・Kaggle CPU 4 shard・strict merge version 2が完了。
roughening x10はoverall RMSEを`11.594894 -> 13.617718`へ`2.022823 ft`悪化させ、
scientific / technical gate FAILで棄却してbranchを閉じた。推論・提出は未実行。

## 仮説

exp410のtarget-late sentinelで見えたroughening 10倍の改善が全773 wellsにも
一般化し、exp072 likelihood-PFの正解particle basinを維持・再捕捉しやすくする。

## 検証方針

- Route: `pf_beam`
- 親: `exp072_exp063_full_replay_feature_cache`
- candidate: resampling roughening position / rateを両方10倍
- control: 保存済みexp072 `likpf_mean`。control PF再実行0
- candidate: 1 variant ×773 wells、500 particles ×128 seeds
- 初回実行先: Kaggle CPU 4 shard
- primary: pooled RMSE gain `>=0.05 ft`、4/5 folds
- raw/missing/1000+/hidden-like/by-well/persistent-offsetをAND gateにする

## 実装

- exact exp072 kernelとstable well seedを維持
- control / candidate parameter setの機械的diffでroughening 2値以外を拒否
- suffix行数のdeterministic LPTで4 shardへ分割
- shard predictionをtruth-freeでfreezeし、strict merge後だけtruthとreporting roleを結合
- exp410の固定12 wells / 16 episodes / 55,104 rowsをpersistent SSE guardに使用
- saved exp072 controlとexp209 reconstructed controlのrow parityを確認
- fixed probe wellのrerun parityを記録するまでdeterministic anchorと呼ばない

## 所見

全OOFでは5/5 folds、raw/missing、1000+、hidden-like 2面がすべて悪化し、
by-well p95は`+14.104742 ft`、worstは`+41.050361 ft`だった。一方、固定16
persistent-offset episodesのSSEは`24.700364%`改善した。局所回復は再現したが、
global roughening増加としては一般化せずprediction候補にしない。

詳細はsteeringの
`design.md`と`requirements.md`を正とする。

## 生成物

- `exp416_roughening_x10_likpf_full_oof_ablation_compact_selfcontained_train.py`
- `exp416_roughening_x10_likpf_full_oof_ablation_compact_selfcontained_train.ipynb`
- `tests/test_exp416_roughening_x10_likpf_full_oof_ablation.py`
- `config.yaml`、`metrics.json`、`SESSION_NOTES.md`

4 shard prediction / audit、merged prediction、fold/scope/by-well/episode metrics、
scientific gate、scientific contract、artifact manifestをKaggle outputへ保存した。
主要SHAと結果は`metrics.json`と`SESSION_NOTES.md`に記録した。

## 次

roughening倍率や関連parameterのsame-OOF救済を行わず終了する。必要なら保存生成物だけの
0-PF failure-regime readoutを別実験・別承認で検討する。probe / inference /
submissionは行わない。
