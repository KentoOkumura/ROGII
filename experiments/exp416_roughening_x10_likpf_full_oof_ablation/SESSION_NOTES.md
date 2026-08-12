# exp416_roughening_x10_likpf_full_oof_ablation セッションノート

## 目的

exp072 likelihood-PFのresampling rougheningだけを10倍にし、exp410 sentinelでの
改善が全773 train wellsへ一般化するかを単一variantで判定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle version 2 `COMPLETE` / scientific・technical gate FAIL / terminal close
- 親: exp072
- 原因根拠: exp410
- CV: candidate `13.617717557749454` / control `11.594894395642696`
- LB: なし
- model / inference / submission: なし
- ユーザー承認範囲: 正規train Notebook採用、Kaggle package、CPU 4 shard
  push / run、strict merge、train-side評価まで
- probe / inference / submission: 未承認

## 固定した変更

- rough position: `0.10 -> 1.00 ft`
- rough rate: `0.001 -> 0.010`
- その他のPF parameter、500 particles、128 seeds、stable seed、算術seed平均は不変
- exp072 cacheの`last_known_tvt + likpf_mean_d`をcontrolへ復元し、exp209 exact
  reconstructed controlでrow-level parityを確認する。control PFは再実行しない

## 実行量（計画 = 実績）

- active scientific variants: 1
- candidate PF well-runs: 773
- control PF well-runs: 0
- seed-well trajectories: 98,944
- particle starts: 49,472,000
- reporting folds: 5
- LightGBM configs / trained folds / boosters: `0 / 0 / 0`
- HMM / Beam / GPU: `0 / 0 / 0`
- Kaggle CPU shards: 4

2026-07-27のユーザー依頼`実行してください`により、この固定量でのKaggle CPU実行が
承認された。control再実行、GPU、probe、inference、submissionは含めない。

## 実装内容

- `exp416_roughening_x10_likpf_full_oof_ablation_compact_selfcontained_train.py`
  をJupytext percent形式で作成
- exp072 kernelのRNG call順、GR scale、missing補間、stable seed、算術seed平均を維持
- control / candidate parameter setのdiffが`rough_position`と`rough_rate`だけで、
  どちらも比率10.0であることをfail closedで確認
- 全773 wellsをsuffix行数のdeterministic LPTで4 shardへ分割
- 各shardはtruth-free candidateをraw / decompressed / schema / logical SHA付きでfreeze
- 4 shard strict merge後だけsuffix truth、saved exp072、exp209 parity control、
  exp226 fold、exp115 hidden-like role、exp410固定16 episodesを読む
- pooled / 5 folds / raw observed / raw missing / 1000+ / hidden-like 2面 /
  by-well p95 / worst well / persistent episode SSEのAND gateを実装
- fixed probe well rerun parityは別runとして実装し、未実行ならdeterministic anchorを禁止
- 同一exp helper importは使わず、Notebook sourceに`__file__`を残していない
- 正規train / inference Notebookは上書きしていない

## 再現性メモ

- seed:
  `sha256("likpf::train::<well_id>") % 2147483647 + 1 + seed_index`
- well内Numba single worker、shard順非依存
- RNG call順はexp072と同じで、roughening振幅だけ変更
- raw / decompressed / schema / code / config / prediction logical SHAを保存済み
- scientific / technical gate FAILのためdeterministic anchorとは呼ばない

## コマンドログ

2026-07-27:

- `make new-steering EXP=exp416_roughening_x10_likpf_full_oof_ablation`
- `make new-exp EXP=exp416_roughening_x10_likpf_full_oof_ablation SOURCE=templates/experiment`
- scaffoldとdesign-only文書を作成
- PF / Notebook / Kaggle実行は0
- ユーザーの`exp416を実装してください`を実装承認として記録
- compact self-contained train候補と
  `experiments/exp416_roughening_x10_likpf_full_oof_ablation/tests/test_exp416_roughening_x10_likpf_full_oof_ablation.py`を作成
- `py_compile`と`ruff --select F821,E9`を通過
- Jupytext変換と`jupytext --to ipynb --test`を通過
- `experiments/exp416_roughening_x10_likpf_full_oof_ablation/tests/test_exp416_roughening_x10_likpf_full_oof_ablation.py`、
  `tests/test_kaggle_notebooks.py`、`tests/test_scaffold.py`の計22 testsを通過
- `task validate-exp`はローカルに`task`コマンドがなく終了コード127だったため、
  同等の`make validate-exp EXP=exp416_roughening_x10_likpf_full_oof_ablation`を実行し
  strict validation通過
- 親compact比較: exp400は2,016行 / 11章、exp416候補は2,319行 / 12章を持ち、
  runtime/config、input、PF kernel、shard、merge、late join、gate、生成物をNotebook上で追える
- implementation SHA:
  - train source:
    `147417db05d2052b26c37a163f74bcc1e27444938f5a955c271571a26aca0fbb`
  - compact train Notebook:
    `4e023fe1463e6efe2c275a0f952410e070314325acc64bf3b8a31a4dab1f1d11`
  - config:
    `ad5b9327706eea6fde0a742ea4fe7226016b08126353cacd2c54328452da6391`
  - contract tests:
    `fffd130e9f174f0bf3c20d2fbf0df8f4ddb658bd572a0b2d1d83e8b28f9fc299`
- PF / Kaggle package / push / run / inference / submissionは0
- ユーザーの`実行してください`を、正規train Notebook採用、Kaggle package、
  CPU 4 shard push / run、strict merge、train-side評価の承認として記録
- compact self-contained sourceを正規train Notebookへ採用
- shard wrapper 4本とstrict merge wrapper 1本をJupytext形式で作成
- 4 shard package:
  - `kentookumura/exp416-rough-x10-shard-0`
  - `kentookumura/exp416-rough-x10-shard-1`
  - `kentookumura/exp416-rough-x10-shard-2`
  - `kentookumura/exp416-rough-x10-shard-3`
- merge package: `kentookumura/exp416-rough-x10-merge`
- 全packageはprivate / CPU / internet off / run-on-push。shardはexp072 / exp209 /
  exp226をinputとし、mergeはそれらに4 shardを追加する
- package前にactive variant 1、candidate 773 wells、control rerun 0、
  98,944 trajectories、49,472,000 particle starts、reporting folds 5、
  LightGBM configs / trained folds / boosters `0 / 0 / 0`、HMM / Beam / GPU
  `0 / 0 / 0`を再確認
- 22 targeted tests、`py_compile`、Ruff F821/E9、Jupytext test、
  strict `make validate-exp`を再通過
- embedded source SHAは
  `147417db05d2052b26c37a163f74bcc1e27444938f5a955c271571a26aca0fbb`。
  exp115 / exp410 3 assetsのembedded SHAはconfig期待値と一致
- この時点でKaggle push / PF run / merge / inference / submissionは0
- 最終package config SHA:
  `ec8df79a503b60972444d59f470148d12547271be0312f3bb199fb013d8862fc`
- 正規train Notebook SHA:
  `35ff5bb5a04b511aa91f7ccdb645cd014ee953935b7329a2d2323ebea0ad42d2`
- 初回4本同時push:
  - shard 1 / 2: version 1 push成功、run-on-pushで開始
  - shard 0 / 3: Kaggle `Maximum batch CPU session count of 5 reached`で未作成
- push後の`kaggle kernels pull -m`でshard 1 / 2の存在を確認。shard 0 / 3は
  `GetKernel` 500で、初回pushの上限拒否と整合するため再実行待ちとした
- 科学条件やpackageを変えず、先行run完了でCPU枠が空き次第、未作成slugだけをpushする
- ユーザー指示`監視は止めていいです。完了したら連絡します。`により、自動監視を停止
- 停止時点:
  - shard 1 / 2: version 1、`RUNNING`
  - shard 0 / 3: 未作成・未実行
  - strict merge: 未push・未実行
- Kaggle上のshard 1 / 2実行は停止していない。ユーザーから完了連絡を受けた後、
  同じpackageでshard 0 / 3をpushし、4本完了後にmergeを開始する

2026-07-28:

- ユーザーから先行runの完了連絡を受領
- shard 1 / 2 version 1を`COMPLETE`と確認
- shard 1:
  - 193 wells / 946,017 rows / 24,704 trajectories / 12,352,000 particle starts
  - elapsed 9,066.877秒、peak RSS 0.526 GB
  - prediction logical SHA
    `6ba78348db71a2d9f83403df7015ad4b0664a29368ad025af6971c9f284b043a`
  - decompressed prediction SHA
    `05c84539b2004f67028737afaa3b293fe443c2dc4c48b46b1b91234acab43a6f`
- shard 2:
  - 193 wells / 946,112 rows / 24,704 trajectories / 12,352,000 particle starts
  - elapsed 8,077.341秒、peak RSS 0.523 GB
  - prediction logical SHA
    `37fe39b4a9c3972018d554326b3db89745aa6b97aa6a504aa9a15e387743b355`
  - decompressed prediction SHA
    `9fe9707d6369d57bac042231bdf5795ed583e0b13652a05233a86f130a42eafe`
- 両shardともstatus `complete`、raw well identity、scientific contract、
  source SHA、config SHAが一致。control rerun / LightGBM / HMM / Beam / GPUは0
- shard 0 / 3 canonical slugは、初回CPU上限拒否後のghost stateにより、
  pull 500・list refなし・再push `Notebook not found`が再現
- 過去のexp243 / exp402と同じ限定recoveryとして、科学code・config・入力・
  wrapper・実行量を変えずslug/titleだけを変更
  - shard 0: `kentookumura/exp416-rough-x10-shard-0-v1` version 1、
    id_no `128831877`
  - shard 3: `kentookumura/exp416-rough-x10-shard-3-v1` version 1、
    id_no `128831879`
- 両recovery shardは`RUNNING`。pullしたNotebookでprivate / CPU /
  internet off、config SHA
  `ec8df79a503b60972444d59f470148d12547271be0312f3bb199fb013d8862fc`、
  source SHA
  `147417db05d2052b26c37a163f74bcc1e27444938f5a955c271571a26aca0fbb`
  を確認
- 前回のユーザー指示に従い、shard 0 / 3の継続監視は行わない。完了連絡後に
  4 shardを検証し、merge packageのinputをrecovery slugへ限定更新して実行する

- ユーザーからrecovery shardの完了連絡を受領
- shard 0 / 3 version 1を`COMPLETE`と確認
- shard 0:
  - 193 wells / 946,128 rows / 24,704 trajectories / 12,352,000 particle starts
  - elapsed 5,425.840秒、peak RSS 0.516 GB
  - prediction logical SHA
    `13722e8c32ac9dbca5ea7e656c64c02804e830eeb5186d9edd6bb262d8a2181b`
  - decompressed prediction SHA
    `5ab6e0e4e7d37f62b6f58a8682375b61789268a0743fe67db54f7e66de44009c`
- shard 3:
  - 194 wells / 945,732 rows / 24,832 trajectories / 12,416,000 particle starts
  - elapsed 5,452.695秒、peak RSS 0.521 GB
  - prediction logical SHA
    `120c4cb871355267e2072e5494c903d9ccf6072a7d3e4fb420e7a6675c3279a8`
  - decompressed prediction SHA
    `63e7653e4fea6a94bc4868d7ae654b0acdc1f00a9d7de3cfc4b3c4b50a12fa60`
- 4 shard合計:
  773 wells / 3,783,989 rows / 98,944 trajectories /
  49,472,000 particle starts。control rerun / LightGBM / HMM / Beam / GPUは0
- 4 shardすべてscientific contract SHA
  `9c9bdaa93f0e64aa2ea54a46ae8fbb2a4f1f4f05a34b0d98e734e2b3c8ac398a`、
  source SHA
  `147417db05d2052b26c37a163f74bcc1e27444938f5a955c271571a26aca0fbb`、
  shard package config SHA
  `ec8df79a503b60972444d59f470148d12547271be0312f3bb199fb013d8862fc`
  で一致
- merge用configの入力identityだけを、実在するshard 0 / 3 recovery slugへ更新。
  scientific parameter、count、gate、codeは変更しない
- merge packageをstrict生成し、22 targeted testsとstrict
  `make validate-exp`を再通過
- merge package:
  - kernel: `kentookumura/exp416-rough-x10-merge`
  - version: 1
  - id_no: `128912230`
  - status: `RUNNING`
  - private / CPU / internet off / run-on-push
  - inputs: exp072 / exp209 / exp226、shard 1 / 2、recovery shard 0 / 3
  - merge config SHA:
    `0620e4d160c5e4737498cf7867bc516f2a3bbb8b99c9aee289211a4a4d27a309`
  - source SHA:
    `147417db05d2052b26c37a163f74bcc1e27444938f5a955c271571a26aca0fbb`
- push後pullで上記metadataとembedded SHAを確認
- ユーザーの継続監視停止方針に従い、起動確認後のpollingは行わない。
  完了連絡後にlogs / metrics / gate / SHAを取得して最終記録する

- ユーザーからmerge version 1の失敗連絡を受領
- kernel statusは`ERROR`。logsではbootstrapと入力previewは成功し、約102秒後に
  `preflight_late_inputs`で
  `exp209_reconstructed_control missing required columns:
  ['likpf_mean_exp209_reconstructed']`として停止
- SHA固定済みexp209 cacheの実列は
  `hmm_mean_tvt`、`hmm_minus_likpf_mean`を含む一方、
  `likpf_mean_exp209_reconstructed`は含まない
- exp410で確立済みのexact reconstruction
  `float32(hmm_mean_tvt - hmm_minus_likpf_mean)`をmerge adapterへ反映する
- 失敗はshard merge・truth attachment・評価・gateより前の入力schema検証で発生。
  4 shardの予測、scientific contract、実行量には影響せず、PF再実行は不要
- 修正範囲はexp209 controlの列allowlistと復元処理のみ。roughening条件、
  candidate予測、評価gate、入力SHA、inference/submission許可は変更しない
- exp209実cacheの先頭8行で復元列、float32 dtype、finiteを確認
- scientific contract SHA
  `9c9bdaa93f0e64aa2ea54a46ae8fbb2a4f1f4f05a34b0d98e734e2b3c8ac398a`
  をテストで固定し、shardとの不変性を確認
- 23 targeted tests、`py_compile`、Ruff F821/E9、compact / aggregateの
  Jupytext round-trip、strict `make validate-exp`を通過
- merge version 2 package:
  - config SHA:
    `0b115a2b24b25eb91ac1e9859bf80decba5df23382012c4ef55cd0749a1ee9d1`
  - source SHA:
    `ebb63e89f76463f89beb8fab89db0c568e0659770014102e69efe5fb9e85c8ab`
  - package notebook SHA:
    `cc82247bc6b3be5ee4c8801a327ef8742ff06c055ffc23ebc51f3cc021872555`
  - private / CPU / internet off / run-on-push、同じ7 kernel inputs
- push前pullで既存kernel id_no `128912230`を確認し、同じkernelへversion 2をpush
- push直後statusは`RUNNING`。pullしたKaggle側Notebookのembedded config/source SHA、
  id_no、7 inputs、private / CPU / internet offを再確認
- ユーザー方針に従い継続monitoringは停止。4 PF shardは再実行せず、
  probe / inference / submissionも実行していない

- ユーザーからmerge version 2の完了連絡を受領
- `kaggle kernels status kentookumura/exp416-rough-x10-merge`で`COMPLETE`、
  `kaggle kernels logs`でversion 2の全metrics / gate / summaryを確認
- primary:
  - candidate RMSE `13.617717557749454`
  - saved exp072 control RMSE `11.594894395642696`
  - improvement `-2.022823162106759 ft`、すなわちcandidateが悪化
  - within-10ft `0.772793 -> 0.695594`
- fold別candidate-minus-control RMSEは
  `+2.475088 / +1.847617 / +2.625954 / +0.583911 / +2.412363 ft`で
  5/5 folds悪化
- scope別candidate-minus-control RMSE:
  - raw GR observed `+1.778505 ft`
  - raw GR missing `+2.541544 ft`
  - MD since 1000+ `+2.199661 ft`
  - hidden-like spatial `+2.139189 ft`
  - hidden-like typewell-purged `+2.128136 ft`
- well-tail:
  - by-well delta p95 `+14.104741606946275 ft`
  - worst-well regression `+41.05036062005086 ft`
- exp410固定16 persistent-offset episodesだけはSSE
  `113224053.55777258 -> 85257299.90140347`、
  reduction `0.24700364257934881`でPASS
- scientific AND gateはpersistent episode以外の主要条件を満たさずFAIL
- technical gate:
  - exp209 reconstructed control row parity max abs
    `0.000471875000584987 ft`が事前上限`0.00001 ft`を超えてFAIL
  - probe未実行はtechnical gate合否条件ではない
  - saved exp072 RMSE parity差`3.2765750077601297e-06 ft`、3,783,989 rows /
    773 wells / folds 0--4、finite 1.0、fallbackなし、truth-freeze、実行量、
    shard runtime / memoryはPASS
- final decision:
  `roughening_x10_rejected_close_without_rescue`
- artifact evidence:
  - scientific contract SHA
    `9c9bdaa93f0e64aa2ea54a46ae8fbb2a4f1f4f05a34b0d98e734e2b3c8ac398a`
  - merged logical prediction SHA
    `6088e3cc3c94f45aa475094e4df51b3dd9247f04acabb6e3921b89fbffbe693c`
  - raw gzip SHA
    `4a956c5625659305bcb3b3d0c51e6bee0fed9a52447397f87bc2b32f4f9deafe`
  - decompressed prediction SHA
    `83247dad59d9d0d1a4705f3b4ad118cf33253347f00e8da4166374ac1f931add`
  - artifact manifest SHA
    `708bb257e3ab360f09821823d5413fa9e1c5c32ef4ddb4917b41573943dffb86`
- Kaggle logsにCV、fold/scope、gate、SHA、生成物名が揃っているため、
  repository方針どおりoutput archive全体はdownloadしていない
- roughening倍率、position/rate別、process noise、ESS、GR sigma、seed/particle、
  well/row gate、same-OOF rescueは探索しない。probe / inference / submissionも行わない

## 次のアクション

exp416をterminal closeする。原因分解が必要な場合だけ、保存済みprediction /
well audit / by-well metricsを使う0-PF readoutを別steering・別承認で検討する。
