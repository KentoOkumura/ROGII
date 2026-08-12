# 設計

## アプローチ

exp332は各windowを1件ずつexact DPへ渡し、4 windows分のgradientを蓄積してからAdamWを1回更新した。本案では、同じ学習順の連続4 windowsを1 tensor batchへまとめる。各windowのposition/rate/state幅をbatch内最大幅へpaddingし、無効cellへ`-inf` potentialとmaskを適用する。各windowのstructured NLLとlocal CEは個別に正規化し、その4件平均をbackwardするため、exp332の実効batchとoptimizer update回数を保つ。

normal posteriorとlabel-conditioned posteriorは、row方向の逐次scanを残しながらwindow batch次元をGPU並列化する。full-well real GR / circular shuffle / geometry-only decodeも、長さ順にstable sortした4 wellsを1 batchとし、controlごとに独立計算する。出力は元well/row順へ戻し、scalar exact decoderとのparityを確認する。

## 固定する科学契約

- window: 256 rows、3 scheduled slots、最大3 active non-overlap windows/well/epoch、8 epochs。
- objective: Gaussian soft-label structured NLL `1.0` + local CE `0.25`、sigma `0.35 ft`。
- boundary: interior truthはloss StateSpecのinitial position/rate priorだけに使用し、encoderにはofficial prefixだけを渡す。
- architecture/preprocessing/optimizer: exp332と同一。AdamW、lr`3e-4`、weight decay`1e-4`、AMP、gradient clip`1.0`。
- state space: step`0.35`、41 rates、rate span`0.10`、`sig_r=0.002`、`sig_p=0.02`、`mom=0.998`、fixed exp209 grammar。
- evaluation: official suffix全体をexact SSM posterior meanでdecodeし、real/shuffle/geometryと保存済みexp209/exp221を同じgateで比較する。

## Batched DP contract

- fixed batch size: 4 windows。exp332の`batch_size=1 × accumulation=4`を`batch_size=4 × accumulation=1`へ置換する。
- batch formation: exp332のtruth-free frozen window順を4件ずつ連続chunk化し、最後の不足batchだけinactive dummyをmaskする。well/error/state-cell量で順序を選び直さない。
- loss reduction: 4件のper-window valid-row normalized lossの算術平均。総valid row平均へ変えない。
- padding: position、rate、rowの全paddingをmanifestへ記録し、normal/label-conditioned partition、posterior、local CE、gradientから除外する。
- numerical parity: fixed 4 windowsでscalar/batchのloss、posterior、gradient、optimizer 1-stepを同一初期model・dropout offで比較する。
- training stochasticity: parity test後の本学習はexp332同様dropout/CUDA/AMPを含み、bitwise deterministic anchorとは扱わない。
- decode batching: stable length orderの4 wells/control。control間のunaryやposteriorを混ぜない。

## Stage 0

- exp332と同じsuffix length quartile×4件の固定16 windowsを使う。
- scalar parity用4 windowsと、batched normal/label-conditioned forward-backward、backward、optimizer、early-stop forward-only、full-well 3-control decodeを計測する。
- exp332と同じfold 0 workloadへp50とp10-throughputで外挿する。
- 必須gate: technical parity全PASS、保守的runtime`<=8.5 h`、peak`<=14 GB`、exp332 `13.151137 h`比speedup`>=1.55x`。
- Stage 0量: active variant 1、temporary neural model 1、persisted Stage A model 0、trained fold 0、LightGBM config/booster/PF/Beam/control再学習各0。
- FAIL時はbatch size、padding、compile/fused kernel、window/loss/decoder/architecture/epochを同じexpで変更せずcloseする。

## Stage A/B/C

Stage 0全PASSと別承認後だけ、exp332と同じStage A fold 0の1 neural modelを学習する。real GRがshuffle/geometryより良いだけでなく、保存済みexp209に`>=0.25 ft`勝ち、p95非悪化、worst regression`<=10 ft`を要求する。Stage A全PASS時のみ別承認でfold 1--4を追加し、pooled/full-scope/tail gateを通った場合だけ同じ実験内のStage C inference候補とする。

## 実験範囲

- 対象実験: `exp347_prefix_gr_unary_batched_window_exact_ssm`
- Route: `ensemble`
- 親実験: `exp332_prefix_gr_unary_fixed_window_structured_ssm`
- 変更する変数: exact window DPとfull-well control decodeのbatch次元並列化だけ。
- 固定する変数: exp332のdata、fold、window schedule、boundary、objective、architecture、optimizer実効batch、state grammar、controls、gate、inference方針。
- 優先順位: exp335 Stage Dなど現行P1の後。exp348より先行するP2 compute audit。

## 再現性設計

- seed policy: seed 42 + stable SHA256 window schedule/batch chunk/control decode order。
- stochastic処理: CUDA convolution、AMP、AdamW、dropout、dataloader order。parallel thread内のglobal RNGは使わない。
- PF/Beam / likelihood-PF / LightGBM: 0。
- GPU runtime: Kaggle T4、internet off、worker 0、CuDNN benchmark false、deterministic algorithmsはwarn-only。
- SHA: input/fold/window/batch/boundary/padding manifest、scalar parity report、model、posterior、prediction、package config/source/notebook、kernel versionを記録する。
- gzipはdecompressed content SHAを主証拠にする。
- deterministic anchor: false。rerunで差分を監査するまでbitwise再現性を主張しない。

## リスク

- row方向scanは逐次のため、batch並列化だけで`1.55x`に届かない可能性がある。
- paddingの広いwindowがbatch全体のcell数を増やし、理論上の並列化利益を相殺する可能性がある。
- gradient加算順、AMP、dropout RNGによりscalar exp332と学習軌跡はbitwise一致しない。
- scalar parityが通っても、full Stage AでGPU利用率やdecode paddingがmicrobenchmarkと異なる可能性がある。
- exp332のteacher boundaryによる長距離drift過小評価リスクは残る。

## Assumption

exp332 Stage 0のpeak`1.203263 GB`は、batch 4のpaddingを含めても14 GB以内へ収まる十分なmemory headroomを示す。ただしruntime改善量は未測定なので、実装前に成功を主張しない。
