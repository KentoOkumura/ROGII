# 設計

## アプローチ

exp394のscaled log-space exact forward-backwardを数式上同値な疎kernelへ置換する。
各H source stateから到達可能なrate 3 × position 5以下の遷移だけをその場で計算し、
`(rate, position, 5)`のdestination / log-probability配列をrowごとに生成しない。

forwardでは同じlocal transition weightsからdocking期待値を計算し、H→E確率と
H→H stay weightを確定してから疎遷移を伝播する。backwardでも同じlocal kernelを
on-the-flyで再構成する。境界ではexp394と同じsourceごとの5位置確率再正規化を維持する。
K16によりrowごとに変化する`effective_delta_z`とrate scheduleも維持する。

fixed16は親exp394保存predictionをbaselineとしてload-only比較する。candidateは
2 wellsを同時decodeし、各well内部はNumba 2 threadsとする。future completion orderに
かかわらずwell / rowでstable sortしてからSHAとparityを評価する。

## 実験範囲

- 対象実験: `exp399_soft_sticky_fused_exact_runtime_audit`
- Route: `pf_beam`
- 親実験: `exp394_soft_sticky_exp226_k16_branch_hmm`
- 変更する変数:
  - transition kernelのmaterializationからon-the-fly sparse計算への書換え
  - dockingとH transition走査の融合
  - well outer workers `1 -> 2`
- 固定する変数:
  - fixed16 well ledger、raw / exp226入力、K16 schedule
  - TVT step `0.35 ft`、band `±100 ft`、rate states `41`
  - emission、initial distribution、switching length、docking sigma
  - E/H transition式、scaled log-space forward-backward、全posterior readout
  - technical gate、runtime limit `30,600 sec`、RSS limit `25 GB`

## 再現性設計

- seed policy: RNGなし。fold / well / row / branch / TVT / rate順を固定する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed baggingの有無: なし。
- 並列処理と乱数の関係: wellごとに状態を完全分離する。future completion orderを捨て、
  最終出力を`well_id,row_idx`でstable sortする。
- CPU/GPU runtime: Kaggle private CPU、internet off、outer workers 2、
  Numba threads per well 2、GPUなし。
- SHA: raw identity、exp226 input、親fixed16 input、scientific contract、source、
  schedule、prediction、branch posteriorを記録する。
- model / submission SHA: model、booster、submissionを生成しないため該当なしと明記する。
- Kaggle bootstrap: prepare後にembedded `config.yaml`とloose configの
  outer workers、Numba threads、run stage、kernel sources一致を監査する。

## リスク

- リークリスク: Stage 0ではhorizontal suffix `TVT`、error、exp263、hidden-like roleを
  freeze前に読まない。親predictionは数値parity専用でcandidate生成へ渡さない。
- 数値リスク: 演算順序変更によりbitwise一致しない可能性がある。dense小trellis、
  親optimized kernel、親fixed16生成物に対する段階的parity gateでfail closedする。
- 境界リスク: exp209の非正規化境界処理はコピーせず、exp394のsource-local正規化を維持する。
- runtimeリスク: E/H switching固有処理が残るため、kernel融合と2-well並列だけで
  `3.684212x`に届かない可能性がある。未達ならfull OOFへ進めない。
- メモリリスク: outer workers 2でwell-local alphaを2本保持する。fixed16実測RSSと
  full projectionを25 GB gateで確認する。
- 再現性リスク: parallel reduction順序はwell内で固定し、well間に共有reductionを置かない。
