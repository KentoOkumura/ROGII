# exp206_discussion711308_dz_dtvt_bpeak_cluster_baseline 結果

## 仮説

discussion 711308 の `dTVT ~= a*dZ+b` と `b` peak cluster は、formation level / offset の spatial coherence を no-ML baseline として説明できる可能性がある。受け入れ要件は Public LB 約 12.8 の再現。

## 結果

| version | selected | CV | Public LB | ref | submission SHA |
| --- | --- | ---: | ---: | --- | --- |
| v1 | `exact_typewell_peak_xy_k8` | 81.7364272463997 | 41.214 | 54395246 | `46832ec24b291f0fb1d6e6ecf2f1a29879da334ee83c10382734334f248f41a3` |
| v2 | `prefix_holdout_source_b_fixeda_h600` | 35.41055512960111 | 34.908 | 54396544 | `fcd44d9ada12214d605eaf301751b6ff932e27de9992008be459e8fd537fed4c` |
| v3 | `discussion_fullxyz_cluster_holdout_ab_k24_h300` | 35.30041735041327 | 29.193 | 54408573 | `0f37a593ff2a3cf3ffedcf4ecfadcaaae5d6d2ac68ba86b564eb43f608afdd23` |
| v4 | `known_tvt_fit_full` | 52.50742292458995 | 57.063 | 54458212 | `ba1903650c6da55cd64656e0eedc475701494e7250b0c62e0dd7d28a84f5e5d2` |

## v2 再現性

- train kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-train` v2
- inference kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-inference` v2
- feature content SHA: `51648ed1e36b6f193fa625037a005a6add6a69f0b09ff5bce308ac420cc79669`
- train OOF raw gzip SHA: `5596d475e7f4b505e506cae41e4fc66cf805f6a4ec67d402700c80068b7a67ed`
- full fit SHA: `e61b45b0dceb0f4354f9882e2cb773b9d538a54331633290e3f9c0e3f0287644`
- variant metrics SHA: `4efc9eadbb04ca4fbc5fcf118a112f685e850b4926e47ea9e0cbc15196f17799`
- test assignments SHA: `6d410493bd4416269d9f1378af1bd82f2c377124a1756ce1c9ee730b2b2546f8`

## v3 再現性

- train kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-train` v3
- inference kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-inference` v3
- train OOF decompressed SHA: `84bca0923a06be2689dc2b44e6a275dd00d262a0a5add55744e7a45a9630b841`
- train OOF raw gzip SHA: `ac5c913d323865cd3e70518b188751620790396f04f7b42a64fb7381ed324835`
- full fit SHA: `8dc878046a32689fc5d4db485dc4b313054b186654baf2071bea300398eacf45`
- variant metrics SHA: `3a9a3e75b767a162c579d47109f4a28256163710f2d7fa79194d0f34b3444779`
- submission SHA: `0f37a593ff2a3cf3ffedcf4ecfadcaaae5d6d2ac68ba86b564eb43f608afdd23`
- test assignments SHA: `7fbbccaea153eda1dbe19f5fbba7c7782429e0fb4b668ed40f5a39bfc328495f`

## 解釈

v1 の rate-fit は Public LB 41.214 で失敗した。v2 では discussion 本文に合わせて row-step `dTVT ~= a*dZ+b` に直し、X/Y/Z + last-300 TVT/Z feature-nearest と visible prefix holdout selector を追加したが、Public LB は 34.908 までしか改善しなかった。

v3 では「clustering test wells by X-Y-Z as well as last-300 TVT」により近づけるため、feature-nearest ではなく full X/Y/Z well geometry と last-300 TVT/XYZ shape samples による deterministic cluster を追加し、cluster/local source の `a,b` を使う variants を実装した。Kaggle train v3 では `discussion_fullxyz_cluster_holdout_ab_k24_h300` が RMSE 35.30041735041327 で、v2 selected 35.41055512960111 より小改善した。Public LB は v2 の 34.908 から 29.193 へ改善した。

したがって、v3 修正後も要件である LB 約 12.8 は達成できていない。`b` peak / local source `b` は診断 signal としては残せるが、standalone direct baseline や提出候補としては採用しない。

## v4 known TVT direct fit

ユーザー指定により、selected variant を `known_tvt_fit_full` に変更した。これは source / cluster `a,b` を選ばず、各 query/test well の known `TVT_input` 全体で `dTVT ~= a*dZ+b` を fit し、last known `TVT_input` から unknown suffix を `a*dZ+b` の累積で予測する。

- fit source: known `TVT_input` rows のみ。
- unknown suffix true `TVT`: 使用しない。
- fallback: known rows 不足などで `a,b` が finite にならない場合のみ train source full-fit median `a,b`。
- Kaggle train v4: CV RMSE 52.50742292458995、MAE 32.70950823131668、within10 0.34398276527759464、bias -4.766830325030437。
- Kaggle inference v4: `No kernel name found in notebook` で失敗。inference source に kernelspec metadata を追加して v5 で再実行した。
- Kaggle inference v5 / submit ref `54458212`: Public LB 57.063、submission SHA `ba1903650c6da55cd64656e0eedc475701494e7250b0c62e0dd7d28a84f5e5d2`。

v4 は指定された「test known TVT で fit して未知 suffix に transform」形式そのものを確認できたが、CV / Public LB とも v3 より悪化した。known prefix 全体の線形 fit は hidden tail の長い外挿に耐えず、この経路は採用しない。

## v4 再現性

- train kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-train` v4
- inference kernel: `kentookumura/exp206-dz-dtvt-bpeak-cluster-inference` v5
- train OOF decompressed SHA: `909d533512f688f8f5bcb3848cf72c55e0493b310e335b691da3bf1540bf46d7`
- train OOF raw gzip SHA: `f724b6b773f41bc310ff119fb942c97d523926f7b1169d92e3203803ba60fd6d`
- full fit SHA: `8dc878046a32689fc5d4db485dc4b313054b186654baf2071bea300398eacf45`
- variant metrics SHA: `4883b0829be9e440e6a0719364bbd4e66da1ef30bc79228617a156664a7b08b1`
- submission SHA: `ba1903650c6da55cd64656e0eedc475701494e7250b0c62e0dd7d28a84f5e5d2`
- test assignments SHA: `a7f990fdc1a01aae6f477053e4371e1102e15af4c23684eee4b1ac1aef958d4e`

## 再現失敗原因

ローカル train pseudo-tail で oracle と selector を分けて診断した。結果は以下の通り。

| diagnostic | RMSE | 解釈 |
| --- | ---: | --- |
| hidden suffix oracle, free `a,b` | 5.342146 | hidden tail の真値から target well 自身の `a,b` を最適化した上限。式そのものは強い。 |
| hidden suffix oracle, `a=-1` fixed + best `b` | 7.587489 | offset / `b` さえ当たれば 12.8 圏内に入る。 |
| target full true fit `a,b` oracle | 12.379107 | target well 全体の真値 fit を使うと要件水準に近い。つまり係数推定・選択がボトルネック。 |
| v2 selected `prefix_holdout_source_b_fixeda_h600` | 35.410555 | Kaggle Public LB 34.908 と整合。submission/package の問題ではない。 |
| source `b` を `a=-1` 固定 fit に揃えた診断 variant | 34.895049 | 現行の自由 `a,b` fit の `b` 流用は小さな不整合だが、主原因ではない。 |
| own prefix tail300 fixed `a=-1` | 40.028805 | visible prefix だけでは hidden tail の `b` を十分に推定できない。 |

主原因は target-free な source / `b` 選択の精度不足。v2 selected の assigned `b` は hidden tail の fixed-`a` oracle `b` に対して weighted mean abs error 0.00815 だった。train pseudo-tail の hidden rows は median 4,840、p90 6,349 なので、`b` が 0.01 外れるだけで median tail で約 48 ft、p90 tail で約 63 ft の累積ドリフトになる。要件水準には `b` error を概ね 0.002-0.003 台まで落とす必要があるが、現行の X/Y/Z + last-300 feature-nearest と prefix holdout では届いていない。

また、visible prefix holdout は平均 RMSE 1.52 と良く見えるが、hidden tail では RMSE 35.41 まで悪化した。prefix 後半で最良の `b` が、その先の長い hidden suffix に外挿できていない。`b` peak label も prefix/full 一致率は weighted 86.47% あるが、2 peak の粗い分類だけでは同一 peak 内の連続的な `b` 差を吸収できない。

## 次

- この backlog は「実装・Kaggle 実行・提出済み、要件未達」として閉じる。
- 再挑戦するなら exp206 の小修正ではなく、discussion 699853 側の offset selector / discrete offset classifier や既存 PF/Beam 候補選択に接続する別設計として扱う。
