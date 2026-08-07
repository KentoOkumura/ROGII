# exp257 nested selector output replacement-only on exp218

## 状態

Kaggle GPU train v1完了、CV guard不通過です。その後のユーザー明示指示により、比較用overrideとしてKaggle inference v2とcode submissionまで完了しました。Public LBは`7.718`（ref `54753824`）で、exp238 hidden-safe `7.775`を`-0.057`改善しました。Public-LB上のML route anchorは更新しますが、CV不通過のためtrain-side採用とは分けて扱います。

## 仮説

exp238で追加したHMM・self-GR HMM・exp226を含む11候補nested selectorを使いながら、
selector出力を`nsel_*`としてLightGBMへ追加しない実験です。

exp218の380特徴に既にある`ll_*` 54列を次の2群へ分けます。

- selector出力29列: exp238 outer5-inner4 scoreからfoldごとに再生成して上書き
- selector入力診断25列: exp218の値をそのまま維持

最終LightGBMの列名・順序・列数はexp218と完全に同じ380列です。`nsel_*`は0列、
候補パスの直接置換・blend・postprocessもありません。selector 20本は再学習せず、
新規学習はreplacement-only LightGBM 3 configs × 5 folds = 15 boostersだけです。

Kaggle trainのguardは不通過のまま維持しています。推論実行は採用昇格ではなく、ユーザー指示によるoverrideです。

## 検証方針

exp238の保存済みouter 5 × inner 4 scoreを固定し、exp218と同じ380列schemaで
3 configs × 5 foldsのLightGBMだけを学習します。比較の主対象は同じouter foldを使った
exp238 add-only OOFです。global、fold、near、1000+、hidden-like、worst-wellとfeature importanceを
確認するまでinferenceへ進みません。

## 所見

380 = 351 unchanged + 29 replaced、`nsel_*` 0列の契約と15 model完走は確認済みです。
`lgb_mean`は8.101331で、同一fold exp238 add-only 7.936690より+0.164641悪化しました。
nearは+0.068730、1000+は+0.184078、改善foldは1/5、worst-well回帰は+13.291303のため、
今回のreplacement-only仮説は採用しません。

Kaggle inference v2は463.95秒で完走しました。14,151行 / 3 well、184 context、COPCF 41/41、欠損context 0、test-test neighbor 0、29列上書き / 25列維持 / `nsel_*` 0 / 最終380列を確認しています。保存済み20 selectorと15 final LightGBMだけを使い、推論中の学習は0です。生成した`submission.csv`はsampleとID順まで一致し、重複・NaN・Infなしで提出前検証PASSです。code submission ref `54753824`はPublic LB `7.718`で完了しました。
