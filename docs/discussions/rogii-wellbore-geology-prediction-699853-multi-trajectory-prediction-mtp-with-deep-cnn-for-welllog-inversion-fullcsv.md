# multi-trajectory prediction (MTP) with deep CNN for welllog inversion

- archived_at: 2026-06-24T12:40:27Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853

Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.0), please consider upgrading to the latest version (2.2.2) id,title,authorName,commentCount,votes,postDate 699853,multi-trajectory prediction (MTP) with deep CNN for welllog inversion,hengck23,93,53,2026-05-15 13:49:18.283000 id,authorName,postDate,votes,content 3467614,hengck23,2026-06-06 20:20:17.057000,3,"

" 3467727,hengck23,2026-06-07 07:25:49.107000,1,"
CNN+SDF +MTP:

top3 prob path and last one is mean of top4+5

for the first time, we have correct prediction inside the top 3. input is 512 window of h tvt (compression=2) and 64 window of t tvt

```text
        #add features-------------------------------------
        history  = (
            t_tvt.reshape(B, 1, 1, T).expand(B, 1, H, T)
            - h_tvt_history.reshape(B, 1, H, 1).expand(B, 1, H, T)
        )
        mask = h_tvt_mask.reshape(B, 1, H, 1).expand(B, 1, H, T)
        history = history*mask

        #-----------------------------------------

        image = torch.concat([
            t_gr.reshape(B,1,1,T).expand(B,1,H,T),
            h_gr.reshape(B,1,H,1).expand(B,1,H,T),
            t_gr.reshape(B,1,1,T).expand(B,1,H,T)-h_gr.reshape(B,1,H,1).expand(B,1,H,T),
            history,
            mask,
        ], dim=1)
        image   = self.norm(image)
        feature = self.backbone(image)

```

`

no x,y,z information. so i am sure the CNN is learning something useful from the gr data. my transformer cannot learn from gr which i don't know why (i suspect it is a normalisation issue?)


" 3467728,hengck23,2026-06-07 07:29:55.810000,0,"
results of training mixture



i think it has to be a mixture model because I can see the prediction hopping around a few modes
" 3467797,Tom,2026-06-07 12:42:31.467000,0,
They look more like some basis
 3467823,hengck23,2026-06-07 15:03:27.597000,2,"
validation results for full length tvt (by probing, all hidden test well has kength <12_000). h tvt window = 384, h tvt window 768 (at compression =16)

you can see sdf is bending correctly, i.e. it indeed learned the steering. the magic is adding well dip tangent feature (sin and cos of dmd/dz) and well direction tagent feature (sin and cos of dx/dy, i.e. geology dip) + gr heatmap


" 3467916,hengck23,2026-06-07 23:28:50.513000,0,"
[@tom99763]

they are less primitive after plotting at the recovered TVT
 sdf = t_tvt - h_tvt, hence h_tvt = t_tvt - sdf for sdf.abs()<2

instead of generating more K per model, it is better to save a few models at different iterations and run them at inference



different iteration


" 3468049,hengck23,2026-06-08 10:14:32.910000,1,"


i think rmse error is some how biased (e.g. increases with length due to error accumulation. post processing your results may help)
" 3466618,hengck23,2026-06-04 11:04:28.190000,1,"


plot of tvt max - tvt min verus tvt length
" 3466626,hengck23,2026-06-04 11:14:49.710000,0,"



" 3465013,hengck23,2026-06-01 05:29:45.583000,3,"


transformer MTP of the previous post. I just need a good verifier
" 3465021,hengck23,2026-06-01 06:03:26.827000,0,"

" 3465080,sleep3r,2026-06-01 10:44:00.597000,1,"
gr matching is ill-conditioned: even at the true tvt horizontal <--> typewell gr only corr ~0.7, and offset error compounds
" 3465106,hengck23,2026-06-01 12:16:28.557000,0,"
The best way is to make a model that can recover h tvt from h gr = interp( h tvt, tw gr, tw tvt). This is perfect correlation but has multiple fp matches. If that works, you can introduce gausssian noise, offset noise, scale noise, simplifcation noise, etc as
" 3465107,hengck23,2026-06-01 12:18:13.850000,0,
My feeling is that we need to train a ranker or scorer rather than rely on generic correlation
 3465111,sleep3r,2026-06-01 12:26:55.423000,0,"
been down exactly this road

mtp heatmap net + a learned catboost ranker over the modes (pairwise yetirank) + the h_tvt-from-h_gr recovery + gaussian/offset/scale/simplification aug

two walls i couldn't pass: gr is barely discriminative for selection - no_gr ≈ shuffled_gr ≈ real_gr on top1, the net basically ignores it. and the ranker looks amazing in-window (spearman score <-> error ~−0.92, top3 rate .92) but it does NOT convert row-level once you go strict well-grouped oof - my best honest gain over gbm was ~+0.03ft, the big in-sample numbers were pure ranker leakage

on top of that ~23% of wells have no good candidate at all, so the scorer is capped no matter how good it is
" 3465159,hengck23,2026-06-01 14:15:47.917000,0,
we do not need to match all GR. we have good dtvt estimate. we just need a few anchor points to push the whole tvt curve to correct the pace.
 3465161,sleep3r,2026-06-01 14:29:27.077000,1,"
agreed a few anchors is all you need - with oracle anchors i get k≈10 down to ~4ft, k≈20 to ~1.7ft, so your pace-correction framing is right

the catch is placing them: a local gr shift-search around an anchor just can't localize it. the typewell gr repeats, so sliding the curve +-tens of ft fits the log about equally well

my gr-picked anchors gave basically zero gain over no correction, even after gating to only high-corr anchors. so imo the wall isn't ""match all gr vs a few anchors"", it's getting even one trustworthy anchor out of gr. how are you deciding which anchors to trust?
" 3465649,hengck23,2026-06-02 15:30:41.990000,3,"
[@sleep3r]

my suggestion is that you start with the native PF method from [https://www.kaggle.com/code/sunnywu27/rogii-wellbore-tvt-physical-model]

then replace the likelihood scorer with a learned local CNN one

```text
        #initialisation
        # tvt = geo_z - z + bias
        # geo_z+bias = tvt + z
        w = np.ones(num_particle) / num_particle
        pos = last_tvt + last_z + 2.0 * rng.standard_normal(num_particle)
        vel = last_vel + 0.01 * rng.standard_normal(num_particle)

        cum_log_likelihood = 0.0
        output = h['TVT_input'].values
        for i in range(h_ps+1, len(h)):
            dm = h_md[i] - h_md[i-1]

            #create particle
            vel = 0.998 * vel + 0.002 * rng.standard_normal(num_particle) #chnage this to torch tensor
            pos = pos + vel*dm + 0.005 * rng.standard_normal(num_particle)

            tvt = pos - h_z[i]
            tvt = np.clip(tvt, t_tvt[0] - 100, t_tvt[-1] + 100)
            pos = tvt + h_z[i]

            #---

            ##--- change this to CNN learned likelhood score ---------------
            #gr  = np.interp(tvt, t_tvt, t_gr)
            #gr_error = gr - h_gr[i]
            #gr_std = 30
            #gr_likelihood = np.exp(-0.5 * np.minimum((gr_error / gr_std) ** 2, 600.))
            ##--- change this to CNN learned likelhood score ---------------
            #e.g.
            t_win_gr = extract a window from t_gr (refrence well)
            h_win_gr = extract a window from h_gr (horizontal well)
            gr_likelihood = net(t_win_gr, h_win_gr)

            image = torch.cat((t_win_gr.unsqueze(1), h_win_gr.unsqueze(2)), dim=0)
            feature = cnn(image)
            --> learn to match sdf = t_win_tvt - h_win_tvt

            likelihood = gr_likelihood
            likelihood = np.maximum(likelihood, 1e-300)
            avg = float((w * likelihood).sum())
            cum_log_likelihood += np.log(max(avg, 1e-300))

            #updte weight
            w = w * likelihood
            if w.sum() == 0:
                w = np.ones(num_particle) / num_particle
            else:
                w = w / w.sum()

            #resample
            effective = 1.0 / np.sum(w ** 2)
            if effective< 0.5 * num_particle:
                idx = rng.choice(num_particle, size=num_particle, replace=True, p=w)
                pos = pos[idx] + 0.1 * rng.standard_normal(num_particle)
                vel = vel[idx] + 0.001 * rng.standard_normal(num_particle)
                w = np.ones(num_particle) / num_particle

            #prediction
            predict = (w*pos).sum() - h_z[i]
            output[i] = predict

```
" 3465758,sleep3r,2026-06-02 19:25:44.183000,0,"
tried your cnn-likelihood idea pretty hard

window matcher (h_gr vs typewell, learn the sdf) + noise aug, fold-safe. couldn't get it to beat the plain point-gr likelihood

matcher's peak sits ~200ft off the true tvt and AUC caps ~0.7, the gr just repeats too much to localize a window

what I did confirm though:

pf framework itself is great - drop in an oracle likelihood and even at alpha=30ft it nails ~1ft, so the whole game is purely likelihood centering and gr alone can't give it

the thing that actually moved my score was blending the pf with a gbm - their error tails are decorrelated, big drop

does your cnn ever beat point-sr on a strict well-grouped oof? that's where mine died
" 3465763,hengck23,2026-06-02 19:30:28.297000,1,
Cnn does improve on specific cases and but general cases.
 3465869,hengck23,2026-06-02 23:54:24.810000,0,"
examples of different methods
 CNN+sdf (using gr) : global gr waveform pattern



transformer on dz (not using gr) : dz prior


PF on single value GR : local gr match based on local stste (velocity + pos)
" 3465975,hengck23,2026-06-03 07:12:22.640000,2,"
The PF code uses lookup geo plane for training wells. what if we model the geo surface using grid interpolation? a validation rmse of 11.09 (non optimized)


" 3466045,PatrickAIForFun,2026-06-03 11:24:17.303000,0,"
Yes, I can confirm - basic / non optimized kriging of geology layers gave a local RMSE ~11 and test rmse ~13.5 for me.
" 3466050,hengck23,2026-06-03 11:50:26.133000,0,"
Try better offset adjustment. Plot the graphs. Validation rmse should be near 11. Use the tvt input, geo predict and given well z near ps to determine best offset
" 3465053,wqi876,2026-06-01 08:22:53.033000,2,
Thank you very much for your discussion. It has been very helpful to me. And your profile picture is so cute!
 3464967,hengck23,2026-06-01 02:29:59.483000,2,"
effects of hacks

no GR features are used.
 input only use x,y,z,dz,dtvt history, tvt history



left: validation, right: train
 red: predict, blackL ground truth
 (do note the scale of the y axis when interpreting results)

predict task: dtvt
 history = 256, future horizon = 1024 (2048 shows smiliar results)
 model: just normal transformer

there are some dift, if i can solve those using GR, maybe good results.
 i am thinking of estimate dift = oberserved GR - interp(predict tvt, typewell_tvt, typewell_gr) at some fixed intervals/anchors (maybe CNN is useful here)
" 3464968,hengck23,2026-06-01 02:34:15.850000,0,"
```text
        seq = torch.cat([
           h_dtvt_history.reshape(B,H,1),
           h_tvt_mask.reshape(B,H,1),
           h_dz.reshape(B,H,1),
           h_x.reshape(B,H,1),
           h_y.reshape(B,H,1),
           h_cos.reshape(B,H,1),
           h_sin.reshape(B,H,1),
        ], dim=2)
        #print(seq.shape)

        seq = self.to_seq(seq) #project to dmodel
        h_idx = torch.arange(H, device=device)

        seq = seq + self.h_idx_emb(h_idx).reshape(1,H,-1) #pos encode
        seq = torch.concat([
            self.cls.reshape(1,1,-1).repeat(B,1,1),
            seq,
        ], dim=1)

        padding_mask = h_padding>0.5 #convert to bool (B,H)
        padding_mask = F.pad(padding_mask, (1,0), value=False) # add cls (B,1+H)
        hidden = self.tx_encoder(seq, src_key_padding_mask=padding_mask)#src_key_padding_mask include cls
        cls, hidden = hidden[:,0], hidden[:,1:]

        dtrajectory = self.dtrajectory(hidden)
        dtrajectory = dtrajectory.permute(0,2,1) + h_dtvt_history.permute(0,2,1) #B,K,H

```
" 3462931,hengck23,2026-05-25 11:16:58.867000,5,"
i discover a hack!



first fig: dz
 second fig: dtvt
 why? annotation leak! (that is how starsteer works)
" 3462934,hengck23,2026-05-25 11:21:05.803000,1,"

" 3462940,Tom,2026-05-25 11:41:34.557000,2,
The red/blue = direction segments sign(dtvt). My test just confirmed the structure underneath it: ANCC (formation top) is ~piecewise-linear with ~15 control points per well (~323 rows apart). That is the sparse StarSteer dip annotation. LOL
 3462946,hengck23,2026-05-25 12:05:10.733000,1,
maybe just prediction dtvt = a(dz)*dz. i.e. your network predict dtvt and use both local dtvt loss and global cumsum tvt loss
 3462951,sleep3r,2026-05-25 12:23:26.710000,1,"
yeah, this seems real. I tried using ANCC only as a train-time teacher:

target: sign(dANCC) = down/flat/up features: test-safe MD/X/Y/Z/GR/TVT_input only

5-fold OOF hidden direction accuracy is ~0.927. so formation-top annotation seems distillable into a test-safe state model. now checking if this state helps chunk/DP path selection


" 3462954,hengck23,2026-05-25 12:42:52.793000,2,"


i plot dz and dtvt on the same plot. they are the same scale !!!! maybe competition will reset
" 3462957,hengck23,2026-05-25 12:50:52.837000,3,"
```text
    h_tvt = h[""TVT""].values
    h_z = h[""Z""].values
    h_md = h[""MD""].values
    h_dtvt = np.gradient(h_tvt)
    h_dz   = np.gradient(h_z)

    plt.plot(h_md, -h_dz)
    plt.plot(h_md,  h_dtvt)
    plt.axvline(x=h_md[h_ps], color='red', alpha=1)
    #plt.plot(dz_smooth)
    plt.show()

```




" 3462961,Tom,2026-05-25 12:58:20.330000,2,"
−dz and dtvt being the same scale and overlapping in long stretches means: wherever the formation is flat, dtvt = −dz exactly (dANCC=0 → TVT = −Z + C). They only diverge at dip events (your ~15 control points), and the parallel-offset stretches in your middle plot are exactly those flat segments where TVT = −Z + a constant
" 3462962,Tom,2026-05-25 12:59:37.323000,1,
time to reset now
 3462966,hengck23,2026-05-25 13:10:10.143000,1,"


```text
   h_dtvt = np.gradient(h_tvt)
    h_dz   = np.gradient(h_z)

    H_unknown = len(h_tvt) - h_ps
    truth_tvt = h_tvt[h_ps:]
    ##---
    #find offset
    offset = h_dtvt[h_ps-500:]+h_dz[h_ps-500:]
    offset = np.median(offset)  #use ML to learn offset

    predict_dtvt = -h_dz[h_ps:]+offset
    predict_tvt = np.zeros((H_unknown,))
    predict_tvt[0] = h_tvt[h_ps]
    for i in range(1, H_unknown):
        predict_tvt[i] = predict_tvt[i-1] + predict_dtvt[i]
    #
    print(len(predict_tvt), len(truth_tvt)) #additional point at h_ps
    rmse = np.sqrt(np.nanmean((predict_tvt - truth_tvt)**2))

    plt.plot(predict_tvt, label=f""predict_tvt {rmse:0.2f}"")
    plt.plot(h_tvt[h_ps:], label=""h_tvt"")

```
" 3462969,hengck23,2026-05-25 13:24:11.323000,1,
i think the offset could be fixed values. my experiments seems to suggest they are limited to set of values
 3463129,hengck23,2026-05-26 01:20:45.760000,4,"



" 3463137,Tom,2026-05-26 01:56:35.577000,1,
cumsum(−dz − offset) with a discrete offset => 7.7 rmse
 3463139,hengck23,2026-05-26 02:06:20.040000,0,
Just need a classifier to choose global offset
 3463219,sleep3r,2026-05-26 08:53:48.643000,1,"
a fine offset-grid oracle gives ~7.64 RMSE on train hidden rows for me

but choosing the offset is the hard part: known-prefix offset gives ~37-39 RMSE, and my fold-safe selector only gets ~14.8. So I think the next step is not direct TVT regression, but learning the offset/state with cumulative TVT loss

I started a no-prior model around this idea: predict residual dC / offset-state from test-safe MD/X/Y/Z/GR/TVT_input, then reconstruct TVT by cumsum

still early, but this formulation feels much closer to the leak than my previous GR/MTP attempts
" 3463221,Tom,2026-05-26 09:01:18.850000,0,
Fuzzy inference or mixture desnity network would help
 3463317,hengck23,2026-05-26 15:24:48.587000,0,"
The first try should be :

```text
1) given current location s
2) given a list of offset = -0.1 to 1.0
3) given a list of  future location s1 = 25,50,75, 100, ... 300
4) compute tvt rmse for each candidate pair (offset,s1) above :   tvt rmse = rmse (true tvt[s0:s1], tvt derived from dz and offset)
5) train a regressor : score = model( h_gr_smooth[s0:s1], sampled gr using dz and offset, aux input)
6) score from (5) must correlate with  tvt rmse from (4). or at least the min point should coincide

```
" 3463320,hengck23,2026-05-26 15:30:35.663000,0,"
brute force search is 12.18 for one fold

```text
    t = pd.read_csv(f""{KAGGLE_DIR}/train/{sample_id}__typewell.csv"")
    h = pd.read_csv(f""{KAGGLE_DIR}/train/{sample_id}__horizontal_well.csv"")
    h_ps = int(np.flatnonzero(h[""TVT_input""].notna().values)[-1])

    h_gr_filled = h[""GR""].interpolate().bfill().ffill().values
    h_gr_smooth = savgol_filter(h_gr_filled, 100, 3)
    h_tvt = h[""TVT""].values
    h_z = h[""Z""].values
    h_md = h[""MD""].values
    h_ancc  = h[""ANCC""].values
    h_dtvt  = np.gradient(h_tvt)
    h_dz    = np.gradient(h_z)
    h_dancc = np.gradient(h_ancc)  #ground truth offset

    span = [100]  # let's try one
    offset =  np.linspace(-0.8, 0.8, 201)  # covers 90% of cases

    rmse_tvt = []
    rmse_gr  = []
    rmse_tvt_score = np.zeros((len(span), len(offset)))
    rmse_gr_score = np.zeros((len(span), len(offset)))

    predict = []
    s0=h_ps
    while s0<len(h_tvt):
        best_tvt = None
        best_tvt_rmse = np.inf
        best_gr = None
        best_gr_rmse = np.inf
        for si, sp in enumerate(span):
            s1 = s0+sp
            s1 = min(s1,len(h_tvt))
            for j in range(len(offset)):

                sm_tvt = last + (h_dz[s0:s1]-offset[j]).cumsum()
                sm_gr  = np.interp(sm_tvt, t[""TVT""].values, t[""GR""].values)
                r_tvt =  do_rmse(sm_tvt,h_tvt[s0:s1])
                r_gr  =  do_rmse(sm_gr,h_gr_smooth[s0:s1]) #- si*0.5

                rmse_gr_score[si,j] = r_gr
                rmse_tvt_score[si,j] = r_tvt
                if r_gr<best_gr_rmse:
                    best_gr_rmse = r_gr
                    best_gr = [s0,s1,sp,j,offset[j], r_tvt]

                if r_tvt < best_tvt_rmse:
                    best_tvt_rmse = r_tvt
                    best_tvt = [s0, s1, sp, j, offset[j], r_gr]

        rmse_tvt.append(best_tvt_rmse)
        rmse_gr.append(best_gr_rmse)

        # plt.imshow(np.hstack([stats.zscore(rmse_tvt_score),stats.zscore(rmse_gr_score)]))
        # plt.waitforbuttonpress()

        s0, s1,_, j, _, r_tvt = best_gr
        s1 = int(0.8*s0+0.2*s1)  #back track ... don't trust it
        if s0==s1: s1=s0+1

        p_gr  = last + (h_dz[s0:s1]-offset[j]).cumsum()
        predict.append(p_gr)
        print(s0, s1-s0, offset[j], r_tvt, best_gr_rmse)
        s0=s1

    predict = np.concatenate(predict)
    r = do_rmse(predict,truth)
    print('***',r)  #rmse for one
    all.append(r)

print(""-------------------------------------"")
print(np.mean(all)) #12.18 (not the same as lb metric with mean over all rows (not sample wise)
exit(0)

```
" 3463322,hengck23,2026-05-26 15:34:45.773000,2,"
there is a mxiture/DP transformer that chatgpt recommend:

Lattice Deduction Transformers [https://arxiv.org/html/2605.08605v1]

```text
class RogiiLatticeTransformer(nn.Module):
    """"""
    Simple lattice transformer for trajectory prediction.

    Input:
        h_seg:       (B, S, L) horizontal GR split into S segments, each length L
        t_gr_bins:   (B, N, L) typewell/reference GR windows for each TVT bin
        alive:       (B, S, N) current lattice candidates, 1=alive, 0=removed

    Output:
        keep_logits: (B, S, N) logits saying whether candidate TVT bin should remain alive
        move_logits: (B, S-1, A) optional movement logits between segments
    """"""

```
" 3464169,Tucker Arrants,2026-05-29 03:05:08.587000,-1,"
I think they need to reset. Surely providing the post-PS trajectory (X/Y/Z) is a problem? It's causally downstream of the answer - the driller steered based on where the formation actually was, so the trajectory ahead of the bit already encodes what we're supposed to be predicting. Feels like it should be masked.
" 3464170,hengck23,2026-05-29 03:11:07.863000,2,
Not direct nor obvious answer. Still needs some clever hack to work. But does make getting answer easier.
 3464209,PatrickAIForFun,2026-05-29 06:46:09.960000,2,"
I don't think a reset is necessary. If you look at all training videos and resources by ROGII one can clearly see that there are two types of geosteering which are done in the real world:

live geosteering: get the data from the current bore-head and give it directions to stay in the oil. Here you are also given XYZ and GR up to the current position and can't change previous decisions.

post-drilling steering: Here you are also given the full XYZ trajectory amd the full GR log and now have to determine the rock structure you are drilling through. This is exactly what our task is and what is shown in most StarSteer-Geosteering training videos. In the real world you are also given the true XYZ and can assume thag during live-steering it followed the rock formation. I guess the goal here is to get a post-hoc understanding of the rock for future well planning.

Either way, in the real world application we are also given this exact same data.
" 3462344,hengck23,2026-05-22 23:54:38.270000,6,"
update on cnn+sdf:

some backbone and decoder architecture are better

augmention using flip + different stretch improve results

time to spend on generator to generate more possible train data: create path --> sample from typewell --> add oise (actually we can do it in test-time or better still offline since, we have the hidden testwell location in host slides)
 " 3462815,hengck23,2026-05-25 05:22:30.283000,3,"
i can do some fast match from visual inspection if i segment the direction of the well

look for highest and lowest point

check neighbourhood values from that point

then you can find large segment and you can almost get find min/max of well tvt

it seems to me the logic is:

if you are lost, continue to move in a direction when you find a prominent GR pattern (usually high or low values), so that you can reset to a known position.

then back track to where you are lost.
 " 3462838,Tom,2026-05-25 07:25:45.937000,2,
Developing a “Trace Back” mechanism could further improve the score. One possible approach is to build a dictionary (or bag-of-signals) that serves as a strong reference for matching
 3462912,hengck23,2026-05-25 10:38:02.067000,5,"
i suddenly have a cheat method.

1) you are at typewell location s at PS.

2) we are not interested in tracinig the well trajetory. rather we are interested in detecting the max and min offset values, where TW_GR( a*tvt + offset) can be matched in horizontal well.

3) so we can create many templates of TW_GR(a*tvt + offset) with different values a and scale.

4) once we have this, just predict trajectory = (max tvt + min tvt)/2. if you do this correctly, you get rmse about 8.5
" 3462930,sleep3r,2026-05-25 11:11:55.463000,1,"
i tried a similar direction: instead of trusting one global GR heatmap, i build local GR-event candidates and then use a chunk-level DP policy to stitch/select a smooth path

early result: this does add useful candidate space. on an 80-well diagnostic, oracle use of these traceback bands improved baseline rmse from ~9.99 to ~8.76 on covered rows. a small chunk-DP smoke test also improved ~9.47 → ~9.24

but the caveat is important: shuffled/zero-GR sanity is still not clean, so the current signal is not pure GR matching yet. it seems useful as sparse reset anchors / candidate bands, but needs full OOF validation before trusting it
" 3462044,hengck23,2026-05-22 02:24:40.230000,4,"
one challenge of the competition is to find good representation. Here is using cnn + sdf (signed distance function)


" 3462047,Tom,2026-05-22 02:30:02.903000,2,"
SDF seems like a solid option. This also reminds me of the Vesuvius Challenge, might be able to transfer some tricks from there.
" 3462066,hengck23,2026-05-22 04:19:41.070000,5,"
[@tom99763]

demo inference and training code are up:
 [https://www.kaggle.com/code/hengck23/cnn-sdf-example]
 [https://www.kaggle.com/datasets/hengck23/hengck23-rogii-cnn-mtp-demo] (training py file)
" 3462067,hengck23,2026-05-22 04:32:09.797000,1,
The fact that CNN can detect micro 2d pattern makes me think that the data are probably synthetic or the signal modelling in geology is really good?
 3462070,hengck23,2026-05-22 04:36:33.553000,1,"
i am thinking of predicting the geology plane, eg ANCC = tvt -z instead. such planes are more linear and benefit from sdf (natural smoothness and planar regularisation from ground truth!)
" 3462197,Tom,2026-05-22 12:53:40.013000,1,
Tvt - z can work better than directly predicting tvt.
 3462225,hengck23,2026-05-22 14:54:09.883000,2,"
instead of

```text
mistfit_gr = t_gr-h_gr

```

use

```text
mistfit_gr = t_gr- interpolate( h_tvt-well_z, h_tvt, h_gr)

```

maybe you can see a linear zero line (matched gr)
" 3462243,Tom,2026-05-22 15:22:53.963000,2,"

" 3462253,hengck23,2026-05-22 15:41:25.333000,3,"


i tried some toy data
" 3462259,hengck23,2026-05-22 15:58:11.203000,2,"
So the pf, k-beam, dp, viterbi etc searches are just detecting lines or multi ple lines hypothesis.

But there is an issue, ancc plane anchoring means the range of tvt is very small if the geological plane is horizontal, ie no gr pattern to match. Need to modify the anchoring
" 3459225,hengck23,2026-05-17 14:22:01.793000,6,"
example notebook [https://www.kaggle.com/code/hengck23/cnn-mtp-example?scriptVersionId=320093395]
" 3459046,hengck23,2026-05-17 08:52:10.707000,5,"
results is good at least for short-term forecast of 8 future interval steps. (each interval uses average of 32 GR values). here are validation resuits. black is truth, orange is probability weighted average, red are top 8 paths (shade = probability)

maybe top 6 is enough, becuase the last 2 never get activated




" 3459062,hengck23,2026-05-17 09:37:13.173000,5,"
try a longer horizon of future steps=16, history=9. as expected, prediction starts to diverge. but good news is that the truth path is still predicted as a lower score candiate, eg top-6 solution …. maybe it can be saved.






" 3461380,hengck23,2026-05-20 12:39:39.033000,4,"
what you get if you use unet and do ""blood vessel"" segmentation

validtation

 training

```text
    def forward(self, typewell, horizontal, hint):

        #todo raw signal channel
        B,T = typewell.shape
        B,H = horizontal.shape

        image = torch.concat([
            typewell.reshape(B,1,T,1).expand(B,1,T,H),
            horizontal.reshape(B,1,1,H).expand(B,1,T,H),
            hint,  #input tvt
        ], dim=1)

```
" 3461383,hengck23,2026-05-20 12:47:29.917000,1,"
i am surprised that some results are perfect and it is bidirectional and needs not to be continuous (e.g. match can happen in the middle of image and propagate out)


" 3461423,Tom,2026-05-20 14:40:26.917000,2,
might can consider problem as iterative image inpainting I think.
 3461450,hengck23,2026-05-20 15:53:58.387000,2,"
I am surprised there is no multiple paths. I only use bce loss. Some paths diverted from the truth with high confidence. It means that if we use gr information, we can very similar train labels that “diverge” from the validation labels. I have no ideas how to correct these
" 3464155,hsiaosuan,2026-05-29 01:37:32.887000,1,
Reminds me of Vesuvius!!
 3462538,sleep3r,2026-05-23 18:35:23.623000,1,"
Tried a sliding-window CNN that predicts TVD corrections over the base prior, using horizontal GR + typewell correlation. Added synthetic pretraining to teach the correlation - worked great on synthetic data (93% accuracy), completely failed to transfer to real wells. Spent way too long on that

Net result: ~+0.03 ft over baseline. Looking at the leaderboard that's basically nothing
" 3462629,sleep3r,2026-05-24 09:55:54.517000,2,"
I rechecked with real train-well panels. The issue seems not only CNN/MTP capacity: the true TVT path often is not a reliable high-score ridge in the GR/typewell heatmap. In our localized/stretch panel, normal GR top10 coverage is 25.6%, shuffled GR is 26.1%. The first thing that strongly changes the search is stratigraphic/zone restriction, but direct Geology/formation labels are not available in test




" 3462675,hengck23,2026-05-24 14:09:20.130000,1,"
how about let GR = concate (gr values, location values). then each GR value is diiferent. correlation is match of values and distance
" 3462692,sleep3r,2026-05-24 15:56:22.933000,1,"
I tested this exact idea: combine GR matching with a test-safe location prior

Concatenating / combining location with GR absolutely helps remove global false ridges But the effect seems to come from the location prior, not from GR itself

When I keep the exact same location prior and replace lateral GR with shuffled GR, the heatmap still looks very similar and the top1 path behaves similarly. So the problem is not just “GR needs location”; the remaining GR/typewell likelihood still does not pass shuffled-GR sanity


" 3462771,hengck23,2026-05-24 23:53:06.373000,1,"
The problem of geosteering is actually ""move the wellbore between the target top and bottom geology region."" Hence, here the inversion is localised, where is the wellbore within the layers? You can estimate the limits first
" 3462777,hengck23,2026-05-25 00:19:28.800000,1,"
check 10a1281a.png in the train dataset



the reference TW GR signal for matching is only ""so short"". many of the horziontal GR ""windows"" are not useful at all except for the peaks
" 3459745,hengck23,2026-05-18 08:06:52.470000,3,"
Take-home message: mathematical correlation versus machine-learned correlation.

So anything that is imperfect can be made perfect by learning. eg, we have our DTW needs to take care of reverse indexing. Although i found a paper on drop-DTW (dropping invalid segments), it didn't work well because of noise. maybe i should learn the dropping and wraping i instead (of using DP)





ROGII used another kind of feature image
" 3459764,hengck23,2026-05-18 08:41:32.733000,2,"


cnn should be very good to capture these micro box patterns (pairs of 2d signal). these are just like 2d tokens. But i need to recreate ROGII segment endpoints annotations.


" 3459776,Tom,2026-05-18 09:08:40.227000,1,"
Thanks, this is very useful info
" 3458284,hengck23,2026-05-15 14:25:59.273000,5,"
code and lesson (lecture notes) [https://github.com/geosteering-no/inversion_school_geosteering/tree/main]




" 3458598,hengck23,2026-05-16 08:33:05.993000,4,"
i make some plots. i think the formulation is not the issue. the issue is that the data is really noisy. It is difficult for human to match if we only see a window segment of vertical and horizontal GR signals.






" 3459711,hengck23,2026-05-18 06:58:50.437000,1,"
learning distance fields




" 3458642,hengck23,2026-05-16 10:27:37.610000,1,"

" 3458635,hengck23,2026-05-16 10:10:03.043000,1,"
 another example
" 3458633,hengck23,2026-05-16 09:56:52.223000,2,"

" 3472604,Pratyaksh,2026-06-14 16:55:15.380000,0,"
The Problem — Compounding Error In Sliding Window Inference Some test wells have 5000+ unknown rows but my model only predicts H_FWD * S = 512 rows per pass. So I use a sliding window at inference: Pass 1: anchor = last known TVT_input → predict 512 rows Pass 2: anchor = last predicted TVT → predict next 512 rows Pass 3: anchor = last predicted TVT → predict next 512 rows … The issue is obvious — each pass inherits the error from the previous one. If pass 1 drifts by 5 TVT units, pass 2 starts from that wrong position and compounds further. My partial mitigation is that the GR signal is always real and known for all rows — so the heatmap is built from true GR regardless of anchor TVT accuracy. The anchor mainly affects which typewell rows get cropped, and with a wide enough crop the model can self-correct via GR alignment. But it's imperfect.
" 3466084,hengck23,2026-06-03 13:33:52.007000,0,"


i show the same solution in two different visualisations. z prior is much stronger than gr prior.
" 3466086,hengck23,2026-06-03 13:39:17.153000,0,"
eg, it is ""easy"" to correct this error?

the initial offset before PS tells a lot about the distance between well z and geo z


" 3466057,hengck23,2026-06-03 12:17:33.370000,0,"
maybe this is helpful
" 3463364,,2026-05-26 18:01:38.883000,0, 3458271,,2026-05-15 14:14:38.087000,0,
