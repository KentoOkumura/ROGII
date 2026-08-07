# [practical notes] from reproducing and stress-testing notebooks

- archived_at: 2026-06-11T13:49:24Z
- source: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/703883

Topic #703883: [practical notes] from reproducing and stress-testing notebooks
  Author: Jiayi Du
  Posted: 2026-06-02 14:15:15.987000
  Votes: 3  Comments: 0

Hi builders,

I’ve been spending the last few days reproducing and stress-testing different ROGII notebooks, and a few competition-specific failure modes stood out. Sharing them here in a public-safe way, without private blend weights, hidden-test assumptions, or anything that would turn this into a copy-paste solution.

The first thing I learned the hard way is that “valid-looking” is not always valid for this competition. Since this is notebook-only and hidden reruns may not have the same sample shape as the visible test files, I now treat any static visible-test CSV writer as unsafe. My current sanity check is simple: the notebook should rebuild predictions from /kaggle/input/competitions/..., write a root-level submission.csv, preserve ids by merge rather than row position, and pass row/order/value checks after the Kaggle run. This caught more issues than model-level debugging did.

Another surprisingly useful check has been comparing candidate submissions against each other on the visible rows. A few notebooks that look quite different in code can collapse into almost identical predictions, especially in the physical / PF-style family. On the other side, a branch that is extremely far away from every known reasonable output is often not “diverse” in a helpful way; it may just be broken, using the wrong state, or writing something misaligned. Pairwise distance is not a leaderboard metric, of course, but it is a good smoke test before spending official submissions.

The most common bug pattern I’ve seen is post-processing hiding an earlier problem. For example, GR NaNs, interpolation gaps, smoothing, nan_to_num, and positional writing can combine into a submission that has the right shape but is geologically meaningless. So I’ve started checking per-well ranges, continuity near the prediction start, obvious jumps, and whether the result still makes sense before and after smoothing. In this task, a smooth wrong curve can be more dangerous than a noisy one because it looks reassuring.

On the modeling side, my read is that a single global offset/state is not expressive enough. The useful direction seems closer to segmented alignment: GR self-correlation along the lateral, typewell/time-warp style matching, neighboring-well/dip consistency, and PF/beam/Viterbi-like path search. Tree models and blends help, but the best candidates seem to respect the geometry/state structure rather than treating this as a plain tabular regression problem.

Curious if others have run into similar practical issues: hidden-compatible notebook structure, duplicate-looking PF/physical branches, GR alignment checks, or post-processing that improves the score but makes the well-level behavior less believable.

Warmly,
Jiayi
https://github.com/Jah-yee

No comments
