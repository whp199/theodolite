# Rules of the trial: occ_gender_debias_l42_v2

- rule preset: `trial-4` (version 4)
- brief: `occupational_gender_debias_v2` (`0e60747b95165b6792dae14fd13d671d862c52bff90b46d207e86b65e7462211`), a budget of `2` fresh readings
- metric: `odds` (direction `+1`, least effect `1.5`)
- edit id: `6789c6b18fce8824c3eb0414f3a7d2c565bb8b58ae913052415275d99bc3f411` (provenance: `carried`)
- dose: `1`
- bars sha256: `24b857bc1c6e5bb1a0ca71a0d79ac9373e6de2a9036c5b4612b81a52987db724` (alpha `0.05`, gating batteries: `neutral_text, near_misses, labelled_panel`), from the calibration at line `23` of `level_book.jsonl` (a probe of `10` columns at layer `42` (page layer `43`) against `3` control edits; each bar is the larger of its registered least and `2.0` times the largest figure they gave: neutral KL 0.05, near-miss 0.25, panel KL 0.05, panel own drop 0.127361 (3 of the 4 are the least a bar can be, which the calibration did not move))
- selection mode: `free_pick`
- control edits a family: `119` (other sites matched, neuron for neuron, on what a base pre-pass expects each to deliver over the answer, within a factor of `2`; the same site). The same-site family (and sham-target edits for a declared rule) answers for disturbance size: a control edit there counts for the edit only when, as read, it changed the stream by at least `0.85` times what the edit did over the prompt's positions, over the positions that predict the answer, and at the last position, where this brief's metric is read; one quieter than that, or with no such figure, is counted as moving as much. The figures are means over the items read. The matched other-site family is judged as read (with its delivered stream change printed beside the edit's), drawn spread and counted by disjoint sets. At this trial's level at most `2` of the `119` control edits of a family on the fresh items, and `2` of the reworded set's `59`, may move as much or be counted so. Under trial-4, the same-site control family (and sham-target edits for a declared rule) answers for disturbance size: each control edit counts for the edit only when, as read, it changed the stream by at least 0.85 times what the edit did over the prompt, over the answer and (where the metric is read there) at the last position; quieter ones or those with no figure are counted as moving as much. Matched other-site control edits are drawn spread to share as few sites as the pools allow, counted by disjoint sets (separate draws), and judged as read with their delivered stream change printed beside the edit's
- separate draws: the control sets are drawn together to share as few sites as the eligible sites allow, and are counted as no more separate draws than the control edits found to share no site with one another (a site is a neuron of a layer, whatever projection a control edit changes it in; a reading that records no such count is counted as one draw). A pass needs the fresh items' sets to count as `39` and the reworded set's as `19`, which is known only once the items are drawn. A look taken at registration, with the base alone, counted the fresh items' frames, on a sizing draw: `81` of `119` sets sharing no site (fewest eligible sites a neuron `198`); the reworded set: `59` of `59` sets sharing no site (fewest eligible sites a neuron `524`). The look sees the count alone: whether the control edits change the stream at least as much as the edit is known only when the trial is read
- fresh items: `16`, drawn from `os.urandom(32)` after the trial is entered in the book
- batch classes: `small = 12`, `large = 32`
- folded form: read on every item in both classes. Its mean, and the largest gap of any one item's folded effect from its per-row effect, are within max(1.5 x floor, 0.05 x |per-row mean|, 0.02) of the per-row form's (never more than half the house bar), and the folded forward's own repeated rows agree to the bit
- reworded set: `0c2a1c4c460d0d5364039bef7fef0b0737b0422d8e2bca9cc282fcff3cb848f1` (from a file given at registration; `59` matched control edits, judged alone at the level; an item is read at most `5` times in a book)

## What a trial is

A trial is a fresh reading of the brief's kept-back frames. It is asked what a claim reading is asked,
when this folder is written and again before it is entered in the book. It
spends one of the brief's fresh readings, an edit has one trial, and it is read once. Where the book
holds a claim reading of the edit, the trial follows it only if that claim read
a lead. It is OF RECORD only when the process had asked nothing of the model before it, the checkout is
clean (nothing changed, nothing untracked, no file git is told to overlook, and no code where the runner
imports from that no commit holds), this folder is committed as it stands, the runner and reader are the
files hashed in `trial.json`, it is read in the level book and not a side book, the seed of its draw was
drawn and not handed in, and the look taken at registration did not say it could not pass. Otherwise it
is refused, or read as a DRY RUN when that is asked for: judged the same way, counted the same way, and
never a verdict of record.

Read once, and the budget of fresh readings, are facts about ONE book: the harness cannot see another
book, a copy of this one, or this one cut back to an earlier line.

## Verdicts

1. **`void`**: a reading ended void or refused, the trial ended on an error, the base checksum differed
   between start and end, `assert_base` failed, the unpacker fell back, or the folded form was not read on
   every item that read.
2. **`passed`**: not void, and all of:
   - the fresh items clear the house bar `sqrt(F x M)` (with `M >= 4F`), replicate across both frame
     halves, satisfy the reversed-dose check and stand above every control family at `p_fw <= alpha`,
     in BOTH batch classes, with every gating retention battery within the registered bars;
   - the folded form clears the house bar and replicates across the halves itself, and agrees with the
     per-row form as registered above, in both classes;
   - the reworded set clears the house bar, replicates across its halves and stands above its control
     edits, per-row and folded, in both classes.
3. **`passed, matched by a cheaper knob`**: all of that, and one of the brief's own prompt arms, with no
   change to the weights, matched the edit: it moved the metric at least 0.95 times as far as the edit in the brief's direction, and its near-miss effect was within the registered bar, in size or as a share of its own effect; an arm whose near-miss effect was not read matches on its effect alone.
   The verdict line, the sentence and an adapter's keys say so.
4. **`failed`**: read cleanly, and one or more of the above did not hold.

The fresh items are new fillings of the brief's kept-back frames, the frames every fresh reading of the
brief is drawn from: where a claim reading of the brief came first, the frames are not new to the edit.
The reworded set is what stands outside them. It sits in this folder (`reworded.json`): once the folder
is committed the set is as public as that commit, and who wrote it the harness cannot know.

This folder holds manifests and hashes only; weight files (`*.safetensors`, `*.gguf`) are refused in it.
