# IRRM-CODEC baseline benchmark

Comparison of IRRM-CODEC against a small set of baselines on CDR3 sequence
reconstruction and Pgen prediction, using human TRB data from
[Zenodo record 19520535](https://zenodo.org/records/19520535).

One pretrained checkpoint per model. The pretrained encoders are frozen — only the
task heads are trained.

## Status

| issue | scope | state |
| --- | --- | --- |
| 1 | train/validation/test splits | done |
| 2 | sequence reconstruction | done |
| 3 | Pgen prediction | done |

## 1. Dataset

Source files are `trb_background_100k.tsv` and `trb_background_embeddings.parquet`,
both inside `redcea_bg.gz` on the Zenodo record. The embeddings parquet is the only
place TCRemP vectors are published, and it holds exactly 100,000 rows per chain rather
than a large background pool.

The AIRR table arrived clean: no duplicate CDR3 amino-acid sequences, no non-standard
residues, no missing V/J calls, and nothing longer than the model's 40-residue limit.
Two filters still removed rows:

| filter | rows | reason |
| --- | ---: | --- |
| non-functional V gene | 335 | `TRBV21-1`, `TRBV23-1`, `TRBV6-7` are pseudogenes; SCEPTR rejects them |
| Pgen = 0 | 2 | `CASSFL`, `CASTTL` give `log10(0) = -inf`, which has no finite gradient |

Both were dropped for every representation and both targets at once, so all models see
one identical row set. Final dataset: **99,663 unique CDR3 sequences**.

The embeddings parquet has no `clone_id`, so it is aligned to the AIRR table by row
order. Because a silent misalignment would corrupt every downstream result invisibly,
the pipeline asserts it: mean TCRemP CDR3-distance correlates with sequence length at
**r = 0.525**, against **r = 0.008** under a shuffled control. The run aborts below
r = 0.3.

Split: one seeded permutation over de-duplicated sequences, so a CDR3 can appear in only
one split. Nested training subsets are prefixes of the same permutation.

| split | rows |
| --- | ---: |
| train | 79,730 |
| validation | 9,966 |
| test | 9,967 |
| subsets | 1,000 ⊂ 10,000 ⊂ 79,730 |

Both targets are cached: `log10(Pgen)` and `log10(Pgen_1mm)`, computed with mirpy's OLGA
wrapper (1 h 43 min on 8 CPU cores). Verified: no CDR3 overlap between any pair of
splits, and target means agree across splits to within 0.01 for both targets.

Reproduce with `python -m benchmark.prepare_splits`.

## 2. Sequence reconstruction

Every representation is reduced to the same **64-dimensional bottleneck** by a
`StandardScaler` + PCA fitted on the training split only, then decoded by the same
IRRM-CODEC decoder (`InverseModel`, 14.48 M parameters at this input width) with
identical settings: AdamW, lr 1e-3, weight decay 1e-4, batch 256, up to 60 epochs,
early stopping with patience 8. Three seeds per representation (42, 43, 44), 15 runs
total, each on one GTX 1080 Ti.

Decoder inputs are standardized with train-split statistics. Without it a shared
learning rate would act differently on each representation — the projections arrive with
standard deviations from 1.0 to 12.2 — making "identical training settings" true only on
paper. The repo's own forward/inverse trainers normalize their embeddings the same way.

### Results at a 64-dimensional bottleneck

Mean ± standard deviation over three seeds, on the held-out test split.

| representation | source dim | expl. var | exact match | token acc | norm. Levenshtein | within ED ≤ 1 | mean ED | train s | seq/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned one-hot | 1000 | 0.380 | **0.9144** ± 0.0006 | 0.9878 | 0.0106 | 0.9643 | 0.171 | 1792 | 6807 |
| SCEPTR | 64 | 1.000 | **0.9005** ± 0.0081 | 0.9866 | 0.0112 | 0.9586 | 0.176 | 1322 | 6781 |
| TCRemP | 9000 | 0.950 | 0.8017 ± 0.0038 | 0.9784 | 0.0197 | 0.9406 | 0.295 | 1796 | 6968 |
| TCR-BERT | 768 | 0.910 | 0.7700 ± 0.0056 | 0.9644 | 0.0291 | 0.8879 | 0.451 | 1174 | 6864 |
| ESM-2 8M | 320 | 0.974 | 0.6510 ± 0.0014 | 0.9326 | 0.0560 | 0.7692 | 0.849 | 1435 | 6720 |

Seed variance is small — at most ±0.008 on exact match — so gaps larger than about 0.02
are real rather than noise.

### Explained variance does not predict reconstruction quality

The relationship is inverted: aligned one-hot retains the least variance (0.380) and
reconstructs best, while ESM-2 retains almost all of it (0.974) and reconstructs worst.

Explained variance measures how much statistical spread survives the projection, not how
much residue-level information survives. One-hot has no correlations for PCA to exploit,
so its variance is spread evenly and the retained fraction looks low — but the 64
components it keeps describe the dominant axes of sequence variation, which is exactly
what the decoder needs.

The consequence is that explained variance cannot be used to normalize or excuse the
comparison. It is reported here because it is informative about each representation's
redundancy, not because it explains the ranking.

### One-hot is a ceiling, not a peer

The bottleneck equalizes dimensionality but not provenance. Aligned one-hot is a
**reversible encoding of the target itself**, so reconstructing from its PCA projection is
an autoencoding task: compress a sequence, decompress it. The other four invert a lossy
transformation of the sequence into a biological feature space. These are different
problems.

One-hot's 0.9144 should therefore be read as an approximate ceiling — what 64 numbers can
carry about a CDR3 when they encode the string directly. Its value is diagnostic: it shows
the decoder's capacity is not the limiting factor, so the spread among the other four
reflects their representations rather than the measuring instrument.

### Mean pooling handicaps the protein language models

ESM-2 and TCR-BERT emit one vector per residue, and the decoder needs one vector per
sequence. Both are mean-pooled over real residues, excluding BOS/EOS and padding — the
same rule for both, chosen for comparability.

Averaging over positions discards positional information, which is precisely what
sequence reconstruction requires. Their low scores partly measure this pooling choice
rather than the representations themselves. A CLS token, or per-position embeddings
concatenated before PCA, would likely score higher. Under the issue's constraint of one
frozen checkpoint producing one vector per receptor, mean pooling is the standard choice,
but the caveat belongs with the numbers.

### Where IRRM-CODEC wins and loses

**Loses: TCRemP is not the strongest compact representation.** SCEPTR reaches 0.9005
against TCRemP's 0.8017 — about 10 points — and does so while losing nothing to the
projection (explained variance 1.000, since it is natively 64-dimensional) where TCRemP
loses 5%. SCEPTR lands within 1.4 points of the one-hot ceiling, meaning its 64 numbers
are nearly as informative about the sequence as a direct compression of the sequence.
This is a genuine negative result for the claim that TCRemP is a good compact receptor
representation.

**Wins: TCRemP survives aggressive compression, and inversion improves under it.** The
committed multi-chain notebook reports exact match 0.7371 for the inverse model on raw
9000-dimensional TRB embeddings. At 64 dimensions — a 140× reduction — this benchmark
reaches **0.8017**. Compression did not cost accuracy; it appears to have removed noise.
That supports the project's premise that the TCRemP space is invertible and usable as a
design space, and it means the practical form of the model can be far smaller than the
published one.

**Wins: it beats both pretrained protein language models.** TCRemP is ahead of TCR-BERT
by 3 points and ESM-2 8M by 15, though the mean-pooling caveat above applies to both.

**Neutral: inference speed does not separate the representations.** All five run at
6,700–7,000 sequences/second, because the cost is dominated by the shared decoder, not by
the input. Encoder cost is what differs and is paid once (measured separately: one-hot
0.013 ms/sequence, SCEPTR 1.30, ESM-2 2.07, TCR-BERT 10.11 on 8 CPU threads).

Reconstruction quality is not what TCRemP was designed for — it encodes similarity to a
reference panel, for clustering and specificity work. Sequence recoverability is a
property IRRM-CODEC adds on top. The reconstruction benchmark therefore bounds one claim
about the representation; the Pgen benchmark tests the project's speed argument, where
the comparison is against OLGA rather than against other encoders.

Reproduce with `benchmark/slurm/reconstruction_array.sbatch`, then
`python -m benchmark.collect_reconstruction`.

## 3. Pgen prediction

Six arms, both targets, three training sizes. The three frozen-embedding arms share one
regression head (512 → 256 → 1) and one training configuration, so only the representation
differs between them. The two IRRM arms differ only in initialization. Ridge is
deterministic and selects its penalty on validation, which is that arm's equivalent of
early stopping.

Targets are standardized on train statistics and mapped back before any metric, so a
freshly initialized network does not spend its first epochs locating an offset near −6.5
that Ridge fits for free. Reported errors are in log10 units.

Neural arms ran ten seeds each on every cell — three from the main sweep plus seven more,
added because the IRRM arms spread by 0.04–0.08 RMSE across seeds, which is larger than
the effect they are meant to measure. 180 runs in total.

### Results on the full training split

Mean ± standard deviation over seeds, on the held-out test split.

**`log10_pgen_1mm`, train = 79,730**

| arm | RMSE | MAE | R² | Pearson r | Spearman ρ | bias | train s | peak RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| irrm_pretrained | **0.2098** ± 0.0493 | 0.1262 | 0.9822 | 0.9928 | 0.9909 | +0.0606 | 232 | 1575 |
| irrm_scratch | 0.2653 ± 0.0376 | 0.1730 | 0.9724 | 0.9894 | 0.9923 | +0.0609 | 135 | 1574 |
| tcr_bert_mlp | 0.3115 ± 0.0073 | 0.2002 | 0.9627 | 0.9813 | 0.9852 | +0.0028 | 62 | 1671 |
| sceptr_mlp | 0.4560 ± 0.0080 | 0.2636 | 0.9200 | 0.9599 | 0.9754 | +0.0484 | 53 | 1197 |
| esm2_8m_mlp | 0.4794 ± 0.0092 | 0.2868 | 0.9116 | 0.9561 | 0.9721 | +0.0650 | 68 | 1301 |
| tfidf_ridge | 0.5542 | 0.3671 | 0.8819 | 0.9391 | 0.9566 | +0.0035 | 10 | 716 |

**`log10_pgen`, train = 79,730**

| arm | RMSE | MAE | R² | Pearson r | Spearman ρ | bias | train s | peak RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| irrm_pretrained | **0.2909** ± 0.0474 | 0.1846 | 0.9753 | 0.9904 | 0.9876 | +0.1154 | 205 | 1582 |
| irrm_scratch | 0.3358 ± 0.0713 | 0.2225 | 0.9666 | 0.9898 | 0.9904 | +0.0998 | 146 | 1569 |
| tcr_bert_mlp | 0.4104 ± 0.0065 | 0.2641 | 0.9520 | 0.9761 | 0.9794 | +0.0299 | 64 | 1669 |
| sceptr_mlp | 0.5833 ± 0.0035 | 0.3536 | 0.9031 | 0.9524 | 0.9649 | +0.0871 | 67 | 1200 |
| esm2_8m_mlp | 0.6031 ± 0.0126 | 0.3811 | 0.8964 | 0.9475 | 0.9599 | +0.0433 | 66 | 1300 |
| tfidf_ridge | 0.6210 | 0.4145 | 0.8901 | 0.9435 | 0.9563 | +0.0087 | 9 | 719 |

The ordering is identical on both targets and at every training size: the two IRRM arms
lead, TCR-BERT is the best frozen representation, then SCEPTR and ESM-2, with Ridge last.
Per-size tables are in `results/pgen/pgen_summary.md`.

For reference, the committed pgen notebook reports RMSE 0.5914, R² 0.9437 and Pearson
0.9874 on `log10_pgen_1mm`. The pretrained arm here reaches RMSE 0.2098, R² 0.9822 and
Pearson 0.9928 on a different split of the same data.

### Does IRRM pretraining improve Pgen prediction?

Welch's t-test on RMSE across ten seeds per arm. A positive delta means pretraining won.

| target | train | irrm_scratch | irrm_pretrained | Δ RMSE | p |
| --- | --- | ---: | ---: | ---: | ---: |
| log10_pgen | 1k | 0.8379 | 0.8550 | −0.017 | 0.527 |
| log10_pgen | 10k | 0.5052 | 0.5422 | −0.037 | 0.157 |
| log10_pgen | all | 0.3358 | 0.2909 | +0.045 | 0.118 |
| log10_pgen_1mm | 1k | 0.6315 | 0.6569 | −0.025 | 0.307 |
| log10_pgen_1mm | 10k | 0.3957 | 0.4087 | −0.013 | 0.500 |
| **log10_pgen_1mm** | **all** | 0.2653 | **0.2098** | **+0.056** | **0.012** |

**Yes, but narrowly: only on `log10_pgen_1mm` and only with the full training set**, where
pretraining cuts RMSE by 20.9%. Every other cell is indistinguishable from zero, and in
three of the four smaller-data cells the sign is negative.

Two things support the one positive result beyond its p-value. It replicated: an earlier
independent run of the same configuration gave Δ = +0.049 at p = 0.017, and the arms were
retrained from scratch in between. And its direction is consistent with the mechanism —
TCRemP encodes similarity to a panel of reference receptors, and `Pgen_1mm` sums
probability over a sequence's single-mismatch neighbours. Both are neighbourhood
quantities, whereas exact `Pgen` is a property of one sequence alone, and that is the
target where transfer does nothing (p = 0.118).

Against it: six comparisons were made. A Bonferroni correction over all six would require
p < 0.008, which this result does not meet. Restricting to the two full-data comparisons,
which are the primary ones, the threshold is p < 0.025 and it does. The finding should be
read as suggestive and mechanistically plausible rather than firmly established.

### Is the advantage larger in the low-data regime?

**No — the opposite.** The effect appears only at 79,730 training sequences and is absent
at 1k and 10k, where the sign is negative on three of four cells. This inverts the usual
expectation that pretraining pays off most when labelled data is scarce.

The bias column shows the mechanism. At 1k the pretrained arm carries bias +0.31 and
+0.24 against +0.16 and +0.15 for the from-scratch arm. The transferred encoder was fitted
to predict 9000-dimensional TCRemP vectors, and its features push predictions in a
systematic direction that a small training set cannot correct. With enough data the head
overcomes it and the transferred features start to pay.

### Quality against speed

End-to-end inference: string in, log10 Pgen out, including tokenization or encoder passes,
measured on one device at a time over 5,000 test sequences.

| arm | GPU (GTX 1080 Ti) | CPU (Xeon Gold, 8 threads) | RMSE (1mm, all) |
| --- | ---: | ---: | ---: |
| tfidf_ridge | 86,661 seq/s | 82,842 seq/s | 0.5542 |
| **irrm** | **64,814 seq/s** | **3,777 seq/s** | **0.2098** |
| esm2_8m_mlp | 9,393 seq/s | 813 seq/s | 0.4794 |
| tcr_bert_mlp | 4,039 seq/s | 216 seq/s | 0.3115 |
| sceptr_mlp | 2,495 seq/s | 1,265 seq/s | 0.4560 |

**IRRM dominates all three pretrained encoders on both axes at once** — 4.6× to 17.5×
faster on CPU, 6.9× to 26× on GPU, and lower RMSE than any of them. There is no trade-off
to make against the large frozen encoders; they are simply worse in both respects, because
their per-sequence encoding cost dwarfs the regression head that follows it.

The real trade-off is against Ridge, which is 1.3× faster on GPU and 22× on CPU while
carrying 2.6× the error. Ridge is the right choice when throughput matters more than
precision; IRRM when it does not.

Caveat on the SCEPTR row: it is the slowest arm on GPU yet mid-pack on CPU, which suggests
its library evaluates on CPU unless `sceptr.enable_hardware_acceleration()` is called,
which this benchmark does not do. Its GPU figure should be read as a CPU computation plus
transfer overhead, not as a GPU measurement.

### Against OLGA

The practical argument for a neural Pgen predictor is speed against the exact calculation.
Measured on the same 8-core laptop CPU, so the two numbers are directly comparable:

| | sequences/second | IRRM speedup |
| --- | ---: | ---: |
| IRRM | 1,969 | — |
| OLGA exact | 42.7 | **46×** |
| OLGA 1-mismatch | 2.7 | **729×** |

On the cluster GPU IRRM reaches 64,814 sequences/second, but OLGA was not benchmarked on
that node, so the laptop comparison is the honest one to quote.

### Where IRRM-CODEC wins and loses

**Wins: accuracy at every training size, on both targets.** IRRM leads all four baselines
in all six cells. On the full split it reaches R² 0.982 and Pearson 0.993 for
`log10_pgen_1mm`, against 0.963 for the best frozen representation.

**Wins: the quality–speed trade-off against pretrained encoders.** Faster and more accurate
than frozen ESM-2, TCR-BERT and SCEPTR simultaneously. For this task the large pretrained
encoders contribute nothing.

**Wins: two to three orders of magnitude over OLGA**, which is the argument the model exists
to make.

**Partial: pretraining helps only in one corner.** The sequence-to-TCRemP initialization
pays off on `log10_pgen_1mm` at full data (−20.9% RMSE) and nowhere else. Anyone adopting
it should expect a modest, target-specific gain, not a general one.

**Loses: seed stability.** The IRRM arms vary by 0.04–0.08 RMSE across seeds where the
frozen arms vary by 0.004–0.015 — roughly tenfold. Validation loss oscillates rather than
descending smoothly, so early stopping fires at inconsistent epochs. The learning rate is
constant at 1e-3 with no decay; a schedule would likely reduce this and is the clearest
next improvement.

**Loses: throughput against a trivial baseline.** TF-IDF + Ridge is 22× faster on CPU. If
an application needs coarse Pgen estimates at maximum rate, k-mers remain the right tool.

Reproduce with `benchmark/slurm/pgen_array.sbatch` and
`benchmark/slurm/pgen_seeds_array.sbatch`, then `python -m benchmark.collect_pgen` and
`python -m benchmark.benchmark_speed`.

## Limitations

- **Mean pooling** for ESM-2 and TCR-BERT discards positional information, which penalizes
  them most on reconstruction. A CLS token or per-position features would likely score
  higher.
- **Aligned one-hot is not a peer baseline** in section 2: it encodes the target
  reversibly, so it bounds the task rather than competing in it.
- **GPU training is not bit-reproducible.** Re-running a configuration with the same seed
  gives slightly different results, because cuDNN selects algorithms at runtime and some
  backward kernels accumulate non-deterministically. Conclusions here rest on ten-seed
  means rather than single runs.
- **One chain, one species.** Everything is human TRB. Nothing here shows the ordering
  holds for TRA or for other loci.
- **One bottleneck width.** Reconstruction was compared at 64 dimensions. Projections at
  32 and 128 are cached in `data/benchmark/trb/bottleneck/` but were not trained.
- **Six statistical comparisons** were made for the pretraining question; only one is
  significant, and it does not survive a Bonferroni correction over all six.

## Environment

Sections 2 and 3 were produced on the ctlab Slurm cluster: GTX 1080 Ti, torch 2.6.0+cu124,
Python 3.11. Section 1 timings and the OLGA comparison are from an 8-core CPU
(Ryzen 7 8845HS); the CPU speed column in section 3 is from a cluster Xeon Gold node.
Peak memory is peak resident set size via `getrusage`, not
`torch.cuda.max_memory_allocated`, and the device is recorded with every measurement.

Encoder checkpoints: `esm2_t6_8M_UR50D` (fair-esm), `wukevin/tcr-bert` (HuggingFace),
SCEPTR default variant (`sceptr` 1.2.0).
