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
| 3 | Pgen prediction | pending |

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

Pending. Both targets are cached and the splits are fixed, so the remaining work is the
model arms themselves.

Measured reference point for the speed question: exact OLGA Pgen runs at **43
sequences/second** per core (23.4 ms/sequence), and `Pgen_1mm` at **2.7/second**
(370 ms/sequence). The committed pgen notebook reports about 11,000 sequences/second for
the neural predictor on CPU, so the speed argument is roughly two orders of magnitude —
this benchmark will quantify it against the frozen-embedding baselines.

## Environment

Results in section 2 were produced on the ctlab Slurm cluster: GTX 1080 Ti,
torch 2.6.0+cu124, Python 3.11. Section 1 timings are from an 8-core CPU (Ryzen 7 8845HS).
Peak-memory figures for section 3 will be resident set size via `psutil`, not
`torch.cuda.max_memory_allocated`, and the device is recorded with every measurement.

Encoder checkpoints: `esm2_t6_8M_UR50D` (fair-esm), `wukevin/tcr-bert` (HuggingFace),
SCEPTR default variant (`sceptr` 1.2.0).
