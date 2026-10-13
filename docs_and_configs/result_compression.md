# Result Compression Policy

With result compression enabled, Classic, EX and EXAD now check whether
compression is worthwhile. Temporary-file compression is unchanged.

## Sampling

Each sampling pass checks the result payload before compression. Four evenly
spaced samples, each at most 256 KiB, bound the total work to 1 MiB per payload.
Samples remain aligned to complete values or records. If their compressed size
is at least 95% of their original size, the payload is stored without compression.
Payloads smaller than 256 KiB use the existing compression path.

The 95% threshold matches BC. EX/EXAD also reuse the decision while the distance
from the sampled layer is less than 10, resampling at a distance of 10 or more.
Reuse requires unchanged dtype, compression level, deletion thresholds and
EXAD row widths. The policy belongs to one solve invocation: restart/resume
creates a fresh policy, and different tables do not share it. File-based,
in-memory, chunked and final cleanup compression share that policy within a run.
Standalone compression calls without a policy sample independently.
Classic continues sampling each payload. Decisions are logged as
`RESULT_COMPRESSION`, including sample sizes or the inherited layer number.
Sampling is a heuristic, not a size guarantee.

## Algorithm Differences

- **EX / EXAD:** Bucket/index compression is unchanged. Only success-value blocks
  switch between compressed and stored encoding. Both file-based and in-memory
  retirement paths use the same rule.
- **Classic:** Trie payloads contain both the lower board bits and success values;
  the actual combined payload is sampled. Trie indices are still generated.
  The final size, including all indices, must save more than 5%; otherwise the
  temporary output is discarded and the source `.book` is retained.
- **AD:** Unchanged. Success values in `.b` files remain uncompressed; eligible
  `.i` indices are compressed without sampling or a size-benefit check. The
  existing minimum index-size condition is retained.
- **BC:** Existing RAW/XZ selection, archive naming and nearby-layer reuse are
  unchanged.

## Compatibility

Stored Classic/EX/EXAD blocks are standard XZ streams containing uncompressed
LZMA2 chunks, created with liblzma's stored-block encoder and retaining CRC64.
They bypass the compression search but keep the existing file headers, versions,
extensions and decoder contract. Old readers can read them without changes.
The small XZ framing overhead remains; stored does not mean a headerless byte copy.

No new layer states, extensions or resume markers are introduced. Mixed layers
remain readable by the current lookup, sampling and resume code. Compression
failure remains an error; it is never interpreted as a successful stored result.
