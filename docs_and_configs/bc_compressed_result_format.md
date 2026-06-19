# BC Compressed Result Format And Reader

This document records the current `.bccmp` final compressed result format and
reader behavior.

Primary code:

```text
native_core/include/BCCompressedResult.h
native_core/src/BCCompressedResult.cpp
engine_core/BookReaderBC.py
```

## 1. Scope

`.bccmp` is a final archive/reader format for BC exact results. It is not the
exact frontier format used during solve. Resume and later solve layers continue
to use exact `.bcpos + .bcsuc` checkpoints.

Current implemented builders:

```text
compress_exact_layer_to_result(position_path, success_path, lut, output_path, options)
compress_in_memory_layer_to_result(position_reader, success_reader, output_path, options)
```

The production runner invokes `compress_exact_layer_to_result(...)` when
`compress=true`. Direct streaming from FamilyChain cell finalization into the
compressed builder is a future optimization; the current production path
compresses exact files.

## 2. File Shape

The file is one `.bccmp`:

```text
Header
Data blocks
Axis table
CellDir table
BucketBlockDir table
ValueBlockDir table
```

The header records:

```text
magic/version/header_bytes
dtype/value_size/row_width
layer_sum/family_count/family_unit/axis_base_coord
position key/rank payload metadata
bucket/value block target and hard-cap sizes
cell_count/live_rows/success_value_count
offset/count pairs for data, axis, cell dir, bucket dir, value dir
original position/success byte sizes
position metadata fingerprint
```

Directory tables are uncompressed and small relative to layer data. Data blocks
are independently xz-compressed with `compress_xz_block_native`.

## 3. Cell Directory

One `CellDirEntry` exists per position cell:

```text
value_base
bucket_block_begin
bucket_block_count
success_rows
flags
```

`value_base` is the first global success value index for that cell in compressed
value space. It is not a byte offset.

## 4. Bucket Blocks

Bucket blocks are ordered by cell and key range. A `BucketBlockDirEntry`
contains:

```text
cid
first_bucket_index
bucket_count
first_success_row
first_key
last_key
compressed_offset
compressed_size
raw_size
```

Raw bucket-block layout:

```text
BucketBlockRawHeader
bucket keys array
bucket success_row_offsets array
local_rank_payload_offsets array
concatenated rank payload slices
```

The rank payload slices reuse the BC position prefix256 + bitmap format; the
compressed format does not split inside a bucket bitmap.

Default block size options:

```text
bucket_block_raw_target_bytes = 32 KiB
bucket_block_raw_hard_cap_bytes = 128 KiB
compression_level = 1
```

## 5. Value Blocks

Value blocks store typed success payload bytes. A `ValueBlockDirEntry` contains:

```text
first_value_index
value_count
value_size
compressed_offset
compressed_size
raw_size
```

Default block size options:

```text
value_block_raw_target_bytes = 4 KiB
value_block_raw_hard_cap_bytes = 16 KiB
compression_level = 1
```

## 6. Point Reader

`BCCompressedResult::PointReader` keeps header, axis, cell directory, and small
directory metadata in memory. A cold lookup reads/decompresses only:

```text
one bucket block
one value block
```

Lookup flow:

```text
canonical/encode board with BCLut and the stored BC axis
derive cid/key/rank
read CellDir[cid]
binary-search that cell's BucketBlockDir range by key
decompress one bucket block
find bucket key inside the raw block
test bitmap bit and compute local rank using prefix256 + local popcount
value_index = cell.value_base + local_success_row * row_width + lane
binary-search ValueBlockDir by value_index
decompress one value block
return raw_value_bits and numeric_value
```

Reader results include raw and compressed bytes for the touched bucket/value
blocks so tests can assert small-granularity lookup.

## 7. Random State Sampling

The compressed reader supports sampling a random stored row through
`sample_cold(...)` / `PointReader::sample(...)`. The Python reader uses this for
the frontend default/random-state button. Exact `.bcpos + .bcsuc` sampling also
exists for non-compressed BC results.

## 8. Python Dispatch

`BookReaderBC` can read:

```text
compressed .bccmp layers
exact .bcpos + .bcsuc layers
```

It normalizes integer success values for frontend display:

```text
uint32 / 4,000,000,000
uint64 / configured uint64 scale
float/double as stored numeric value
```

This normalization belongs in the Python reader/front-end presentation layer,
not in the native cold lookup result.
