These two independently written public-example documents are synthetic calibration and held-out fixtures. They contain no private user data. The collector consumes the first 4,096 model tokens; each full document has 160 records.

The Windows captures consumed the exact UTF-8 bytes, including the BOM and CRLF line endings. The `.gitattributes` rule preserves those bytes so the recorded prompt SHA-256 values remain reproducible. Do not normalize these two files when repeating the recorded experiment. They are narrow repetitive fixtures, not a task-accuracy benchmark.
