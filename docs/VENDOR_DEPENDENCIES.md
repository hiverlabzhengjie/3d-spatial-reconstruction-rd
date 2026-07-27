# Vendor Dependencies

## Depth Anything 3

| Field | Value |
|---|---|
| Local path | `Depth-Anything-3-main/` |
| Upstream | `https://github.com/ByteDance-Seed/Depth-Anything-3` |
| Local snapshot files | 161 |
| Local snapshot bytes | 25,297,499 |
| Aggregate SHA-256 | `683cad1fec1186cd2a22f2b6d083b73d4c83c7ab1140f45ba24876612bc51d43` |
| Git treatment | Unmodified local vendor dependency; ignored by project Git |

The source arrived as a snapshot without its original `.git` metadata, so an
upstream commit hash cannot be asserted. Before an experiment that uses DA3:

1. verify that the local path exists;
2. recompute and compare the fingerprint;
3. record the exact model repository and revision/cache identity separately;
4. keep all Apple MPS compatibility changes in project-owned adapters.

The aggregate fingerprint was produced from the sorted per-file SHA-256 output:

```text
find Depth-Anything-3-main -type f -print0 |
  LC_ALL=C sort -z |
  xargs -0 shasum -a 256 |
  shasum -a 256
```

If the vendor snapshot is intentionally upgraded, highlight the reason to the
user, update this record and relevant decision/stage documentation, and commit
the new fingerprint.
