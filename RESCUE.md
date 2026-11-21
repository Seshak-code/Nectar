# Rescue branch

The desktop that held Nectar's working tree failed before this work was committed.
This branch rebuilds the files from the Claude Code session's history (every Write, Edit and full Read, replayed in order).
It is a safety net, not the real tree: the SSD copy wins wherever the two differ.

- 88 files rebuilt: 74 exact, the rest listed below.
- The Juice code the session deleted (cmd/, pkg/, docker/, go.mod, go.sum, TERMS.md, assets/) is removed here too.
- `rescue/plan.md` is the session's plan and progress log.
- `rescue/failed-edits.json` lists edits that could not be replayed onto the rebuilt text (file, old_string, new_string). Some are superseded by later edits.

## Not exact

| File | Status |
|---|---|
| `native/nectar-cuda/README.md` | partial (13 edits did not apply) |
| `native/nectar-cuda/client/src/dark_api.rs` | partial (2 edits did not apply) |
| `native/nectar-cuda/client/src/lib.rs` | partial (3 edits did not apply) |
| `native/nectar-cuda/codegen/src/emit.rs` | whitespace-matched |
| `native/nectar-cuda/codegen/src/main.rs` | partial (2 edits did not apply) |
| `native/nectar-cuda/codegen/src/model.rs` | partial (1 edits did not apply) |
| `native/nectar-cuda/install/Install-Borrower.ps1` | partial (1 edits did not apply) |
| `native/nectar-cuda/lending/src/discovery.rs` | partial (1 edits did not apply) |
| `native/nectar-cuda/probe/src/main.rs` | partial (1 edits did not apply) |
| `native/nectar-cuda/protocol/src/lib.rs` | partial (13 edits did not apply) |
| `native/nectar-cuda/server/src/main.rs` | partial (8 edits did not apply) |
| `native/nectar-cuda/server/src/vulkan.rs` | partial (1 edits did not apply) |
| `native/nectar-cuda/svc/src/pipe.rs` | partial (1 edits did not apply) |
| `native/nectar-cuda/trace/src/lib.rs` | partial (3 edits did not apply) |
| `native/nectar-cuda/vk/src/connection.rs` | no full text |
| `native/nectar-cuda/vkwire/src/lib.rs` | no full text |

## Missing entirely

- Files created only by shell commands, which the history doesn't hold: the generated CUDA and Vulkan bindings (rerun the codegen crates), and some Cargo.toml files (for example trace, vk, vkwire, vkprobe, redirect).
- `.gitignore` and `LICENSE.md` changes (LICENSE.md was updated to "Copyright (c) 2026 Sriram"); master's versions are kept.
- Line endings: files are written with the line endings the session recorded, mostly LF.

Do not build from this branch expecting it to compile. Recover the SSD, then diff it against this branch.
