# TODO

> Superseded by `CLAUDE.md` as of 2026-04-28. Items below are legacy from the corner-detector era; they're either done or no longer the bottleneck. Keeping for historical context.

1. ~~If possible, reverse engineer the corner locations for the new 500 images of the green board (chess-dataset) and add them to our dataset.~~ Done — present as `chess_dataset_recovered` in `annotations.json`.
2. ~~Test if we can drop the square detection input channel (ch2 heatmap) — if performance holds, inference goes from ~30s to <1s.~~ Obsoleted by the whole-board pipeline (single forward pass per board).
3. Try heatmap output with peak-finding/argmax instead of direct coordinate regression for corner detection. — Not done, but corner detector is already at 0.25% mean error; not the bottleneck.
