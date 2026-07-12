/**
 * Node validation harness for the pure-TS pipeline (src/ml/pipelineCore.ts).
 *
 * Runs the exact code the app will run (jpeg-js decode → TS resize/Canny/warp →
 * ONNX models) on the extracted chessred2k val sample and reports corner error
 * and cell accuracy, to be compared against training/verify_onnx_pipeline.py.
 *
 * Usage: node scripts/validate_pipeline.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";
import * as ort from "onnxruntime-node";
import type { RGBAImage, Point } from "../src/ml/pipelineCore.ts";
import {
  CORNER_INPUT_SIZE,
  BOARD_INPUT_SIZE,
  resizeRGBA,
  cornerInputTensor,
  coordsToPoints,
  sortConvex,
  warpRGBA,
  rot90RGBA,
  boardInputTensor,
  decodeBoardLogits,
  pickBestRotation,
} from "../src/ml/pipelineCore.ts";
import type { BoardPrediction } from "../src/ml/pipelineCore.ts";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const CATEGORY_NAME_TO_CLASS_ID: Record<string, number> = {
  "white-pawn": 10, "white-rook": 12, "white-knight": 9, "white-bishop": 7,
  "white-queen": 11, "white-king": 8,
  "black-pawn": 3, "black-rook": 5, "black-knight": 2, "black-bishop": 0,
  "black-queen": 4, "black-king": 1,
  empty: 6,
};
const EMPTY = 6;

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.name.endsWith(".jpg")) out.push(p);
  }
  return out.sort();
}

async function main() {
  const ann = JSON.parse(fs.readFileSync(path.join(REPO, "annotations.json"), "utf8"));
  const cats: Record<number, string> = {};
  for (const c of ann.categories) cats[c.id] = c.name;
  const pathToId: Record<string, number> = {};
  for (const im of ann.images) pathToId[im.path] = im.id;
  const cornerGT: Record<number, Record<string, [number, number]>> = {};
  for (const c of ann.annotations.corners) cornerGT[c.image_id] = c.corners;
  const pieceGT: Record<number, Record<string, number>> = {};
  for (const p of ann.annotations.pieces) {
    (pieceGT[p.image_id] ??= {})[p.chessboard_position] =
      CATEGORY_NAME_TO_CLASS_ID[cats[p.category_id] ?? "empty"] ?? EMPTY;
  }

  const cornerSess = await ort.InferenceSession.create(
    path.join(REPO, "assets/models/corner_unet.onnx")
  );
  const boardSess = await ort.InferenceSession.create(
    path.join(REPO, "assets/models/whole_board.onnx")
  );

  const files = walk(path.join(REPO, "valsample"));
  const accs: number[] = [];
  const oracleAccs: number[] = [];
  const cornerErrs: number[] = [];
  let okPick = 0;

  console.log(`${"image".padEnd(20)} corner_err%  picked_k     acc  oracle`);
  for (const file of files) {
    const rel = path.relative(path.join(REPO, "valsample"), file);
    const imageId = pathToId[rel];

    const raw = jpeg.decode(fs.readFileSync(file), { useTArray: true, formatAsRGBA: true });
    const full: RGBAImage = { data: raw.data, width: raw.width, height: raw.height };

    // App path: downscale to max side 1024 first
    const scale = 1024 / Math.max(full.width, full.height);
    const img = resizeRGBA(full, Math.round(full.width * scale), Math.round(full.height * scale));

    // Stage 1: corners
    const corner384 = resizeRGBA(img, CORNER_INPUT_SIZE, CORNER_INPUT_SIZE);
    const cornerFeed = new ort.Tensor("float32", cornerInputTensor(corner384), [
      1, 2, CORNER_INPUT_SIZE, CORNER_INPUT_SIZE,
    ]);
    const cornerOut = await cornerSess.run({ input: cornerFeed });
    const coords = cornerOut.coords.data as Float32Array;
    const pts = coordsToPoints(coords, img.width, img.height);

    // Corner localization error vs GT (min over cyclic rolls, in full-res coords)
    const gt = cornerGT[imageId];
    const gtPts: Point[] = [gt.top_left, gt.top_right, gt.bottom_right, gt.bottom_left];
    const diag = Math.hypot(full.width, full.height);
    let bestErr = Infinity;
    for (let k = 0; k < 4; k++) {
      let e = 0;
      for (let i = 0; i < 4; i++) {
        const p = pts[(i + k) % 4];
        e += Math.hypot(p[0] / scale - gtPts[i][0], p[1] / scale - gtPts[i][1]);
      }
      bestErr = Math.min(bestErr, e / 4 / diag);
    }
    cornerErrs.push(bestErr);

    // Stage 2: warp + 4 rotations
    const sorted = sortConvex(pts);
    const warped = warpRGBA(img, sorted, BOARD_INPUT_SIZE);
    const candidates: BoardPrediction[] = [];
    for (let k = 0; k < 4; k++) {
      const rot = rot90RGBA(warped, k);
      const feed = new ort.Tensor("float32", boardInputTensor(rot), [
        1, 3, BOARD_INPUT_SIZE, BOARD_INPUT_SIZE,
      ]);
      const out = await boardSess.run({ input: feed });
      candidates.push(decodeBoardLogits(out.logits.data as Float32Array));
    }
    const picked = pickBestRotation(candidates);

    // Ground-truth labels (canonical: row 0 = rank 8)
    const labels = new Uint8Array(64).fill(EMPTY);
    for (const [pos, cid] of Object.entries(pieceGT[imageId] ?? {})) {
      const row = 7 - (pos.charCodeAt(1) - 49);
      const col = pos.charCodeAt(0) - 97;
      labels[row * 8 + col] = cid;
    }
    const accOf = (c: (typeof candidates)[number]) => {
      let n = 0;
      for (let i = 0; i < 64; i++) if (c.classes[i] === labels[i]) n++;
      return n / 64;
    };
    const acc = accOf(candidates[picked]);
    const oracle = Math.max(...candidates.map(accOf));
    accs.push(acc);
    oracleAccs.push(oracle);
    if (acc === oracle) okPick++;

    console.log(
      `${path.basename(rel).padEnd(20)} ${(bestErr * 100).toFixed(3).padStart(10)}%  ` +
        `${String(picked).padStart(8)} ${acc.toFixed(4).padStart(7)} ${oracle.toFixed(4).padStart(7)}` +
        (acc === oracle ? "" : "  MISPICK")
    );
  }

  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  console.log("-".repeat(64));
  console.log(`mean corner err: ${(mean(cornerErrs) * 100).toFixed(3)}% of diagonal`);
  console.log(`mean cell acc: ${mean(accs).toFixed(4)} (oracle ${mean(oracleAccs).toFixed(4)}), picked-best ${okPick}/${files.length}`);
}

main();
