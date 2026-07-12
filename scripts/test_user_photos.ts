/**
 * Run the TS pipeline on the 5 user photos in assets/ (ground-truth FEN is
 * encoded in each filename with ':' for '/'). Measures the real-world,
 * out-of-distribution accuracy the app will show on the user's own board.
 *
 * Usage: node scripts/test_user_photos.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import jpeg from "jpeg-js";
import * as ort from "onnxruntime-node";
import type { RGBAImage, BoardPrediction } from "../src/ml/pipelineCore.ts";
import {
  CORNER_INPUT_SIZE,
  BOARD_INPUT_SIZE,
  CLASS_TO_PIECE,
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

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function fenToLabels(fenPosition: string): Uint8Array {
  const labels = new Uint8Array(64).fill(6);
  const ranks = fenPosition.split("/");
  for (let row = 0; row < 8; row++) {
    let col = 0;
    for (const ch of ranks[row]) {
      if (/\d/.test(ch)) col += Number(ch);
      else labels[row * 8 + col++] = CLASS_TO_PIECE.indexOf(ch);
    }
  }
  return labels;
}

async function main() {
  const files = fs
    .readdirSync(path.join(REPO, "assets"))
    .filter((f) => f.endsWith(".jpeg") && f.includes(":"));

  const cornerSess = await ort.InferenceSession.create(
    path.join(REPO, "assets/models/corner_unet.onnx")
  );
  const boardSess = await ort.InferenceSession.create(
    path.join(REPO, "assets/models/whole_board.onnx")
  );

  const accs: number[] = [];
  for (const f of files) {
    const labels = fenToLabels(f.replace(/\.jpeg$/, "").replace(/\(\d+\)$/, "").replace(/:/g, "/"));
    const raw = jpeg.decode(fs.readFileSync(path.join(REPO, "assets", f)), {
      useTArray: true,
      formatAsRGBA: true,
    });
    const full: RGBAImage = { data: raw.data, width: raw.width, height: raw.height };
    const scale = Math.min(1, 1024 / Math.max(full.width, full.height));
    const img = resizeRGBA(full, Math.round(full.width * scale), Math.round(full.height * scale));

    const c384 = resizeRGBA(img, CORNER_INPUT_SIZE, CORNER_INPUT_SIZE);
    const cOut = await cornerSess.run({
      input: new ort.Tensor("float32", cornerInputTensor(c384), [1, 2, CORNER_INPUT_SIZE, CORNER_INPUT_SIZE]),
    });
    const pts = sortConvex(coordsToPoints(cOut.coords.data as Float32Array, img.width, img.height));
    const warped = warpRGBA(img, pts, BOARD_INPUT_SIZE);

    const candidates: BoardPrediction[] = [];
    for (let k = 0; k < 4; k++) {
      const out = await boardSess.run({
        input: new ort.Tensor("float32", boardInputTensor(rot90RGBA(warped, k)), [1, 3, BOARD_INPUT_SIZE, BOARD_INPUT_SIZE]),
      });
      candidates.push(decodeBoardLogits(out.logits.data as Float32Array));
    }
    const accOf = (c: BoardPrediction) => {
      let n = 0;
      for (let i = 0; i < 64; i++) if (c.classes[i] === labels[i]) n++;
      return n / 64;
    };
    const picked = pickBestRotation(candidates);
    const acc = accOf(candidates[picked]);
    const oracle = Math.max(...candidates.map(accOf));
    accs.push(acc);
    console.log(`${f.slice(0, 40).padEnd(42)} acc=${acc.toFixed(4)} oracle=${oracle.toFixed(4)}`);
  }
  console.log(`\nmean acc on user photos: ${(accs.reduce((a, b) => a + b, 0) / accs.length).toFixed(4)}`);
}

main();
