/**
 * Pure-TS core of the two-stage board recognition pipeline.
 *
 * No React Native imports — this module is shared verbatim by the mobile app
 * (src/ml/inference.ts) and the Node validation harness
 * (scripts/validate_pipeline.ts). Every function mirrors the exact
 * preprocessing the models were trained with (cv2 semantics).
 *
 * Pipeline:
 *   RGBA photo (≈1024px)
 *     → bilinear resize 384×384 → gray + Canny(80,200) → corner model → 4 corners
 *     → convex-sort corners → homography warp 256×256
 *     → 4 rotations → whole-board model → confidence-picked orientation
 *     → per-cell class + confidence
 */

export interface RGBAImage {
  data: Uint8Array | Uint8ClampedArray; // RGBA, 4 bytes/pixel
  width: number;
  height: number;
}

export type Point = [number, number];

export const CORNER_INPUT_SIZE = 384;
export const BOARD_INPUT_SIZE = 256;
export const NUM_CLASSES = 13;

// Class layout shared with the whole-board model (alphabetical ImageFolder order)
export const CLASS_TO_PIECE: (string | null)[] = [
  "b", "k", "n", "p", "q", "r", null, "B", "K", "N", "P", "Q", "R",
];

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

/** Bilinear resize of an RGBA image (cv2 INTER_LINEAR pixel-center alignment). */
export function resizeRGBA(src: RGBAImage, dw: number, dh: number): RGBAImage {
  const { data, width: sw, height: sh } = src;
  const out = new Uint8ClampedArray(dw * dh * 4);
  const xRatio = sw / dw;
  const yRatio = sh / dh;
  for (let dy = 0; dy < dh; dy++) {
    const sy = (dy + 0.5) * yRatio - 0.5;
    const y0 = Math.max(0, Math.floor(sy));
    const y1 = Math.min(sh - 1, y0 + 1);
    const fy = Math.min(1, Math.max(0, sy - y0));
    for (let dx = 0; dx < dw; dx++) {
      const sx = (dx + 0.5) * xRatio - 0.5;
      const x0 = Math.max(0, Math.floor(sx));
      const x1 = Math.min(sw - 1, x0 + 1);
      const fx = Math.min(1, Math.max(0, sx - x0));
      const i00 = (y0 * sw + x0) * 4;
      const i01 = (y0 * sw + x1) * 4;
      const i10 = (y1 * sw + x0) * 4;
      const i11 = (y1 * sw + x1) * 4;
      const o = (dy * dw + dx) * 4;
      for (let c = 0; c < 4; c++) {
        const top = data[i00 + c] * (1 - fx) + data[i01 + c] * fx;
        const bot = data[i10 + c] * (1 - fx) + data[i11 + c] * fx;
        out[o + c] = top * (1 - fy) + bot * fy;
      }
    }
  }
  return { data: out, width: dw, height: dh };
}

/** RGBA → uint8 grayscale with cv2 BGR2GRAY weights (R*0.299 + G*0.587 + B*0.114). */
export function rgbaToGrayU8(img: RGBAImage): Uint8Array {
  const { data, width, height } = img;
  const out = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = Math.round(0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]);
  }
  return out;
}

/** Reflect-101 border index (cv2 default): -1 -> 1, n -> n-2. */
function reflect101(i: number, n: number): number {
  if (i < 0) return -i;
  if (i >= n) return 2 * n - i - 2;
  return i;
}

/** Separable 5×5 Gaussian blur on uint8 grayscale, matching cv2.GaussianBlur((5,5), sigma). */
export function gaussianBlur5U8(gray: Uint8Array, w: number, h: number, sigma: number): Uint8Array {
  // cv2.getGaussianKernel(5, sigma)
  const k = new Float64Array(5);
  let sum = 0;
  for (let i = 0; i < 5; i++) {
    const x = i - 2;
    k[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += k[i];
  }
  for (let i = 0; i < 5; i++) k[i] /= sum;

  const tmp = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -2; i <= 2; i++) acc += k[i + 2] * gray[y * w + reflect101(x + i, w)];
      tmp[y * w + x] = acc;
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -2; i <= 2; i++) acc += k[i + 2] * tmp[reflect101(y + i, h) * w + x];
      out[y * w + x] = Math.min(255, Math.max(0, Math.round(acc)));
    }
  }
  return out;
}

/**
 * Canny edge detector matching cv2.Canny(img, low, high) defaults:
 * Sobel 3×3, L1 gradient magnitude, OpenCV's integer NMS quantization,
 * 8-connected hysteresis. Returns 0/255 map.
 */
export function cannyU8(gray: Uint8Array, w: number, h: number, low: number, high: number): Uint8Array {
  const dx = new Int32Array(w * h);
  const dy = new Int32Array(w * h);
  const mag = new Int32Array(w * h);
  const px = (x: number, y: number) => gray[reflect101(y, h) * w + reflect101(x, w)];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const gx =
        (px(x + 1, y - 1) + 2 * px(x + 1, y) + px(x + 1, y + 1)) -
        (px(x - 1, y - 1) + 2 * px(x - 1, y) + px(x - 1, y + 1));
      const gy =
        (px(x - 1, y + 1) + 2 * px(x, y + 1) + px(x + 1, y + 1)) -
        (px(x - 1, y - 1) + 2 * px(x, y - 1) + px(x + 1, y - 1));
      const i = y * w + x;
      dx[i] = gx;
      dy[i] = gy;
      mag[i] = Math.abs(gx) + Math.abs(gy); // L2gradient=false
    }
  }

  // Non-maximum suppression with OpenCV's fixed-point angle quantization
  const TG22 = 13573; // tan(22.5°) * 2^15
  const state = new Uint8Array(w * h); // 0 none, 1 weak, 2 strong
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const m = mag[i];
      if (m <= low) continue;
      const ax = Math.abs(dx[i]);
      const ay = Math.abs(dy[i]) << 15;
      const tg22x = ax * TG22;
      let isMax = false;
      if (ay < tg22x) {
        // horizontal gradient → compare left/right
        if (m > mag[i - 1] && m >= mag[i + 1]) isMax = true;
      } else {
        const tg67x = tg22x + ((ax + ax) << 15);
        if (ay > tg67x) {
          // vertical gradient → compare up/down
          if (m > mag[i - w] && m >= mag[i + w]) isMax = true;
        } else {
          // diagonal
          const s = (dx[i] ^ dy[i]) < 0 ? -1 : 1;
          if (m > mag[i - w - s] && m > mag[i + w + s]) isMax = true;
        }
      }
      if (isMax) state[i] = m > high ? 2 : 1;
    }
  }

  // Hysteresis: BFS from strong pixels through weak ones
  const out = new Uint8Array(w * h);
  const stack: number[] = [];
  for (let i = 0; i < state.length; i++) {
    if (state[i] === 2) {
      out[i] = 255;
      stack.push(i);
    }
  }
  while (stack.length) {
    const i = stack.pop()!;
    const y = Math.floor(i / w);
    const x = i - y * w;
    for (let ny = y - 1; ny <= y + 1; ny++) {
      for (let nx = x - 1; nx <= x + 1; nx++) {
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const j = ny * w + nx;
        if (state[j] === 1 && !out[j]) {
          out[j] = 255;
          stack.push(j);
        }
      }
    }
  }
  return out;
}

/**
 * Build the 2-channel corner-model input tensor [1,2,384,384] from an RGBA
 * image already resized to 384×384: channel 0 = gray/255, channel 1 = Canny/255.
 */
export function cornerInputTensor(resized384: RGBAImage): Float32Array {
  const s = CORNER_INPUT_SIZE;
  const gray = rgbaToGrayU8(resized384);
  const blurred = gaussianBlur5U8(gray, s, s, 1.4);
  const edges = cannyU8(blurred, s, s, 80, 200);
  const out = new Float32Array(2 * s * s);
  for (let i = 0; i < s * s; i++) {
    out[i] = gray[i] / 255;
    out[s * s + i] = edges[i] / 255;
  }
  return out;
}

/**
 * Order 4 points clockwise (in image coordinates) around their centroid,
 * starting from the one nearest the image's top-left. Guarantees a convex,
 * non-crossed quad regardless of board orientation in the photo.
 */
export function sortConvex(pts: Point[]): Point[] {
  const cx = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4;
  const cy = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4;
  const sorted = [...pts].sort(
    (a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx)
  );
  let start = 0;
  let best = Infinity;
  for (let i = 0; i < 4; i++) {
    const s = sorted[i][0] + sorted[i][1];
    if (s < best) {
      best = s;
      start = i;
    }
  }
  return [0, 1, 2, 3].map((i) => sorted[(start + i) % 4]);
}

/** DLT homography from 4 point correspondences (row-major 3×3, h8=1). */
export function computeHomography(src: Point[], dst: Point[]): number[] {
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [sx, sy] = src[i];
    const [dx, dy] = dst[i];
    A.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy]);
    b.push(dx);
    A.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy]);
    b.push(dy);
  }
  const n = 8;
  const aug = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) aug[row][j] -= f * aug[col][j];
    }
  }
  const hm = new Array(9);
  for (let i = 0; i < 8; i++) hm[i] = aug[i][n] / aug[i][i];
  hm[8] = 1;
  return hm;
}

/**
 * Warp the board quad to a square RGBA image (bilinear, black outside),
 * matching cv2.warpPerspective(src→[0,0],[S,0],[S,S],[0,S]).
 * `corners` order: TL, TR, BR, BL of the destination.
 */
export function warpRGBA(src: RGBAImage, corners: Point[], outSize: number): RGBAImage {
  const S = outSize;
  const dst: Point[] = [[0, 0], [S, 0], [S, S], [0, S]];
  const Hinv = computeHomography(dst, corners); // dest pixel → source pixel
  const { data, width: sw, height: sh } = src;
  const out = new Uint8ClampedArray(S * S * 4);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      // cv2.warpPerspective maps integer dest coords (x, y)
      const wd = Hinv[6] * x + Hinv[7] * y + Hinv[8];
      const sx = (Hinv[0] * x + Hinv[1] * y + Hinv[2]) / wd;
      const sy = (Hinv[3] * x + Hinv[4] * y + Hinv[5]) / wd;
      const o = (y * S + x) * 4;
      if (sx < -1 || sy < -1 || sx > sw || sy > sh) {
        out[o + 3] = 255;
        continue; // black
      }
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      for (let c = 0; c < 3; c++) {
        const v00 = sample(data, sw, sh, x0, y0, c);
        const v01 = sample(data, sw, sh, x0 + 1, y0, c);
        const v10 = sample(data, sw, sh, x0, y0 + 1, c);
        const v11 = sample(data, sw, sh, x0 + 1, y0 + 1, c);
        out[o + c] = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy + v11 * fx * fy;
      }
      out[o + 3] = 255;
    }
  }
  return { data: out, width: S, height: S };
}

function sample(
  data: Uint8Array | Uint8ClampedArray,
  w: number,
  h: number,
  x: number,
  y: number,
  c: number
): number {
  if (x < 0 || y < 0 || x >= w || y >= h) return 0;
  return data[(y * w + x) * 4 + c];
}

/** np.rot90(img, k): rotate RGBA image counter-clockwise k×90°. */
export function rot90RGBA(img: RGBAImage, k: number): RGBAImage {
  k = ((k % 4) + 4) % 4;
  if (k === 0) return img;
  const { data, width: w, height: h } = img;
  const out = new Uint8ClampedArray(data.length);
  const dw = k % 2 === 0 ? w : h;
  const dh = k % 2 === 0 ? h : w;
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      let sx: number, sy: number;
      if (k === 1) {
        sy = x;
        sx = w - 1 - y;
      } else if (k === 2) {
        sx = w - 1 - x;
        sy = h - 1 - y;
      } else {
        sy = h - 1 - x;
        sx = y;
      }
      const o = (y * dw + x) * 4;
      const s = (sy! * w + sx!) * 4;
      out[o] = data[s];
      out[o + 1] = data[s + 1];
      out[o + 2] = data[s + 2];
      out[o + 3] = data[s + 3];
    }
  }
  return { data: out, width: dw, height: dh };
}

/** Whole-board input tensor [1,3,256,256]: RGB/255, ImageNet-normalized, CHW. */
export function boardInputTensor(warped256: RGBAImage): Float32Array {
  const s = BOARD_INPUT_SIZE;
  const { data } = warped256;
  const out = new Float32Array(3 * s * s);
  for (let i = 0, p = 0; i < s * s; i++, p += 4) {
    for (let c = 0; c < 3; c++) {
      out[c * s * s + i] = (data[p + c] / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
    }
  }
  return out;
}

export interface BoardPrediction {
  /** class id per cell, row-major 8×8, row 0 = rank 8 (a8 top-left) */
  classes: Uint8Array;
  /** softmax confidence of the argmax class per cell */
  confidence: Float32Array;
  /** mean of per-cell max softmax — used to pick the orientation */
  meanConfidence: number;
}

/** Decode whole-board logits [13,8,8] into per-cell classes + confidences. */
export function decodeBoardLogits(logits: Float32Array): BoardPrediction {
  const cells = 64;
  const classes = new Uint8Array(cells);
  const confidence = new Float32Array(cells);
  let meanConf = 0;
  for (let i = 0; i < cells; i++) {
    let maxV = -Infinity;
    let maxC = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const v = logits[c * cells + i];
      if (v > maxV) {
        maxV = v;
        maxC = c;
      }
    }
    let denom = 0;
    for (let c = 0; c < NUM_CLASSES; c++) denom += Math.exp(logits[c * cells + i] - maxV);
    classes[i] = maxC;
    confidence[i] = 1 / denom;
    meanConf += confidence[i];
  }
  return { classes, confidence, meanConfidence: meanConf / cells };
}

/** Index of the rotation candidate with the highest mean confidence. */
export function pickBestRotation(candidates: BoardPrediction[]): number {
  let best = 0;
  for (let i = 1; i < candidates.length; i++) {
    if (candidates[i].meanConfidence > candidates[best].meanConfidence) best = i;
  }
  return best;
}

/** Corner-model output ([8] normalized coords) → pixel points TL,TR,BR,BL (board-space). */
export function coordsToPoints(coords: Float32Array, width: number, height: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < 4; i++) pts.push([coords[2 * i] * width, coords[2 * i + 1] * height]);
  return pts;
}
