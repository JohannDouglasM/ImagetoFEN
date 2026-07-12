/**
 * On-device chess board recognition using ONNX Runtime.
 *
 * Two-stage pipeline (validated against the training pipeline in
 * scripts/validate_pipeline.ts — keep both in sync):
 *
 * 1. Corner detection: U-Net dual-head on 384×384 gray+Canny → 4 board corners
 * 2. Perspective warp to 256×256 (pure JS)
 * 3. Whole-board classifier (ResNet-34 + cell attention) on 4 rotations of the
 *    warped board; the orientation with the highest mean per-cell confidence
 *    wins (the corner model localizes corners precisely but does not reliably
 *    know which corner is a8).
 *
 * All image math lives in pipelineCore.ts, shared with the Node harness.
 */

import { InferenceSession, Tensor } from "onnxruntime-react-native";
import { Asset } from "expo-asset";
import * as ImageManipulator from "expo-image-manipulator";
import { decode as decodeJpeg } from "jpeg-js";
import { SquareResult, PieceType } from "../chess/fenBuilder";
import {
  RGBAImage,
  Point,
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
  BoardPrediction,
} from "./pipelineCore";

/** Max side of the working image the pipeline samples from. */
const WORK_SIZE = 1024;

export interface BoardCandidate {
  /** 0–3: how many 90° CCW rotations of the warped board this prediction saw */
  rotation: number;
  squares: SquareResult[];
  /** 8×8 softmax confidence of the predicted class per cell */
  confidence: number[][];
  meanConfidence: number;
}

export interface PipelineResult {
  /** All 4 orientation candidates, sorted best-first by mean confidence. */
  candidates: BoardCandidate[];
  /** Detected board corners in ORIGINAL photo pixel coords (visual order TL,TR,BR,BL). */
  corners: Point[];
  originalWidth: number;
  originalHeight: number;
}

let cornerSession: InferenceSession | null = null;
let boardSession: InferenceSession | null = null;

async function loadSession(
  cached: InferenceSession | null,
  moduleRef: number
): Promise<InferenceSession> {
  if (cached) return cached;
  const [asset] = await Asset.loadAsync(moduleRef);
  if (!asset.localUri) throw new Error("Failed to load model asset");
  return InferenceSession.create(asset.localUri);
}

async function loadCornerModel(): Promise<InferenceSession> {
  cornerSession = await loadSession(
    cornerSession,
    require("../../assets/models/corner_unet.onnx")
  );
  return cornerSession;
}

async function loadBoardModel(): Promise<InferenceSession> {
  boardSession = await loadSession(
    boardSession,
    require("../../assets/models/whole_board.onnx")
  );
  return boardSession;
}

/** Decode a base64 JPEG string into an RGBA image. */
function base64ToRGBA(base64: string): RGBAImage {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const decoded = decodeJpeg(bytes, { useTArray: true, formatAsRGBA: true });
  return { data: decoded.data, width: decoded.width, height: decoded.height };
}

function predictionToCandidate(pred: BoardPrediction, rotation: number): BoardCandidate {
  const squares: SquareResult[] = [];
  const confidence: number[][] = [];
  for (let row = 0; row < 8; row++) {
    confidence.push([]);
    for (let col = 0; col < 8; col++) {
      const i = row * 8 + col;
      squares.push({ row, col, piece: CLASS_TO_PIECE[pred.classes[i]] as PieceType | null });
      confidence[row].push(pred.confidence[i]);
    }
  }
  return { rotation, squares, confidence, meanConfidence: pred.meanConfidence };
}

/**
 * Run the full two-stage inference pipeline.
 * Progress: 8 coarse steps (decode, corners, warp, 4× classify, done).
 */
export async function runFullPipeline(
  imageUri: string,
  onProgress?: (done: number, total: number) => void
): Promise<PipelineResult> {
  const TOTAL = 8;
  onProgress?.(0, TOTAL);

  // Downscale to the working size on the native side, then decode once in JS.
  // ImageManipulator applies EXIF orientation, so pixels come out upright.
  const probe = await ImageManipulator.manipulateAsync(imageUri, [], {
    format: ImageManipulator.SaveFormat.JPEG,
  });
  const origW = probe.width;
  const origH = probe.height;
  const scale = Math.min(1, WORK_SIZE / Math.max(origW, origH));
  const resized = await ImageManipulator.manipulateAsync(
    imageUri,
    [{ resize: { width: Math.round(origW * scale), height: Math.round(origH * scale) } }],
    { format: ImageManipulator.SaveFormat.JPEG, compress: 0.92, base64: true }
  );
  const img = base64ToRGBA(resized.base64!);
  onProgress?.(1, TOTAL);

  // Stage 1: corners
  const cornerModel = await loadCornerModel();
  const corner384 = resizeRGBA(img, CORNER_INPUT_SIZE, CORNER_INPUT_SIZE);
  const cornerFeed = new Tensor("float32", cornerInputTensor(corner384), [
    1, 2, CORNER_INPUT_SIZE, CORNER_INPUT_SIZE,
  ]);
  const cornerOut = await cornerModel.run({ input: cornerFeed });
  const coords = cornerOut.coords.data as Float32Array;
  const pts = sortConvex(coordsToPoints(coords, img.width, img.height));
  onProgress?.(2, TOTAL);

  // Stage 2: warp + classify all 4 orientations
  const warped = warpRGBA(img, pts, BOARD_INPUT_SIZE);
  onProgress?.(3, TOTAL);

  const boardModel = await loadBoardModel();
  const predictions: BoardPrediction[] = [];
  for (let k = 0; k < 4; k++) {
    const rot = rot90RGBA(warped, k);
    const feed = new Tensor("float32", boardInputTensor(rot), [
      1, 3, BOARD_INPUT_SIZE, BOARD_INPUT_SIZE,
    ]);
    const out = await boardModel.run({ input: feed });
    predictions.push(decodeBoardLogits(out.logits.data as Float32Array));
    onProgress?.(4 + k, TOTAL);
  }

  const bestK = pickBestRotation(predictions);
  const order = [bestK, ...[0, 1, 2, 3].filter((k) => k !== bestK)];
  const candidates = order
    .map((k) => predictionToCandidate(predictions[k], k))
    .sort((a, b) => b.meanConfidence - a.meanConfidence);

  // Corners back in original photo coordinates (for saved training annotations)
  const cornersOrig: Point[] = pts.map(([x, y]) => [x / scale, y / scale]);

  onProgress?.(TOTAL, TOTAL);
  return { candidates, corners: cornersOrig, originalWidth: origW, originalHeight: origH };
}
