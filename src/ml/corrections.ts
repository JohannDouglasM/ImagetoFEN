/**
 * Saved-correction store: every position the user verifies or fixes becomes a
 * labeled training sample (photo + FEN + detected corners). This is the data
 * flywheel that will close the domain gap on real-world boards.
 *
 * Layout under <documents>/corrections/:
 *   <fen-with-colons>_<timestamp>.jpg    photo (FEN encoded in the filename,
 *                                        same convention as assets/*.jpeg)
 *   <fen-with-colons>_<timestamp>.json   corners + metadata
 */

import { Directory, File, Paths } from "expo-file-system";
import * as Sharing from "expo-sharing";
import type { Point } from "./pipelineCore";

export interface CorrectionMeta {
  fenPosition: string;
  corners: Point[]; // TL,TR,BR,BL visual order, original photo pixel coords
  imageWidth: number;
  imageHeight: number;
  savedAt: string;
  /** true if the user edited at least one square before saving */
  userEdited: boolean;
}

function correctionsDir(): Directory {
  const dir = new Directory(Paths.document, "corrections");
  try {
    dir.create({ intermediates: true, idempotent: true });
  } catch {
    // already exists
  }
  return dir;
}

export async function saveCorrection(
  imageUri: string,
  meta: CorrectionMeta
): Promise<{ photoUri: string }> {
  const dir = correctionsDir();
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const base = `${meta.fenPosition.replace(/\//g, ":")}_${stamp}`;

  const photo = new File(dir, `${base}.jpg`);
  new File(imageUri).copy(photo);

  const json = new File(dir, `${base}.json`);
  json.write(JSON.stringify(meta, null, 2));

  return { photoUri: photo.uri };
}

export function countCorrections(): number {
  try {
    return correctionsDir()
      .list()
      .filter((e) => e.name.endsWith(".jpg")).length;
  } catch {
    return 0;
  }
}

/** Share the most recent correction photo (FEN label is in the filename). */
export async function shareLatestCorrection(): Promise<boolean> {
  const photos = correctionsDir()
    .list()
    .filter((e): e is File => e instanceof File && e.name.endsWith(".jpg"))
    .sort((a, b) => a.name.localeCompare(b.name));
  if (!photos.length) return false;
  if (!(await Sharing.isAvailableAsync())) return false;
  await Sharing.shareAsync(photos[photos.length - 1].uri);
  return true;
}
