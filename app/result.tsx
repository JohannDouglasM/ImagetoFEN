import { useState, useEffect, useCallback } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  Dimensions,
  ActivityIndicator,
} from "react-native";
import { useLocalSearchParams } from "expo-router";
import * as Clipboard from "expo-clipboard";
import * as Haptics from "expo-haptics";
import * as Linking from "expo-linking";
import ChessBoard from "../src/components/ChessBoard";
import PieceEditor from "../src/components/PieceEditor";
import {
  PieceType,
  SquareResult,
  buildFullFen,
  buildFenPosition,
  validateFen,
} from "../src/chess/fenBuilder";
import { fenToPgn, lichessAnalysisUrl } from "../src/chess/pgnExporter";
import { runFullPipeline, PipelineResult } from "../src/ml/inference";
import { saveCorrection, countCorrections, shareLatestCorrection } from "../src/ml/corrections";

const FILE_LABELS = ["a", "b", "c", "d", "e", "f", "g", "h"];

export default function ResultScreen() {
  const { imageUri } = useLocalSearchParams<{
    imageUri: string;
  }>();

  const [position, setPosition] = useState<(PieceType | null)[][]>(
    Array.from({ length: 8 }, () => Array(8).fill(null))
  );
  const [confidence, setConfidence] = useState<number[][] | null>(null);
  const [pipeline, setPipeline] = useState<PipelineResult | null>(null);
  const [candidateIndex, setCandidateIndex] = useState(0);
  const [edited, setEdited] = useState(false);
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingStatus, setLoadingStatus] = useState("Analyzing board...");
  const [editSquare, setEditSquare] = useState<{ row: number; col: number } | null>(null);
  const [activeColor, setActiveColor] = useState<"w" | "b">("w");
  const [copied, setCopied] = useState<string | null>(null);

  const screenWidth = Dimensions.get("window").width;
  const boardSize = Math.min(screenWidth - 48, 360);

  useEffect(() => {
    runInference();
  }, []);

  const applyCandidate = (result: PipelineResult, index: number) => {
    const cand = result.candidates[index];
    const board: (PieceType | null)[][] = Array.from({ length: 8 }, () => Array(8).fill(null));
    for (const sq of cand.squares) {
      board[sq.row][sq.col] = sq.piece;
    }
    setPosition(board);
    setConfidence(cand.confidence);
    setCandidateIndex(index);
    setEdited(false);
    setSaved(false);
  };

  const runInference = async () => {
    setLoading(true);
    try {
      if (!imageUri) {
        Alert.alert("Error", "Missing image.");
        setLoading(false);
        return;
      }

      const result = await runFullPipeline(imageUri, (done, total) => {
        setLoadingStatus(done < total ? `Analyzing board… ${done}/${total}` : "Done!");
      });
      setPipeline(result);
      applyCandidate(result, 0);
    } catch (err) {
      console.error("Inference error:", err);
      Alert.alert(
        "Error",
        `Failed to classify the chess position: ${err instanceof Error ? err.message : "Unknown error"}`
      );
    } finally {
      setLoading(false);
    }
  };

  const rotateOrientation = () => {
    if (!pipeline) return;
    if (edited) {
      Alert.alert(
        "Discard edits?",
        "Changing orientation reloads the model's prediction and discards your edits.",
        [
          { text: "Cancel", style: "cancel" },
          {
            text: "Rotate",
            onPress: () =>
              applyCandidate(pipeline, (candidateIndex + 1) % pipeline.candidates.length),
          },
        ]
      );
      return;
    }
    applyCandidate(pipeline, (candidateIndex + 1) % pipeline.candidates.length);
  };

  const getSquareResults = useCallback((): SquareResult[] => {
    const results: SquareResult[] = [];
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        results.push({ row, col, piece: position[row][col] });
      }
    }
    return results;
  }, [position]);

  const fen = buildFullFen(getSquareResults(), { activeColor });
  const validation = validateFen(getSquareResults());
  const pgn = fenToPgn(fen);

  const copyToClipboard = async (text: string, label: string) => {
    await Clipboard.setStringAsync(text);
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    setCopied(label);
    setTimeout(() => setCopied(null), 2000);
  };

  const openInLichess = () => {
    const url = lichessAnalysisUrl(fen);
    Linking.openURL(url);
  };

  const handleSquarePress = (row: number, col: number) => {
    setEditSquare({ row, col });
  };

  const handlePieceSelect = (piece: PieceType | null) => {
    if (!editSquare) return;
    setPosition((prev) => {
      const next = prev.map((r) => [...r]);
      next[editSquare.row][editSquare.col] = piece;
      return next;
    });
    // Clear the uncertainty flag on the square the user just verified
    setConfidence((prev) => {
      if (!prev) return prev;
      const next = prev.map((r) => [...r]);
      next[editSquare.row][editSquare.col] = 1;
      return next;
    });
    setEdited(true);
    setSaved(false);
    setEditSquare(null);
  };

  const handleSave = async () => {
    if (!pipeline || !imageUri) return;
    try {
      await saveCorrection(imageUri, {
        fenPosition: buildFenPosition(getSquareResults()),
        corners: pipeline.corners,
        imageWidth: pipeline.originalWidth,
        imageHeight: pipeline.originalHeight,
        savedAt: new Date().toISOString(),
        userEdited: edited,
      });
      setSaved(true);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      const n = countCorrections();
      Alert.alert(
        "Saved as training data",
        `${n} labeled photo${n === 1 ? "" : "s"} collected. Share this one now?`,
        [
          { text: "Later", style: "cancel" },
          { text: "Share", onPress: () => shareLatestCorrection() },
        ]
      );
    } catch (err) {
      Alert.alert("Error", `Could not save: ${err instanceof Error ? err.message : err}`);
    }
  };

  const uncertainCount = confidence
    ? confidence.flat().filter((c) => c < 0.85).length
    : 0;

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#e94560" />
        <Text style={styles.loadingText}>{loadingStatus}</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.boardContainer}>
        <ChessBoard
          position={position}
          onSquarePress={handleSquarePress}
          selectedSquare={editSquare}
          size={boardSize}
          confidence={confidence}
        />
      </View>

      {uncertainCount > 0 && (
        <Text style={styles.uncertainHint}>
          ⚠ {uncertainCount} square{uncertainCount === 1 ? "" : "s"} the model isn't sure about —
          tap to verify.
        </Text>
      )}

      <TouchableOpacity style={styles.rotateButton} onPress={rotateOrientation}>
        <Text style={styles.rotateText}>
          ↻ Wrong orientation? Rotate board
        </Text>
      </TouchableOpacity>

      {!validation.valid && (
        <View style={styles.warningBox}>
          <Text style={styles.warningTitle}>Position warnings:</Text>
          {validation.errors.map((e, i) => (
            <Text key={i} style={styles.warningText}>
              • {e}
            </Text>
          ))}
        </View>
      )}

      <View style={styles.fenContainer}>
        <Text style={styles.label}>FEN</Text>
        <TouchableOpacity
          style={styles.fenBox}
          onPress={() => copyToClipboard(fen, "FEN")}
        >
          <Text style={styles.fenText} numberOfLines={2}>
            {fen}
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.toggleRow}>
        <Text style={styles.label}>To move:</Text>
        <TouchableOpacity
          style={[styles.toggleButton, activeColor === "w" && styles.toggleActive]}
          onPress={() => setActiveColor("w")}
        >
          <Text style={styles.toggleText}>White</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.toggleButton, activeColor === "b" && styles.toggleActive]}
          onPress={() => setActiveColor("b")}
        >
          <Text style={styles.toggleText}>Black</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => copyToClipboard(fen, "FEN")}
        >
          <Text style={styles.actionText}>
            {copied === "FEN" ? "Copied!" : "Copy FEN"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => copyToClipboard(pgn, "PGN")}
        >
          <Text style={styles.actionText}>
            {copied === "PGN" ? "Copied!" : "Copy PGN"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.lichessButton]}
          onPress={openInLichess}
        >
          <Text style={styles.actionText}>Open in Lichess</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.saveButton, saved && styles.savedButton]}
          onPress={handleSave}
          disabled={saved}
        >
          <Text style={styles.actionText}>
            {saved ? "✓ Saved as training data" : "Position correct? Save as training data"}
          </Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.hint}>
        Tap any square on the board to edit a piece. Saving a verified position helps the
        model learn your board.
      </Text>

      <PieceEditor
        visible={editSquare !== null}
        squareLabel={
          editSquare
            ? `${FILE_LABELS[editSquare.col]}${8 - editSquare.row}`
            : ""
        }
        onSelect={handlePieceSelect}
        onClose={() => setEditSquare(null)}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#16213e",
  },
  content: {
    padding: 24,
    alignItems: "center",
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: "#16213e",
    justifyContent: "center",
    alignItems: "center",
    gap: 16,
  },
  loadingText: {
    color: "#a8a8b8",
    fontSize: 16,
  },
  boardContainer: {
    marginBottom: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  uncertainHint: {
    color: "#ff9f43",
    fontSize: 13,
    marginBottom: 10,
    textAlign: "center",
  },
  rotateButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    marginBottom: 16,
    borderRadius: 8,
    backgroundColor: "#0f3460",
    borderWidth: 1,
    borderColor: "#1a4a80",
  },
  rotateText: {
    color: "#a8c8e8",
    fontSize: 14,
  },
  warningBox: {
    backgroundColor: "#4a2020",
    borderRadius: 10,
    padding: 12,
    width: "100%",
    marginBottom: 16,
  },
  warningTitle: {
    color: "#ff6b6b",
    fontWeight: "bold",
    marginBottom: 4,
  },
  warningText: {
    color: "#ffaaaa",
    fontSize: 13,
  },
  fenContainer: {
    width: "100%",
    marginBottom: 16,
  },
  label: {
    color: "#a8a8b8",
    fontSize: 14,
    marginBottom: 6,
  },
  fenBox: {
    backgroundColor: "#0f3460",
    borderRadius: 10,
    padding: 14,
    borderWidth: 1,
    borderColor: "#1a4a80",
  },
  fenText: {
    color: "#fff",
    fontSize: 13,
    fontFamily: "monospace",
  },
  toggleRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: 20,
    alignSelf: "flex-start",
  },
  toggleButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: "#0f3460",
  },
  toggleActive: {
    backgroundColor: "#e94560",
  },
  toggleText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
  },
  actions: {
    width: "100%",
    gap: 10,
    marginBottom: 20,
  },
  actionButton: {
    backgroundColor: "#0f3460",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#1a4a80",
  },
  lichessButton: {
    backgroundColor: "#4a4a00",
    borderColor: "#6a6a20",
  },
  saveButton: {
    backgroundColor: "#1a4a2e",
    borderColor: "#2a6a4e",
  },
  savedButton: {
    opacity: 0.6,
  },
  actionText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  hint: {
    color: "#666",
    fontSize: 13,
    textAlign: "center",
    marginBottom: 40,
  },
});
