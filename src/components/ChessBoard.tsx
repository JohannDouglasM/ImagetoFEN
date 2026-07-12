import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { PieceType } from "../chess/fenBuilder";

const PIECE_UNICODE: Record<PieceType, string> = {
  K: "♔",
  Q: "♕",
  R: "♖",
  B: "♗",
  N: "♘",
  P: "♙",
  k: "♚",
  q: "♛",
  r: "♜",
  b: "♝",
  n: "♞",
  p: "♟",
};

const FILE_LABELS = ["a", "b", "c", "d", "e", "f", "g", "h"];

type Props = {
  position: (PieceType | null)[][]; // 8x8, row 0 = rank 8
  onSquarePress?: (row: number, col: number) => void;
  selectedSquare?: { row: number; col: number } | null;
  size?: number;
  /** 8x8 model confidence per square; squares below `confidenceThreshold` get flagged */
  confidence?: number[][] | null;
  confidenceThreshold?: number;
};

export default function ChessBoard({
  position,
  onSquarePress,
  selectedSquare,
  size = 320,
  confidence,
  confidenceThreshold = 0.85,
}: Props) {
  // RN borders are inside the view's width — subtract them or the 8th square
  // wraps onto the next line and the board renders as a staircase.
  const BORDER = 2;
  const squareSize = (size - 2 * BORDER) / 8;

  return (
    <View style={[styles.board, { width: size, height: size }]}>
      {position.map((rank, row) => (
        <View key={row} style={styles.rank}>
          {rank.map((piece, col) => {
          const isLight = (row + col) % 2 === 0;
          const isSelected = selectedSquare?.row === row && selectedSquare?.col === col;
          const isUncertain =
            !isSelected && (confidence?.[row]?.[col] ?? 1) < confidenceThreshold;

          return (
            <TouchableOpacity
              key={`${row}-${col}`}
              style={[
                styles.square,
                {
                  width: squareSize,
                  height: squareSize,
                  backgroundColor: isSelected
                    ? "#e94560"
                    : isLight
                      ? "#f0d9b5"
                      : "#b58863",
                },
                isUncertain && styles.uncertain,
              ]}
              onPress={() => onSquarePress?.(row, col)}
              activeOpacity={0.7}
            >
              {piece && (
                <Text
                  style={[
                    styles.piece,
                    { fontSize: squareSize * 0.7 },
                  ]}
                >
                  {PIECE_UNICODE[piece]}
                </Text>
              )}
              {row === 7 && (
                <Text style={[styles.fileLabel, { color: isLight ? "#b58863" : "#f0d9b5" }]}>
                  {FILE_LABELS[col]}
                </Text>
              )}
              {col === 0 && (
                <Text style={[styles.rankLabel, { color: isLight ? "#b58863" : "#f0d9b5" }]}>
                  {8 - row}
                </Text>
              )}
            </TouchableOpacity>
          );
          })}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  board: {
    flexDirection: "column",
    borderWidth: 2,
    borderColor: "#333",
    borderRadius: 4,
    overflow: "hidden",
  },
  rank: {
    flexDirection: "row",
  },
  square: {
    justifyContent: "center",
    alignItems: "center",
  },
  uncertain: {
    borderWidth: 2,
    borderColor: "#ff9f43",
  },
  piece: {
    textAlign: "center",
  },
  fileLabel: {
    position: "absolute",
    bottom: 1,
    right: 3,
    fontSize: 9,
    fontWeight: "bold",
  },
  rankLabel: {
    position: "absolute",
    top: 1,
    left: 3,
    fontSize: 9,
    fontWeight: "bold",
  },
});
