// onnxruntime-react-native carries a leftover Expo-module config in its
// package.json, so Expo SDK 55's autolinking classifies it as a conflicting
// Expo module and skips React Native linking entirely — leaving
// NativeModules.Onnxruntime null at runtime. Declaring platforms explicitly
// here forces it through the RN autolinking path.
module.exports = {
  dependencies: {
    "onnxruntime-react-native": {
      platforms: {
        android: {
          packageImportPath: "import ai.onnxruntime.reactnative.OnnxruntimePackage;",
          packageInstance: "new OnnxruntimePackage()",
        },
        ios: {},
      },
    },
  },
};
