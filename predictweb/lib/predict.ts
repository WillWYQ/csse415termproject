// Prediction logic: preprocessing + ONNX inference (client-side only).
// onnxruntime-web is loaded from CDN as a <script> tag (see layout.tsx),
// making it available as window.ort — this sidesteps bundler WASM issues.

import { PREPROCESSING } from "./preprocessing_offer.js";

export interface FormValues {
  GPA: string;
  Prior_Internships: string;
  Extra_Curricular_Activities: string;
  Networking_Events_Attended: string;
  Months_Searching: string;
  Applications_Submitted: string;
  First_Round_Interviews: string;
  Second_Round_Interviews: string;
  University_Rating: string;
  School_Size: string;
  Region: string;
  Major_Category: string;
  Primary_Search_Platform: string;
}

export interface PredictionResult {
  prediction: number;
  probability: number;
  label: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ORT = any;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let session: any = null;

function getOrt(): ORT {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ort = (window as any).ort;
  if (!ort) throw new Error("onnxruntime-web not yet loaded from CDN");
  return ort;
}

async function loadSession() {
  if (session) return session;
  const ort = getOrt();
  // Point WASM files at the same CDN version to avoid cross-origin issues
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
  session = await ort.InferenceSession.create("/model_offer.onnx");
  return session;
}

function buildFeatureVector(values: FormValues): Float32Array {
  const {
    numericCols,
    scalerMean,
    scalerScale,
    oheCols,
    catCols,
  } = PREPROCESSING as {
    numericCols: string[];
    scalerMean: number[];
    scalerScale: number[];
    oheCols: string[];
    catCols: string[];
    catValues: Record<string, string[]>;
    catFirstDropped: Record<string, string>;
    nFeatures: number;
  };

  const raw = values as unknown as Record<string, string>;

  // Standardise numeric features: (x - mean) / std
  const numericScaled = numericCols.map(
    (col, i) => (parseFloat(raw[col] ?? "0") - scalerMean[i]) / scalerScale[i]
  );

  // OHE: for each dummy column name ("Region_South"), check if user selected that value
  const oheValues = oheCols.map((colName) => {
    for (const catCol of catCols) {
      if (colName.startsWith(`${catCol}_`)) {
        const encodedVal = colName.slice(catCol.length + 1);
        return raw[catCol] === encodedVal ? 1 : 0;
      }
    }
    return 0;
  });

  return Float32Array.from([...numericScaled, ...oheValues]);
}

export async function predict(values: FormValues): Promise<PredictionResult> {
  const ort = getOrt();
  const sess = await loadSession();

  const features = buildFeatureVector(values);
  const tensor = new ort.Tensor("float32", features, [1, features.length]);
  const results = await sess.run({ float_input: tensor });

  // ONNX outputs: label (int64[1]), probabilities (float32[1,2])
  const probabilities = results.probabilities.data as Float32Array;
  const prob1 = probabilities[1];
  const prediction = prob1 >= 0.5 ? 1 : 0;

  return {
    prediction,
    probability: prob1,
    label: prediction === 1 ? "Offer Received" : "No Offer",
  };
}
