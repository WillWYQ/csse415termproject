// Prediction logic for 2nd-round interview model (C1 Random Forest, 86.2% accuracy).
// Preprocessing + ONNX inference run entirely client-side via onnxruntime-web.

import { PREPROCESSING_2ND } from "./preprocessing_2nd.js";

export interface FormValues2nd {
  GPA: string;
  Prior_Internships: string;
  Extra_Curricular_Activities: string;
  Networking_Events_Attended: string;
  Months_Searching: string;
  Applications_Submitted: string;
  First_Round_Interviews: string;
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
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
  session = await ort.InferenceSession.create("/model_2nd.onnx");
  return session;
}

function buildFeatureVector(values: FormValues2nd): Float32Array {
  const {
    numericCols,
    scalerMean,
    scalerScale,
    oheCols,
    catCols,
  } = PREPROCESSING_2ND as {
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

  const numericScaled = numericCols.map(
    (col, i) => (parseFloat(raw[col] ?? "0") - scalerMean[i]) / scalerScale[i]
  );

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

export async function predict2nd(values: FormValues2nd): Promise<PredictionResult> {
  const ort = getOrt();
  const sess = await loadSession();

  const features = buildFeatureVector(values);
  const tensor = new ort.Tensor("float32", features, [1, features.length]);
  const results = await sess.run({ float_input: tensor });

  const probabilities = results.probabilities.data as Float32Array;
  const prob1 = probabilities[1];
  const prediction = prob1 >= 0.5 ? 1 : 0;

  return {
    prediction,
    probability: prob1,
    label: prediction === 1 ? "2nd Round Interview" : "No 2nd Round",
  };
}
