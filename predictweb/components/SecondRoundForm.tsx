"use client";

import { useState } from "react";
import { motion } from "motion/react";
import { PREPROCESSING_2ND } from "@/lib/preprocessing_2nd.js";
import { predict2nd, type FormValues2nd, type PredictionResult } from "@/lib/predict_2nd";
import { ResultCard } from "./ResultCard";
import { HoverBorderGradient } from "./ui/hover-border-gradient";
import { Sparkle } from "@phosphor-icons/react";

const PREPROC = PREPROCESSING_2ND as {
  numericCols: string[];
  catCols: string[];
  catValues: Record<string, string[]>;
};

const NUMERIC_META: Record<string, { label: string; min: number; max: number; step: number; defaultVal: string }> = {
  GPA:                         { label: "GPA",                             min: 2.0, max: 4.0, step: 0.1, defaultVal: "3.2" },
  Prior_Internships:           { label: "Prior Internships",               min: 0,   max: 5,   step: 1,   defaultVal: "1" },
  Extra_Curricular_Activities: { label: "Extracurricular Activities",      min: 0,   max: 11,  step: 1,   defaultVal: "2" },
  Networking_Events_Attended:  { label: "Networking Events Attended",      min: 0,   max: 14,  step: 1,   defaultVal: "3" },
  Months_Searching:            { label: "Months Searching",                min: 1,   max: 12,  step: 1,   defaultVal: "6" },
  Applications_Submitted:      { label: "Applications Submitted",          min: 5,   max: 300, step: 1,   defaultVal: "54" },
  First_Round_Interviews:      { label: "1st Round Interviews",            min: 0,   max: 21,  step: 1,   defaultVal: "2" },
};

const CAT_LABELS: Record<string, string> = {
  University_Rating:        "University Rating",
  School_Size:              "School Size",
  Region:                   "Region",
  Major_Category:           "Major Category",
  Primary_Search_Platform:  "Primary Job Search Platform",
};

function buildDefaults(): FormValues2nd {
  const defaults: Partial<FormValues2nd> = {};
  for (const col of PREPROC.numericCols) {
    (defaults as unknown as Record<string, string>)[col] = NUMERIC_META[col]?.defaultVal ?? "0";
  }
  for (const col of PREPROC.catCols) {
    (defaults as unknown as Record<string, string>)[col] = PREPROC.catValues[col]?.[0] ?? "";
  }
  return defaults as FormValues2nd;
}

export function SecondRoundForm() {
  const [values, setValues] = useState<FormValues2nd>(buildDefaults);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleChange(key: string, val: string) {
    setValues((prev) => ({ ...prev, [key]: val }));
    setResult(null);
    setError(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await predict2nd(values);
      setResult(res);
    } catch (err) {
      setError("Prediction failed. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  if (result) {
    return <ResultCard result={result} onReset={() => setResult(null)} />;
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {/* Numeric inputs */}
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-500 mb-4">
          Academic &amp; Job Search Profile
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          {PREPROC.numericCols.map((col) => {
            const meta = NUMERIC_META[col];
            const val = (values as unknown as Record<string, string>)[col] ?? "0";
            return (
              <div key={col} className="flex flex-col gap-1.5">
                <div className="flex justify-between items-baseline">
                  <label className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                    {meta.label}
                  </label>
                  <span className="text-sm font-semibold text-violet-600 dark:text-violet-400 tabular-nums">
                    {val}
                  </span>
                </div>
                <input
                  type="range"
                  min={meta.min}
                  max={meta.max}
                  step={meta.step}
                  value={val}
                  onChange={(e) => handleChange(col, e.target.value)}
                  className="w-full h-1.5 rounded-full appearance-none cursor-pointer
                    bg-neutral-200 dark:bg-neutral-700
                    accent-violet-500"
                />
                <div className="flex justify-between text-[10px] text-neutral-400 dark:text-neutral-600">
                  <span>{meta.min}</span>
                  <span>{meta.max}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Categorical inputs */}
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-500 mb-4">
          Background &amp; Platform
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          {PREPROC.catCols.map((col) => {
            const options = PREPROC.catValues[col] ?? [];
            const val = (values as unknown as Record<string, string>)[col];
            return (
              <div key={col} className="flex flex-col gap-1.5">
                <label className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
                  {CAT_LABELS[col] ?? col}
                </label>
                <select
                  value={val}
                  onChange={(e) => handleChange(col, e.target.value)}
                  className="w-full px-3 py-2 rounded-lg text-sm
                    bg-neutral-50 dark:bg-neutral-800
                    border border-neutral-200 dark:border-neutral-700
                    text-neutral-800 dark:text-neutral-200
                    focus:outline-none focus:ring-2 focus:ring-violet-500
                    transition-colors"
                >
                  {options.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </div>
            );
          })}
        </div>
      </div>

      {/* Error */}
      {error && (
        <p className="text-sm text-red-500 dark:text-red-400 text-center">{error}</p>
      )}

      {/* Submit */}
      <div className="flex justify-center pt-2">
        <HoverBorderGradient
          containerClassName="rounded-full"
          as="button"
          type="submit"
          disabled={loading}
          className="flex items-center gap-2 px-8 py-3 text-sm font-semibold bg-white dark:bg-black text-neutral-900 dark:text-white disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {loading ? (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 0.8, ease: "linear" }}
              className="w-4 h-4 border-2 border-current border-t-transparent rounded-full"
            />
          ) : (
            <Sparkle size={16} weight="fill" className="text-violet-500" />
          )}
          {loading ? "Predicting…" : "Predict My Chances"}
        </HoverBorderGradient>
      </div>

      {/* Model info */}
      <p className="text-center text-[11px] text-neutral-400 dark:text-neutral-600">
        Random Forest · 86.2% test accuracy · cascade model · runs entirely in your browser
      </p>
    </form>
  );
}
