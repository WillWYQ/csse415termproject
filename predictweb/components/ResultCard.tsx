"use client";

import { motion, AnimatePresence } from "motion/react";
import { CheckCircle, XCircle, ArrowCounterClockwise } from "@phosphor-icons/react";
import { SparklesCore } from "./ui/sparkles";
import type { PredictionResult } from "@/lib/predict";

interface ResultCardProps {
  result: PredictionResult;
  onReset: () => void;
}

export function ResultCard({ result, onReset }: ResultCardProps) {
  const isOffer = result.prediction === 1;
  const pct = Math.round(result.probability * 100);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="relative overflow-hidden rounded-2xl border bg-white dark:bg-neutral-900 border-neutral-200 dark:border-neutral-800 p-8"
      >
        {/* Sparkles overlay for positive result */}
        {isOffer && (
          <div className="absolute inset-0 pointer-events-none">
            <SparklesCore
              id="result-sparkles"
              background="transparent"
              minSize={0.4}
              maxSize={1.2}
              particleDensity={50}
              particleColor="#7c3aed"
              className="w-full h-full"
            />
          </div>
        )}

        <div className="relative flex flex-col items-center text-center gap-6">
          {/* Icon */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.15, type: "spring", stiffness: 200 }}
          >
            {isOffer ? (
              <CheckCircle size={64} weight="fill" className="text-violet-500" />
            ) : (
              <XCircle size={64} weight="fill" className="text-neutral-400 dark:text-neutral-600" />
            )}
          </motion.div>

          {/* Label */}
          <div>
            <h2 className={`text-3xl font-bold mb-1 ${isOffer ? "text-violet-600 dark:text-violet-400" : "text-neutral-700 dark:text-neutral-300"}`}>
              {isOffer ? "Likely to Get an Offer!" : "Offer Less Likely"}
            </h2>
            <p className="text-neutral-500 dark:text-neutral-400 text-sm">
              Based on Gradient Boosting (91.3% accuracy on test set)
            </p>
          </div>

          {/* Probability bar */}
          <div className="w-full max-w-sm">
            <div className="flex justify-between text-xs text-neutral-500 dark:text-neutral-400 mb-2">
              <span>Offer probability</span>
              <span className="font-semibold text-neutral-800 dark:text-neutral-200">{pct}%</span>
            </div>
            <div className="h-2.5 w-full rounded-full bg-neutral-100 dark:bg-neutral-800 overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${isOffer ? "bg-violet-500" : "bg-neutral-400"}`}
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ delay: 0.3, duration: 0.7, ease: "easeOut" }}
              />
            </div>
          </div>

          {/* Confidence note */}
          <p className="text-xs text-neutral-400 dark:text-neutral-600 max-w-xs">
            {isOffer
              ? "Your profile shows strong indicators for receiving a job offer."
              : "Consider increasing applications, networking events, or interview rounds to improve your chances."}
          </p>

          {/* Reset button */}
          <button
            onClick={onReset}
            className="flex items-center gap-2 text-sm text-neutral-500 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-neutral-200 transition-colors mt-2"
          >
            <ArrowCounterClockwise size={16} />
            Try different inputs
          </button>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
