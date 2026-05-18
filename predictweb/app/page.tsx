"use client";

import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { OfferForm } from "@/components/OfferForm";
import { SecondRoundForm } from "@/components/SecondRoundForm";
import { ComingSoonCard } from "@/components/ComingSoonCard";
import { AnimatedTabs } from "@/components/ui/tabs";
import { BackgroundBeams } from "@/components/ui/background-beams";

const TABS = [
  { label: "🎯 Offer Received", value: "offer" },
  { label: "🔁 2nd Interview", value: "second" },
  { label: "1st Interview", value: "first" },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState("offer");

  return (
    <div className="relative min-h-screen flex flex-col">
      {/* Animated background (dark mode only via CSS) */}
      <div className="absolute inset-0 dark:block hidden pointer-events-none">
        <BackgroundBeams />
      </div>

      {/* Light mode subtle dot grid */}
      <div
        className="absolute inset-0 dark:hidden pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage: "radial-gradient(circle, #000 1px, transparent 1px)",
          backgroundSize: "24px 24px",
        }}
      />

      <Navbar />

      <main className="relative flex-1 flex flex-col items-center pt-28 pb-16 px-4">
        {/* Hero */}
        <div className="text-center mb-10 max-w-2xl">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-neutral-900 dark:text-white mb-4">
            Will You Get a{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-500 to-cyan-500">
              Job Offer?
            </span>
          </h1>
          <p className="text-neutral-500 dark:text-neutral-400 text-lg">
            Enter your profile below. Our ML model predicts your offer probability
            instantly — right in your browser.
          </p>
        </div>

        {/* Model tabs */}
        <div className="mb-8">
          <AnimatedTabs
            tabs={TABS}
            activeTab={activeTab}
            onChange={setActiveTab}
          />
        </div>

        {/* Content card */}
        <div className="w-full max-w-2xl rounded-2xl border border-neutral-200/80 dark:border-neutral-800/80 bg-white/90 dark:bg-neutral-900/80 backdrop-blur-sm shadow-xl p-8">
          {activeTab === "offer" && <OfferForm />}
          {activeTab === "second" && <SecondRoundForm />}
          {activeTab === "first" && (
            <ComingSoonCard
              title="1st Interview Predictor"
              description="Predict whether you'll land a first-round interview based on your academic and application profile."
              features={["GPA", "University Rating", "Applications Submitted", "Major Category", "Platform"]}
            />
          )}
        </div>

        {/* Footer note */}
        <p className="mt-8 text-xs text-neutral-400 dark:text-neutral-600 text-center">
          Models trained on 100,000 synthetic job-search records · MA 415 Final Project ·{" "}
          <a
            href="https://career.yueqiao.dev"
            target="_blank"
            rel="noopener noreferrer"
            className="underline underline-offset-2 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors"
          >
            career.yueqiao.dev
          </a>
        </p>
      </main>
    </div>
  );
}
