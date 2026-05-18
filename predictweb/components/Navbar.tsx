"use client";

import Link from "next/link";
import { BriefcaseMetal, ArrowSquareOut } from "@phosphor-icons/react";
import { ThemeToggle } from "./ThemeToggle";

export function Navbar() {
  return (
    <header className="fixed top-0 inset-x-0 z-50 flex items-center justify-between px-6 py-4 border-b border-neutral-200/60 dark:border-neutral-800/60 bg-white/80 dark:bg-neutral-950/80 backdrop-blur-md">
      {/* Logo */}
      <Link href="/" className="flex items-center gap-2 font-semibold text-neutral-900 dark:text-white">
        <BriefcaseMetal size={22} weight="fill" className="text-violet-500" />
        <span className="tracking-tight">Job Predictor</span>
      </Link>

      {/* Right actions */}
      <div className="flex items-center gap-3">
        {/* Models dropdown placeholder — links added as models are trained */}
        <nav className="hidden sm:flex items-center gap-1 text-sm text-neutral-600 dark:text-neutral-400">
          <span className="px-3 py-1.5 rounded-full bg-violet-50 dark:bg-violet-950 text-violet-700 dark:text-violet-300 font-medium text-xs">
            Offer Received ✓
          </span>
          <span className="px-3 py-1.5 rounded-full text-neutral-400 dark:text-neutral-600 text-xs cursor-not-allowed">
            1st Interview (soon)
          </span>
          <span className="px-3 py-1.5 rounded-full text-neutral-400 dark:text-neutral-600 text-xs cursor-not-allowed">
            2nd Interview (soon)
          </span>
        </nav>

        <a
          href="https://career.yueqiao.dev"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden sm:flex items-center gap-1 text-sm text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white transition-colors"
        >
          career.yueqiao.dev
          <ArrowSquareOut size={14} />
        </a>

        <ThemeToggle />
      </div>
    </header>
  );
}
