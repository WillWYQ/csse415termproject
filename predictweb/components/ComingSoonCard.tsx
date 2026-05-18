"use client";

import { Lock } from "@phosphor-icons/react";

interface ComingSoonCardProps {
  title: string;
  description: string;
  features: string[];
}

export function ComingSoonCard({ title, description, features }: ComingSoonCardProps) {
  return (
    <div className="relative flex flex-col items-center justify-center min-h-[400px] rounded-2xl border border-dashed border-neutral-300 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-900/50 p-12 text-center">
      <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-neutral-100 dark:bg-neutral-800 mb-6">
        <Lock size={28} weight="duotone" className="text-neutral-400 dark:text-neutral-500" />
      </div>
      <h2 className="text-2xl font-semibold text-neutral-800 dark:text-neutral-200 mb-2">
        {title}
      </h2>
      <p className="text-neutral-500 dark:text-neutral-400 mb-6 max-w-sm">
        {description}
      </p>
      <div className="flex flex-wrap gap-2 justify-center">
        {features.map((f) => (
          <span
            key={f}
            className="px-3 py-1 text-xs rounded-full border border-neutral-200 dark:border-neutral-700 text-neutral-500 dark:text-neutral-400"
          >
            {f}
          </span>
        ))}
      </div>
      <p className="mt-8 text-xs text-neutral-400 dark:text-neutral-600">
        Model training in progress — check back soon
      </p>
    </div>
  );
}
