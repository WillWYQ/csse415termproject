"use client";

import React from "react";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";

export type Tab = {
  label: string;
  value: string;
};

interface AnimatedTabsProps {
  tabs: Tab[];
  activeTab: string;
  onChange: (value: string) => void;
  className?: string;
}

export function AnimatedTabs({
  tabs,
  activeTab,
  onChange,
  className,
}: AnimatedTabsProps) {
  return (
    <div
      className={cn(
        "flex flex-row items-center justify-start relative rounded-full bg-neutral-800 p-1 gap-1",
        className
      )}
    >
      {tabs.map((tab) => (
        <button
          key={tab.value}
          onClick={() => onChange(tab.value)}
          className={cn(
            "relative z-10 px-4 py-1.5 rounded-full text-sm font-medium transition-colors duration-200 cursor-pointer",
            activeTab === tab.value
              ? "text-black"
              : "text-neutral-400 hover:text-neutral-200"
          )}
        >
          {activeTab === tab.value && (
            <motion.div
              layoutId="animated-tab-pill"
              className="absolute inset-0 rounded-full bg-white"
              transition={{
                type: "spring",
                bounce: 0.2,
                duration: 0.4,
              }}
            />
          )}
          <span className="relative z-10">{tab.label}</span>
        </button>
      ))}
    </div>
  );
}
