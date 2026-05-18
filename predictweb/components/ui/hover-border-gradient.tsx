"use client";

import React, {
  useState,
  useEffect,
  useRef,
  ElementType,
  ComponentPropsWithoutRef,
} from "react";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";

type Direction = "TOP" | "LEFT" | "BOTTOM" | "RIGHT";

function getDirection(
  ev: React.MouseEvent<HTMLElement>,
  obj: HTMLElement
): Direction {
  const { width, height, top, left } = obj.getBoundingClientRect();
  const x = ev.clientX - left - width / 2;
  const y = ev.clientY - top - height / 2;

  if (Math.abs(x) > Math.abs(y)) {
    return x > 0 ? "RIGHT" : "LEFT";
  } else {
    return y > 0 ? "BOTTOM" : "TOP";
  }
}

const directionMap: Record<Direction, number> = {
  TOP: 0,
  RIGHT: 90,
  BOTTOM: 180,
  LEFT: 270,
};

type HoverBorderGradientProps<T extends ElementType = "button"> = {
  as?: T;
  containerClassName?: string;
  className?: string;
  duration?: number;
  clockwise?: boolean;
  children?: React.ReactNode;
} & Omit<ComponentPropsWithoutRef<T>, "as" | "containerClassName" | "className" | "duration" | "clockwise" | "children">;

export function HoverBorderGradient<T extends ElementType = "button">({
  as,
  containerClassName,
  className,
  duration = 1,
  clockwise = true,
  children,
  ...props
}: HoverBorderGradientProps<T>) {
  const Tag = (as ?? "button") as ElementType;
  const containerRef = useRef<HTMLElement>(null);
  const [hovered, setHovered] = useState(false);
  const [angle, setAngle] = useState(0);

  useEffect(() => {
    if (!hovered) {
      const interval = setInterval(() => {
        setAngle((prev) => (clockwise ? prev + 1 : prev - 1) % 360);
      }, duration * 10);
      return () => clearInterval(interval);
    }
  }, [hovered, duration, clockwise]);

  const handleMouseEnter = (ev: React.MouseEvent<HTMLElement>) => {
    if (containerRef.current) {
      const dir = getDirection(ev, containerRef.current);
      setAngle(directionMap[dir]);
    }
    setHovered(true);
  };

  const handleMouseLeave = () => {
    setHovered(false);
  };

  return (
    <Tag
      ref={containerRef}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={cn(
        "relative flex rounded-full border content-center bg-black/20 hover:bg-black/10 transition duration-500 dark:bg-white/20 items-center flex-col flex-nowrap gap-10 h-min justify-center overflow-visible p-px decoration-clone w-fit",
        containerClassName
      )}
      {...props}
    >
      <div
        className={cn(
          "w-auto z-10 bg-black px-4 py-2 rounded-[inherit]",
          className
        )}
      >
        {children}
      </div>
      <motion.div
        className="absolute inset-0 overflow-hidden rounded-[inherit]"
        style={{ filter: "blur(2px)" }}
      >
        <motion.div
          className="absolute inset-[-100%]"
          animate={{ rotate: angle }}
          transition={{
            ease: "linear",
            duration: hovered ? 0 : duration,
          }}
          style={{
            background: `conic-gradient(from 0deg at 50% 50%, #6344F5, #18CCFC, #AE48FF, #6344F5)`,
          }}
        />
      </motion.div>
      <div className="absolute inset-[2px] rounded-[inherit] bg-black" />
    </Tag>
  );
}
