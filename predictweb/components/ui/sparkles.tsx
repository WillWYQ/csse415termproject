"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { cn } from "@/lib/utils";

interface Particle {
  x: number;
  y: number;
  size: number;
  speedX: number;
  speedY: number;
  opacity: number;
  opacityDelta: number;
  color: string;
}

interface SparklesCoreProps {
  id?: string;
  background?: string;
  minSize?: number;
  maxSize?: number;
  particleDensity?: number;
  className?: string;
  particleColor?: string;
}

export function SparklesCore({
  id,
  background = "transparent",
  minSize = 0.4,
  maxSize = 1,
  particleDensity = 100,
  className,
  particleColor = "#FFFFFF",
}: SparklesCoreProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  const initParticles = useCallback(
    (width: number, height: number) => {
      const count = Math.floor((width * height) / (10000 / particleDensity));
      particlesRef.current = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        size: Math.random() * (maxSize - minSize) + minSize,
        speedX: (Math.random() - 0.5) * 0.4,
        speedY: (Math.random() - 0.5) * 0.4 - 0.2,
        opacity: Math.random(),
        opacityDelta: (Math.random() * 0.01 + 0.003) * (Math.random() > 0.5 ? 1 : -1),
        color: particleColor,
      }));
    },
    [minSize, maxSize, particleDensity, particleColor]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ width, height });
        canvas.width = width;
        canvas.height = height;
        initParticles(width, height);
      }
    });

    resizeObserver.observe(canvas.parentElement ?? canvas);
    return () => resizeObserver.disconnect();
  }, [initParticles]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || dimensions.width === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particlesRef.current.forEach((p) => {
        p.x += p.speedX;
        p.y += p.speedY;
        p.opacity += p.opacityDelta;

        if (p.opacity <= 0 || p.opacity >= 1) {
          p.opacityDelta = -p.opacityDelta;
          p.opacity = Math.max(0, Math.min(1, p.opacity));
        }

        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        ctx.save();
        ctx.globalAlpha = p.opacity;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();

        // Draw a small star shape for larger particles
        if (p.size > 0.8) {
          const starSize = p.size * 2;
          ctx.fillStyle = p.color;
          ctx.globalAlpha = p.opacity * 0.5;
          for (let i = 0; i < 4; i++) {
            const angle = (i / 4) * Math.PI * 2;
            ctx.fillRect(
              p.x + Math.cos(angle) * starSize - 0.5,
              p.y + Math.sin(angle) * starSize - 0.5,
              1,
              1
            );
          }
        }

        ctx.restore();
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [dimensions]);

  return (
    <canvas
      ref={canvasRef}
      id={id}
      className={cn("absolute inset-0 w-full h-full", className)}
      style={{ background }}
    />
  );
}
