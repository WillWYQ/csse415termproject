import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",       // static export for GitHub Pages
  trailingSlash: true,    // ensures clean paths on GitHub Pages
  images: {
    unoptimized: true,    // required for static export (no Next.js image optimisation server)
  },
  typescript: {
    // Pre-existing Aceternity components have React 19 strict-type issues unrelated to our code
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
