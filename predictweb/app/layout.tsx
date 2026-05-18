import type { Metadata } from "next";
import { Geist } from "next/font/google";
import { ThemeProvider } from "next-themes";
import Script from "next/script";
import "./globals.css";
import { cn } from "@/lib/utils";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Job Predictor — Will You Get an Offer?",
  description:
    "Predict your job offer probability using machine learning. Enter your profile and get an instant prediction powered by a Gradient Boosting model trained on 100k job seekers.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className={cn(geistSans.variable)}>
      <body className="min-h-full flex flex-col font-sans antialiased bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 transition-colors">
        {/* Load ONNX Runtime Web from CDN — avoids bundler WASM issues */}
        <Script
          src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js"
          strategy="beforeInteractive"
        />
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
