import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  icons: "/static/icon.png",
  title: "AIE Dashboard",
  description: "AIE Inference Server Dashboard",
  robots: "noindex",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
