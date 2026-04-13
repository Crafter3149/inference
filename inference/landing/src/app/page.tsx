"use client";

import React, { useState, useEffect } from "react";
import { OverviewPage } from "./dashboard/pages/OverviewPage";
import { ModelsPage } from "./dashboard/pages/ModelsPage";
import { InferencePage } from "./dashboard/pages/InferencePage";
import { WorkflowPage } from "./dashboard/pages/WorkflowPage";

type Page = "overview" | "models" | "inference" | "workflow";

const NAV_ITEMS: { key: Page; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "models", label: "Models" },
  { key: "inference", label: "Inference" },
  { key: "workflow", label: "Workflow" },
];

function hashToPage(hash: string): Page {
  const h = hash.replace("#/", "").replace("#", "") || "overview";
  if (["overview", "models", "inference", "workflow"].includes(h)) return h as Page;
  return "overview";
}

export default function Home() {
  const [page, setPage] = useState<Page>("overview");

  useEffect(() => {
    setPage(hashToPage(window.location.hash));
    const onHash = () => setPage(hashToPage(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="min-h-screen bg-surface flex">
      {/* Sidebar */}
      <aside className="w-56 bg-surface-card border-r border-border flex flex-col h-screen fixed left-0 top-0">
        <div className="p-4 border-b border-border">
          <h1 className="text-lg font-bold text-white">AIE Dashboard</h1>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {NAV_ITEMS.map((item) => (
            <a
              key={item.key}
              href={item.key === "overview" ? "#/" : `#/${item.key}`}
              className={`flex items-center px-3 py-2 rounded-lg text-sm transition-colors ${
                page === item.key
                  ? "bg-accent text-white"
                  : "text-gray-400 hover:bg-surface-hover hover:text-white"
              }`}
            >
              {item.label}
            </a>
          ))}
        </nav>
        <div className="p-3 border-t border-border">
          <a
            href="/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-400 hover:bg-surface-hover hover:text-white transition-colors"
          >
            <span className="text-xs">&#8599;</span> API Docs
          </a>
        </div>
      </aside>

      {/* Main content */}
      <main className="ml-56 flex-1 p-6">
        {page === "overview" && <OverviewPage />}
        {page === "models" && <ModelsPage />}
        {page === "inference" && <InferencePage />}
        {page === "workflow" && <WorkflowPage />}
      </main>
    </div>
  );
}
