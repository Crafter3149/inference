import React from "react";

interface CardProps {
  title?: string;
  badge?: string | number;
  children: React.ReactNode;
  className?: string;
}

export function Card({ title, badge, children, className = "" }: CardProps) {
  return (
    <div className={`bg-surface-card border border-border rounded-xl p-5 ${className}`}>
      {title && (
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
            {title}
          </h3>
          {badge !== undefined && (
            <span className="px-2 py-0.5 text-xs bg-accent/20 text-accent-hover rounded-full">
              {badge}
            </span>
          )}
        </div>
      )}
      {children}
    </div>
  );
}
