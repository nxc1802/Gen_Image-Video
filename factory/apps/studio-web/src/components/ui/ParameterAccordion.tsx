"use client";

import React, { useState } from "react";

interface ParameterAccordionProps {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

export function ParameterAccordion({
  title,
  defaultOpen = false,
  children,
}: ParameterAccordionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border-y border-hairline">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="w-full py-2.5 flex items-center justify-between text-left"
      >
        <span className="text-[11px] font-mono uppercase tracking-[0.06em] text-ink-muted font-medium">
          {title}
        </span>
        <span className="font-mono text-[14px] text-ink font-semibold leading-none select-none">
          {open ? "\u2212" : "+"}
        </span>
      </button>
      {open && <div className="pb-4 pt-1 space-y-3">{children}</div>}
    </div>
  );
}
