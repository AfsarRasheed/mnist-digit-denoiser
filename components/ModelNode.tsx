interface ModelNodeProps {
  active?: boolean;
  vertical?: boolean;
  className?: string;
}

/**
 * Compact visual stand-in for "the corrupted image is passed through the
 * trained autoencoder." It only ever shows two truthful states — idle and
 * actively processing — never fabricated internals (no layer/activation
 * visualization, since the API doesn't expose any).
 */
export default function ModelNode({ active, vertical, className = "" }: ModelNodeProps) {
  return (
    <div className={`flex items-center gap-2 ${vertical ? "flex-col" : "flex-row"} ${className}`}>
      <Connector vertical={vertical} />
      <div
        className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 transition-colors ${
          active ? "border-accent/40 bg-accent-soft" : "border-border-strong bg-surface-2"
        }`}
      >
        <span className={`relative flex h-2 w-2 flex-shrink-0 rounded-full ${active ? "bg-accent" : "bg-muted-2"}`}>
          {active && <span className="absolute inset-0 animate-ping rounded-full bg-accent opacity-60" />}
        </span>
        <span className={`whitespace-nowrap text-[11px] font-medium ${active ? "text-accent" : "text-muted"}`}>
          Autoencoder
        </span>
      </div>
      <Connector vertical={vertical} />
    </div>
  );
}

function Connector({ vertical }: { vertical?: boolean }) {
  return (
    <svg
      className={`h-3.5 w-3.5 flex-shrink-0 text-muted-2 ${vertical ? "rotate-90" : ""}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14m0 0-5-5m5 5-5 5" />
    </svg>
  );
}
