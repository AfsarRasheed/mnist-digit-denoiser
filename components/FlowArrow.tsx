interface FlowArrowProps {
  vertical?: boolean;
  className?: string;
}

export default function FlowArrow({ vertical, className = "" }: FlowArrowProps) {
  return (
    <svg
      className={`h-3.5 w-3.5 flex-shrink-0 text-muted-2 ${vertical ? "rotate-90" : ""} ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14m0 0-5-5m5 5-5 5" />
    </svg>
  );
}
