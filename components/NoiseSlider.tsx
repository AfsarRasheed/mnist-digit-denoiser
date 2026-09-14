"use client";

import { MIN_SIGMA, MAX_SIGMA, TRAINING_SIGMA } from "@/lib/ml/noise";

interface NoiseSliderProps {
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}

function pctOf(v: number) {
  return ((v - MIN_SIGMA) / (MAX_SIGMA - MIN_SIGMA)) * 100;
}

export default function NoiseSlider({ value, onChange, disabled }: NoiseSliderProps) {
  const fillPct = pctOf(value);
  const defaultPct = pctOf(TRAINING_SIGMA);

  return (
    <div className="w-full max-w-xs">
      <div className="mb-2 flex items-baseline justify-between">
        <label htmlFor="noise-sigma" className="text-xs text-muted">
          Noise level
        </label>
        <span className="font-mono text-sm font-medium text-foreground">σ = {value.toFixed(2)}</span>
      </div>

      <div className="relative flex h-4 items-center">
        <input
          id="noise-sigma"
          type="range"
          min={MIN_SIGMA}
          max={MAX_SIGMA}
          step={0.05}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="slider focus-ring relative z-10"
          style={{
            background: `linear-gradient(to right, var(--accent) ${fillPct}%, var(--border-strong) ${fillPct}%)`,
          }}
        />
        <div
          className="pointer-events-none absolute top-1/2 h-2 w-px -translate-y-1/2 bg-foreground/25"
          style={{ left: `${defaultPct}%` }}
          title={`Training default: σ = ${TRAINING_SIGMA.toFixed(2)}`}
        />
      </div>

      <div className="mt-1.5 flex justify-between text-[11px] text-muted-2">
        <span>Light</span>
        <span>Default σ {TRAINING_SIGMA.toFixed(2)}</span>
        <span>Heavy</span>
      </div>
    </div>
  );
}
