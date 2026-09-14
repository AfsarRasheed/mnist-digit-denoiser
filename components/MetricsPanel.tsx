import type { PredictMetrics } from "@/lib/ml/types";

interface MetricsPanelProps {
  metrics: PredictMetrics | null;
}

interface MetricGroup {
  title: string;
  items: { label: string; value: string }[];
}

function buildGroups(m: PredictMetrics | null): MetricGroup[] {
  return [
    {
      title: "Quality",
      items: [
        { label: "Reconstruction MSE", value: m ? m.mse.toFixed(5) : "—" },
        { label: "PSNR", value: m ? `${m.psnr.toFixed(1)} dB` : "—" },
      ],
    },
    {
      title: "Input",
      items: [
        { label: "Input size", value: m ? m.inputDims : "—" },
        { label: "Noise level", value: m ? `σ ${m.noiseSigma.toFixed(2)}` : "—" },
      ],
    },
    {
      title: "Performance",
      items: [{ label: "Inference time", value: m ? `${m.inferenceTimeMs.toFixed(2)} ms` : "—" }],
    },
    {
      title: "Model",
      items: [{ label: "Parameters", value: m ? m.paramCount.toLocaleString() : "—" }],
    },
  ];
}

export default function MetricsPanel({ metrics }: MetricsPanelProps) {
  const groups = buildGroups(metrics);

  return (
    <div>
      <div className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4">
        {groups.map((group) => (
          <div key={group.title} className="flex flex-col gap-3">
            <p className="eyebrow">{group.title}</p>
            <div className="flex flex-col gap-3">
              {group.items.map((item) => (
                <div key={item.label}>
                  <p className="text-xs text-muted">{item.label}</p>
                  <p className="mt-0.5 font-mono text-[15px] font-medium text-foreground">{item.value}</p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-5 text-[11px] leading-relaxed text-muted-2">
        MSE and PSNR compare the reconstructed image against the original clean image — the same
        comparison used to evaluate the model during training. Lower MSE and higher PSNR indicate a
        closer reconstruction.
      </p>
    </div>
  );
}
