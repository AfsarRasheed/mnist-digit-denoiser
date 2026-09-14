import FlowArrow from "./FlowArrow";

const STAGES = ["Input", "Corruption", "Output"];

/** Static orientation preview shown before any input has been processed —
 * the same 3-stage shape as the real pipeline, without pretending to show data. */
export default function GhostPipeline() {
  return (
    <div className="flex items-center justify-center gap-2.5 opacity-70">
      {STAGES.map((label, i) => (
        <div key={label} className="flex items-center gap-2.5">
          <div className="flex flex-col items-center gap-1.5">
            <div className="h-11 w-11 rounded-md border border-dashed border-border-strong" />
            <span className="text-[10px] text-muted-2">{label}</span>
          </div>
          {i < STAGES.length - 1 && <FlowArrow />}
        </div>
      ))}
    </div>
  );
}
