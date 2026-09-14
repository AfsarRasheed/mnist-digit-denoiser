/**
 * Small, client-safe display constants. Kept separate from weights.ts /
 * model.ts (which pull in the ~124 KB extracted weight arrays) so
 * client components can show the parameter count without bundling the
 * weights themselves — those stay server-only, used only by /api/predict.
 */
export const MODEL_PARAM_COUNT_DISPLAY = "5,841";
