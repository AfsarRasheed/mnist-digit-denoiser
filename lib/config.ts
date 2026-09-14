/**
 * Base URL for the inference API. Empty string means "same origin" (the
 * default — the Next.js app serves both the frontend and /api/predict).
 * Only set NEXT_PUBLIC_API_BASE_URL if the API is deployed separately from
 * the frontend (see README "ML Backend" section).
 */
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "";
