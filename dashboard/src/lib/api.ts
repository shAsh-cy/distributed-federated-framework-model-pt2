/**
 * Typed API client over the generated OpenAPI types (src/lib/api.gen.ts,
 * regenerated from the committed docs/openapi.json via `npm run gen:api`).
 * The schema file is the contract; nothing here invents a shape.
 */
import type { components } from "./api.gen";

export type RunSummary = components["schemas"]["RunSummary"];
export type RunDetail = components["schemas"]["RunDetail"];
export type StartRunResponse = components["schemas"]["StartRunResponse"];
export type StopRunResponse = components["schemas"]["StopRunResponse"];

export type Dataset = {
  name: string;
  num_classes: number;
  model: string;
  partition_schemes: string[];
};
export type Algorithm = {
  name: string;
  description: string;
  differentially_private: boolean;
};
export type Architecture = {
  name: string;
  input_shape: number[];
  parameter_count: number;
  tensors: { name: string; shape: number[] }[];
};

export const API_BASE = "/api";

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(`API ${status}: ${detail}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* keep statusText */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  datasets: () => request<Dataset[]>("/datasets"),
  algorithms: () => request<Algorithm[]>("/algorithms"),
  architectures: () => request<Architecture[]>("/architectures"),
  runs: () => request<RunSummary[]>("/runs"),
  run: (id: string) => request<RunDetail>(`/runs/${id}`),
  startRun: (config: Record<string, unknown>) =>
    request<StartRunResponse>("/runs", {
      method: "POST",
      body: JSON.stringify({ config }),
    }),
  stopRun: (id: string) => request<StopRunResponse>(`/runs/${id}/stop`, { method: "POST" }),
};

export { ApiError };
