import type {
  AnalysisMethodResponse,
  AnalysisRunResult,
  DatasetFilter,
  DatasetModesResponse,
  DatasetPreviewResponse,
  DatasetVideosResponse,
  EvaluationLatestResponse,
  EvaluationReport,
  JobListResponse,
  JobSnapshot,
  MonitoringSystemResponse,
  PredictionHorizonResponse,
  JobSubmissionResponse,
  TensorListResponse,
  TrainingExperimentsResponse,
  TribeStatusResponse,
} from "./types";

const BASE_URL = "/api";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

async function requestBlob(path: string, init?: RequestInit): Promise<Blob> {
  const response = await fetch(`${BASE_URL}${path}`, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return await response.blob();
}

export const api = {
  datasets: {
    modes: () => requestJson<DatasetModesResponse>("/datasets/modes"),
    addChannel: (mode: string, url: string) =>
      requestJson<Record<string, unknown>>(`/datasets/modes/${encodeURIComponent(mode)}/channels`, {
        method: "POST",
        body: JSON.stringify({ url }),
      }),
    removeChannel: (mode: string, url: string) =>
      requestJson<Record<string, unknown>>(`/datasets/modes/${encodeURIComponent(mode)}/channels`, {
        method: "DELETE",
        body: JSON.stringify({ url }),
      }),
    preview: (payload: { mode: string; filter?: DatasetFilter; count?: number }) =>
      requestJson<DatasetPreviewResponse>("/datasets/preview", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    build: (payload: { mode: string; filter?: DatasetFilter; count?: number; force?: boolean }) =>
      requestJson<JobSubmissionResponse>("/datasets/build", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    validate: (payload: { filter?: DatasetFilter; count?: number }) =>
      requestJson<JobSubmissionResponse>("/datasets/validate", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    report: () => requestJson<Record<string, unknown>>("/datasets/report"),
    videos: (limit = 50, offset = 0) =>
      requestJson<DatasetVideosResponse>(`/datasets/videos?limit=${limit}&offset=${offset}`),
  },
  tribe: {
    run: (payload: { force?: boolean; retry_failed?: boolean; limit?: number }) =>
      requestJson<JobSubmissionResponse>("/tribe/run", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    status: () => requestJson<TribeStatusResponse>("/tribe/status"),
  },
  tensors: {
    list: () => requestJson<TensorListResponse>("/tensors/"),
    get: (videoId: number) => requestJson<TensorListResponse>(`/tensors/${videoId}`),
  },
  training: {
    start: (config: Record<string, unknown>) =>
      requestJson<JobSubmissionResponse>("/training/start", {
        method: "POST",
        body: JSON.stringify({ config }),
      }),
    pause: (jobId: string) => requestJson<JobSnapshot>(`/training/${jobId}/pause`, { method: "POST" }),
    resume: (jobId: string) => requestJson<JobSnapshot>(`/training/${jobId}/resume`, { method: "POST" }),
    stop: (jobId: string) => requestJson<JobSnapshot>(`/training/${jobId}/stop`, { method: "POST" }),
    job: (jobId: string) => requestJson<JobSnapshot>(`/training/${jobId}`),
    experiments: () => requestJson<TrainingExperimentsResponse>("/training/experiments"),
  },
  evaluation: {
    run: () => requestJson<EvaluationReport>("/evaluation/run", { method: "POST" }),
    latest: () => requestJson<EvaluationLatestResponse>("/evaluation/latest"),
  },
  prediction: {
    url: (payload: { url: string; confidence_level?: number; horizon_days?: number | null }) =>
      requestJson<JobSubmissionResponse>("/prediction/url", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    upload: (file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      return requestJson<JobSubmissionResponse>("/prediction/upload", {
        method: "POST",
        body: formData,
      });
    },
    job: (jobId: string) => requestJson<JobSnapshot>(`/prediction/${jobId}`),
    horizon: () => requestJson<PredictionHorizonResponse>("/prediction/horizon"),
  },
  monitoring: {
    system: () => requestJson<MonitoringSystemResponse>("/monitoring/system"),
    jobs: () => requestJson<JobListResponse>("/monitoring/jobs"),
    logs: (limit = 200) => requestJson<{ items: string[] }>(`/monitoring/logs?limit=${limit}`),
    wsUrl: () => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${protocol}//${window.location.host}/api/monitoring/ws`;
    },
  },
  pipeline: {
    run: (payload: { mode: string; n_videos: number; horizon_days?: number | null }) =>
      requestJson<JobSubmissionResponse>("/pipeline/run", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    job: (jobId: string) => requestJson<JobSnapshot>(`/pipeline/${jobId}`),
  },
  analysis: {
    methods: () => requestJson<AnalysisMethodResponse[]>("/analysis/methods"),
    run: (payload: Record<string, unknown>) =>
      requestJson<AnalysisRunResult>("/analysis/run", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
    export: async (payload: Record<string, unknown>, fmt: "csv" | "npz" | "pt") => {
      const blob = await requestBlob("/analysis/export", {
        method: "POST",
        body: JSON.stringify({ ...payload, fmt }),
      });
      return blob;
    },
  },
};

export function downloadBlob(blob: Blob, filename: string): void {
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.URL.revokeObjectURL(url);
}
