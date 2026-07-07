export type JobStatus =
  | "pending"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "cancelled";

export interface JobSnapshot {
  id: string;
  type: string;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  progress: Record<string, unknown>;
  logs: string[];
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface JobListResponse {
  items: JobSnapshot[];
}

export interface DatasetModeInfo {
  hint?: string;
  channels?: string[];
  [key: string]: unknown;
}

export interface DatasetModesResponse {
  [mode: string]: DatasetModeInfo;
}

export interface DatasetFilter {
  min_duration_seconds?: number | null;
  max_duration_seconds?: number | null;
  language?: string | null;
  categories?: string[] | null;
  min_likes?: number | null;
  max_likes?: number | null;
  min_views?: number | null;
  max_views?: number | null;
  date_from?: string | null;
  date_to?: string | null;
}

export interface DatasetPreviewResponse {
  estimated_count: number;
  publication_date_histogram: {
    by_month: Record<string, number>;
    by_year: Record<string, number>;
  };
  estimated_storage_bytes: number;
}

export interface DatasetBuildResponse {
  job_id: string;
}

export interface DatasetVideosResponse {
  items: Array<Record<string, unknown>>;
  total: number;
  limit: number;
  offset: number;
}

export interface TribeStatusResponse {
  cached: number;
  remaining: number;
  failed: number;
  processing_video: string | null;
  eta_seconds: number | null;
}

export interface TensorInfo {
  video_id: number;
  shape: number[];
  dtype: string;
  num_timesteps: number | null;
  num_features: number | null;
  file_size_bytes: number;
  path: string;
}

export interface TensorListResponse {
  items: TensorInfo[];
  count: number;
  storage_bytes: number;
}

export interface TrainingStartRequest {
  config: Record<string, unknown>;
}

export type TrainingExperimentsResponse = Array<Record<string, unknown>>;

export interface JobSubmissionResponse {
  job_id: string;
}

export interface EvaluationLatestResponse {
  metrics?: Record<string, unknown>;
  residuals?: Record<string, unknown>;
  calibration?: Record<string, unknown>;
  n_samples?: number;
}

export interface EvaluationReport {
  metrics: Record<string, unknown>;
  residuals: Record<string, unknown>;
  calibration: Record<string, unknown>;
  n_samples: number;
}

export interface PredictionHorizonResponse {
  horizon_days: number;
}

export interface PredictionResult {
  video_id: number;
  youtube_id: string | null;
  horizon_days: number;
  horizon_label: string;
  prediction: {
    likes?: number;
    views?: number;
    engagement?: number;
    virality?: number;
    confidence_level?: number;
    point_estimates?: Record<string, number>;
    intervals?: Record<string, unknown>;
    uncertainty?: Record<string, unknown>;
    [key: string]: unknown;
  };
  explanation: Record<string, unknown>;
}

export interface MonitoringSystemResponse {
  cpu_percent: number;
  memory_used: number;
  memory_total: number;
  disk_used: number;
  disk_total: number;
  gpu: Record<string, unknown> | null;
}

export interface MonitoringPayload {
  system: MonitoringSystemResponse;
  active_jobs: JobSnapshot[];
}

export interface AnalysisMethodResponse {
  name: string;
  description: string;
}

export interface AnalysisRunResult {
  method: string;
  n_samples?: number;
  n_features?: number;
  n_components?: number;
  clamped?: boolean;
  error?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface PipelineRunResponse {
  job_id: string;
}
