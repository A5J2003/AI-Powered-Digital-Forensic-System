// src/lib/api.ts
// Typed client for the AI Forensic FastAPI backend

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

export type JobStatus = "pending" | "running" | "done" | "error";

export interface SubmitResponse {
  job_id: string;
  filename: string;
  status: JobStatus;
}

export interface StatusResponse {
  job_id: string;
  status: JobStatus;
  filename?: string;
  error?: string;
}

export interface ModalityResult {
  label: 0 | 1 | null;
  confidence: number;
  fake_prob?: number;
  real_prob?: number;
  details?: Record<string, unknown>;
  explainability?: Record<string, unknown>;
  explanation?: Record<string, unknown>;
}

export interface FinalResult {
  label: 0 | 1;
  confidence: number;
  explanation: string;
  debug?: Record<string, number>;
}

export interface AnalysisResult {
  file_hash: string;
  integrity_verified: boolean;
  final: FinalResult;
  modalities: {
    video?: ModalityResult;
    audio?: ModalityResult;
    text?: ModalityResult;
  };
  ingestion?: Record<string, unknown>;
}

export interface JobMeta {
  job_id: string;
  status: JobStatus;
  filename?: string;
  case_path?: string;
  final?: Partial<FinalResult>;
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

async function apiFetch<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore parse errors
    }
    throw new Error(detail);
  }

  return res.json() as Promise<T>;
}

// ─────────────────────────────────────────────
// API Methods
// ─────────────────────────────────────────────

/** Upload a file and start analysis. Returns job_id immediately. */
export async function analyzeFile(file: File): Promise<SubmitResponse> {
  const form = new FormData();
  form.append("file", file);
  return apiFetch<SubmitResponse>("/analyze", {
    method: "POST",
    body: form,
  });
}

/** Poll the status of a running job. */
export async function pollStatus(jobId: string): Promise<StatusResponse> {
  return apiFetch<StatusResponse>(`/status/${jobId}`);
}

/** Fetch the full result of a completed job. */
export async function fetchResult(jobId: string): Promise<AnalysisResult> {
  return apiFetch<AnalysisResult>(`/result/${jobId}`);
}

/** Trigger a PDF report download in the browser. */
export async function downloadReport(jobId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/report/${jobId}`);
  if (!res.ok) {
    throw new Error(`Report not available: ${res.statusText}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `forensic_report_${jobId}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** List all jobs (newest first). */
export async function listJobs(): Promise<JobMeta[]> {
  return apiFetch<JobMeta[]>("/jobs");
}

/** Check if the backend server is reachable. */
export async function checkHealth(): Promise<boolean> {
  try {
    await apiFetch<{ status: string }>("/health");
    return true;
  } catch {
    return false;
  }
}
