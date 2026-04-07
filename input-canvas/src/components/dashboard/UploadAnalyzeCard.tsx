import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import {
  UploadCloud,
  File,
  Mic,
  Video,
  FileText,
  CheckCircle,
  XCircle,
  Loader2,
  X,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import { analyzeFile, pollStatus, type JobStatus } from "@/lib/api";

type Chip = { label: string; icon: LucideIcon; accept: string };

const chips: Chip[] = [
  { label: "Video",   icon: Video,    accept: "video/*" },
  { label: "Audio",   icon: Mic,      accept: "audio/*" },
  { label: "PDF",     icon: File,     accept: ".pdf" },
  { label: "DOCX",    icon: FileText, accept: ".docx" },
  { label: "Text",    icon: FileText, accept: ".txt,.srt" },
];

const ACCEPTED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".wav", ".mp3",
  ".flac", ".pdf", ".docx", ".txt", ".srt"];

type UploadState = "idle" | "uploading" | "pending" | "running" | "done" | "error";

interface UploadStats {
  jobId: string | null;
  filename: string | null;
  status: UploadState;
  progress: number;      // 0–100 visual
  error?: string;
}

const STATUS_LABEL: Record<UploadState, string> = {
  idle: "",
  uploading: "Uploading…",
  pending: "Queued — waiting for pipeline…",
  running: "Analysing — this may take a few minutes…",
  done: "Analysis complete!",
  error: "Analysis failed",
};

const POLL_INTERVAL_MS = 3000;

export function UploadAnalyzeCard({ onBrowse }: { onBrowse?: () => void }) {
  const { toast } = useToast();
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [uploadState, setUploadState] = useState<UploadStats>({
    jobId: null,
    filename: null,
    status: "idle",
    progress: 0,
  });
  const [dragging, setDragging] = useState(false);

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (jobId: string) => {
      pollTimerRef.current = setInterval(async () => {
        try {
          const s = await pollStatus(jobId);
          const st = s.status as JobStatus;

          if (st === "done") {
            stopPolling();
            setUploadState((prev) => ({ ...prev, status: "done", progress: 100 }));
            toast({ title: "✅ Analysis complete!", description: "View results in My Files." });
            setTimeout(() => navigate("/files"), 1500);
          } else if (st === "error") {
            stopPolling();
            setUploadState((prev) => ({
              ...prev,
              status: "error",
              error: s.error ?? "Unknown pipeline error",
            }));
            toast({
              title: "❌ Analysis failed",
              description: s.error ?? "Check backend logs.",
              variant: "destructive",
            });
          } else {
            // pending or running — animate progress bar
            setUploadState((prev) => ({
              ...prev,
              status: st as UploadState,
              progress: Math.min(prev.progress + 3, 90),
            }));
          }
        } catch (err) {
          // network hiccup — keep polling
          console.warn("Poll error:", err);
        }
      }, POLL_INTERVAL_MS);
    },
    [navigate, stopPolling, toast]
  );

  const handleFile = useCallback(
    async (file: File) => {
      const ext = "." + file.name.split(".").pop()?.toLowerCase();
      if (!ACCEPTED_EXTENSIONS.some((e) => ext.startsWith(e.replace("*", "")))) {
        toast({
          title: "Unsupported file type",
          description: `Accepted: ${ACCEPTED_EXTENSIONS.join(", ")}`,
          variant: "destructive",
        });
        return;
      }

      stopPolling();
      setUploadState({ jobId: null, filename: file.name, status: "uploading", progress: 10 });

      try {
        const response = await analyzeFile(file);
        setUploadState({
          jobId: response.job_id,
          filename: file.name,
          status: "pending",
          progress: 20,
        });
        toast({ title: "📤 File uploaded", description: "Pipeline started." });
        startPolling(response.job_id);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Upload failed";
        setUploadState((prev) => ({ ...prev, status: "error", error: msg }));
        toast({ title: "Upload failed", description: msg, variant: "destructive" });
      }
    },
    [startPolling, stopPolling, toast]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";            // reset so same file can be re-picked
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const reset = () => {
    stopPolling();
    setUploadState({ jobId: null, filename: null, status: "idle", progress: 0 });
  };

  const isIdle = uploadState.status === "idle";
  const isActive = ["uploading", "pending", "running"].includes(uploadState.status);
  const isDone = uploadState.status === "done";
  const isError = uploadState.status === "error";

  return (
    <section className="space-y-4">
      <header className="space-y-1">
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-foreground">
          Upload &amp; Analyze
        </h1>
        <p className="text-sm text-muted-foreground">
          Upload media files for AI-powered deepfake detection.
        </p>
      </header>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        className={cn(
          "rounded-2xl border border-border/60 bg-card/50 backdrop-blur-sm",
          "p-5 sm:p-6"
        )}
      >
        {/* ── Drop zone ── */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={cn(
            "relative rounded-2xl border border-dashed transition-colors duration-200",
            "bg-background/40 px-4 py-10 sm:py-12",
            "flex flex-col items-center text-center gap-3",
            dragging ? "border-primary bg-primary/5" : "border-border/70"
          )}
        >
          <AnimatePresence mode="wait">
            {isIdle && (
              <motion.div key="idle" className="flex flex-col items-center gap-3"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <div className="h-12 w-12 rounded-2xl bg-muted flex items-center justify-center">
                  <UploadCloud className="h-6 w-6 text-muted-foreground" />
                </div>
                <div className="space-y-1">
                  <div className="text-sm font-medium text-foreground">Drag &amp; drop files here</div>
                  <div className="text-xs text-muted-foreground">or click to browse • video / audio / PDF / DOCX / TXT</div>
                </div>
                <Button
                  type="button"
                  className="mt-2"
                  onClick={() => { fileInputRef.current?.click(); onBrowse?.(); }}
                >
                  Browse files
                </Button>
              </motion.div>
            )}

            {isActive && (
              <motion.div key="active" className="flex flex-col items-center gap-4 w-full max-w-sm"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <Loader2 className="h-10 w-10 text-primary animate-spin" />
                <div className="text-sm font-medium text-foreground">{uploadState.filename}</div>
                <div className="text-xs text-muted-foreground">{STATUS_LABEL[uploadState.status]}</div>
                <div className="w-full">
                  <Progress value={uploadState.progress} className="h-2" />
                </div>
              </motion.div>
            )}

            {isDone && (
              <motion.div key="done" className="flex flex-col items-center gap-3"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <CheckCircle className="h-12 w-12 text-green-500" />
                <div className="text-sm font-semibold text-foreground">Analysis complete!</div>
                <div className="text-xs text-muted-foreground">Redirecting to My Files…</div>
              </motion.div>
            )}

            {isError && (
              <motion.div key="error" className="flex flex-col items-center gap-3"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <XCircle className="h-12 w-12 text-destructive" />
                <div className="text-sm font-semibold text-foreground">Analysis failed</div>
                <div className="text-xs text-muted-foreground max-w-xs">{uploadState.error}</div>
                <Button variant="outline" size="sm" onClick={reset}>Try again</Button>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Reset X button when not idle */}
          {!isIdle && !isDone && (
            <button
              onClick={reset}
              className="absolute top-4 right-4 text-muted-foreground hover:text-foreground"
              aria-label="Cancel"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* ── Accepted format chips ── */}
        {isIdle && (
          <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
            {chips.map((chip) => (
              <Badge key={chip.label} variant="secondary" className="gap-1.5 cursor-pointer"
                onClick={() => { fileInputRef.current?.setAttribute("accept", chip.accept); fileInputRef.current?.click(); }}>
                <chip.icon className="h-3.5 w-3.5" />
                {chip.label}
              </Badge>
            ))}
          </div>
        )}

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept={ACCEPTED_EXTENSIONS.join(",")}
          onChange={onInputChange}
        />
      </motion.div>
    </section>
  );
}
