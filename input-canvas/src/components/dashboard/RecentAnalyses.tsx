import { useEffect } from "react";
import { motion } from "framer-motion";
import { MoreHorizontal, FileText, Download, RefreshCw, AlertCircle } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import { listJobs, downloadReport, type JobMeta } from "@/lib/api";

type Risk = "Low" | "Medium" | "High";

function getRisk(confidence: number, label: 0 | 1 | undefined): Risk {
  if (label !== 1) return "Low";
  if (confidence >= 0.75) return "High";
  if (confidence >= 0.5) return "Medium";
  return "Low";
}

function riskBadgeVariant(risk: Risk): "default" | "secondary" | "destructive" {
  if (risk === "High") return "destructive";
  if (risk === "Medium") return "default";
  return "secondary";
}

function kindLabel(filename?: string): string {
  if (!filename) return "File";
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  if (["mp4", "avi", "mov", "mkv"].includes(ext)) return "Video";
  if (["wav", "mp3", "flac"].includes(ext)) return "Audio";
  if (["pdf"].includes(ext)) return "PDF";
  if (["docx", "doc"].includes(ext)) return "DOCX";
  return "Text";
}

export function RecentAnalyses({
  onViewReport,
}: {
  onViewReport?: (id: string) => void;
}) {
  const { toast } = useToast();
  const navigate = useNavigate();

  const {
    data: jobs,
    isLoading,
    isError,
    refetch,
  } = useQuery<JobMeta[]>({
    queryKey: ["jobs"],
    queryFn: listJobs,
    refetchInterval: 10_000,   // auto-refresh every 10 s
    staleTime: 5_000,
  });

  const handleDownload = async (jobId: string) => {
    try {
      await downloadReport(jobId);
    } catch (err: unknown) {
      toast({
        title: "Download failed",
        description: err instanceof Error ? err.message : "Report not available",
        variant: "destructive",
      });
    }
  };

  const handleViewReport = (jobId: string) => {
    onViewReport?.(jobId);
    navigate(`/reports?job=${jobId}`);
  };

  // ── Loading skeleton ─────────────────────────────────────────────
  if (isLoading) {
    return (
      <section className="space-y-4">
        <div className="flex items-center gap-2 text-muted-foreground">
          <RefreshCw className="h-4 w-4 animate-spin" />
          <span className="text-sm">Loading analyses…</span>
        </div>
        {[1, 2].map((i) => (
          <div key={i} className="h-24 rounded-2xl bg-muted/40 animate-pulse" />
        ))}
      </section>
    );
  }

  // ── Error ────────────────────────────────────────────────────────
  if (isError) {
    return (
      <section className="space-y-4">
        <div className="flex items-center gap-2 text-destructive">
          <AlertCircle className="h-4 w-4" />
          <span className="text-sm">
            Could not reach the backend. Make sure the FastAPI server is running on{" "}
            <code className="text-xs bg-muted rounded px-1">localhost:8000</code>.
          </span>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry
        </Button>
      </section>
    );
  }

  // ── Empty state ──────────────────────────────────────────────────
  if (!jobs || jobs.length === 0) {
    return (
      <section className="space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-foreground">Recent analyses</h2>
            <p className="text-sm text-muted-foreground">No files analysed yet.</p>
          </div>
          <Button variant="secondary" size="sm" onClick={() => refetch()} className="gap-2">
            <RefreshCw className="h-4 w-4" />
            Refresh
          </Button>
        </div>
        <Card className="p-8 text-center text-muted-foreground text-sm bg-card/50 border-border/60">
          Upload a file from the <strong>Upload</strong> tab to get started.
        </Card>
      </section>
    );
  }

  // ── Job list ─────────────────────────────────────────────────────
  const doneJobs   = jobs.filter((j) => j.status === "done");
  const activeJobs = jobs.filter((j) => j.status !== "done");

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Recent analyses</h2>
          <p className="text-sm text-muted-foreground">
            {doneJobs.length} completed &bull; {activeJobs.length} in progress
          </p>
        </div>
        <Button variant="secondary" className="gap-2" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      {/* Active / in-progress jobs */}
      {activeJobs.map((job) => (
        <motion.div
          key={job.job_id}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="p-4 sm:p-5 bg-card/50 backdrop-blur-sm border-border/60">
            <div className="flex items-center gap-3">
              <RefreshCw className="h-5 w-5 text-primary animate-spin flex-shrink-0" />
              <div className="min-w-0">
                <div className="font-medium text-foreground truncate">
                  {job.filename ?? job.job_id}
                </div>
                <div className="text-xs text-muted-foreground capitalize mt-1">
                  {job.status}…
                </div>
              </div>
              <Badge variant="secondary" className="ml-auto capitalize">
                {job.status}
              </Badge>
            </div>
          </Card>
        </motion.div>
      ))}

      {/* Completed jobs */}
      {doneJobs.map((job, idx) => {
        const label = job.final?.label as 0 | 1 | undefined;
        const conf  = job.final?.confidence ?? 0;
        const risk  = getRisk(conf, label);

        // FIX: the backend stores "confidence" as the fake probability.
        // For a REAL verdict (label=0) we invert it so the card shows
        // how confident the system is that the file IS real, which matches
        // what the report page displays.
        const confidencePct = label === 0
          ? Math.round((1 - conf) * 100)  // REAL: show real confidence
          : Math.round(conf * 100);        // FAKE: fake prob is already correct

        const verdictStr = label === 1 ? "FAKE" : label === 0 ? "REAL" : "—";
        const kind = kindLabel(job.filename ?? undefined);

        return (
          <motion.div
            key={job.job_id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 * idx, duration: 0.3 }}
          >
            <Card
              className={cn(
                "p-4 sm:p-5 bg-card/50 backdrop-blur-sm border-border/60",
                label === 1 && "border-destructive/30"
              )}
            >
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                {/* Left — file info */}
                <div className="min-w-0">
                  <div className="flex items-start gap-3">
                    <div className="h-10 w-10 rounded-xl bg-muted flex items-center justify-center text-muted-foreground text-xs font-semibold flex-shrink-0">
                      {kind.slice(0, 3).toUpperCase()}
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <div className="font-medium text-foreground truncate max-w-[200px]">
                          {job.filename ?? job.job_id}
                        </div>
                        <Badge variant={riskBadgeVariant(risk)}>{risk} risk</Badge>
                        {label !== undefined && (
                          <Badge
                            variant={label === 1 ? "destructive" : "secondary"}
                            className="font-mono"
                          >
                            {verdictStr}
                          </Badge>
                        )}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Case ID: {job.job_id.slice(0, 8)}…
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right — confidence + actions */}
                <div className="sm:w-[320px] w-full">
                  <div className="flex items-center justify-between">
                    <div className="text-xs text-muted-foreground">Confidence</div>
                    <div className="text-sm font-semibold text-foreground">
                      {label !== undefined ? `${confidencePct}%` : "—"}
                    </div>
                  </div>
                  <Progress
                    value={label !== undefined ? confidencePct : 0}
                    className={cn(
                      "mt-2",
                      label === 1 ? "[&>div]:bg-destructive" : "[&>div]:bg-green-500"
                    )}
                  />

                  <div className="mt-3 flex items-center justify-end gap-2">
                    <Button size="sm" onClick={() => handleViewReport(job.job_id)}>
                      View report
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      className="gap-2"
                      onClick={() => handleDownload(job.job_id)}
                    >
                      <Download className="h-4 w-4" />
                      Export PDF
                    </Button>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button size="sm" variant="ghost" aria-label="More actions">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => handleViewReport(job.job_id)}>
                          <FileText className="h-4 w-4 mr-2" />
                          Full report
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleDownload(job.job_id)}>
                          <Download className="h-4 w-4 mr-2" />
                          Download PDF
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        );
      })}
    </section>
  );
}