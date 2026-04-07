import { useMemo } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  AlertCircle,
  RefreshCw,
  Download,
  CheckCircle,
  XCircle,
  ShieldAlert,
  FileText,
  Mic,
  Video,
  ArrowLeft,
} from "lucide-react";

import { DashboardLayout } from "@/components/dashboard/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { fetchResult, downloadReport, listJobs, type AnalysisResult, type JobMeta } from "@/lib/api";
import { cn } from "@/lib/utils";

// ─── Types / helpers ────────────────────────────────────────────────────────

function VerdictBadge({ label }: { label: 0 | 1 | null | undefined }) {
  if (label === null || label === undefined) {
    return <Badge variant="secondary">N/A</Badge>;
  }
  return (
    <Badge variant={label === 1 ? "destructive" : "secondary"} className="text-sm px-3 py-1 font-mono">
      {label === 1 ? "FAKE" : "REAL"}
    </Badge>
  );
}

function ModalityCard({
  title,
  icon: Icon,
  result,
}: {
  title: string;
  icon: React.ElementType;
  result?: AnalysisResult["modalities"]["video"];
}) {
  const label = result?.label;
  const conf = result?.confidence ?? 0;
  const confPct = Math.round(conf * 100);
  const isAvailable = label !== null && label !== undefined;

  return (
    <Card className="bg-card/50 backdrop-blur-sm border-border/60">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2 text-muted-foreground">
          <Icon className="h-4 w-4" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {isAvailable ? (
          <>
            <div className="flex items-center justify-between">
              <VerdictBadge label={label} />
              <span className="text-sm font-semibold">{confPct}%</span>
            </div>
            <Progress
              value={confPct}
              className={cn(
                "h-2",
                label === 1 ? "[&>div]:bg-destructive" : "[&>div]:bg-green-500"
              )}
            />
          </>
        ) : (
          <p className="text-xs text-muted-foreground">Not available for this file type.</p>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Page ───────────────────────────────────────────────────────────────────

export default function ReportsPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const jobId = params.get("job");

  // List of all jobs (for the "select a job" view)
  const { data: jobs, isLoading: jobsLoading } = useQuery<JobMeta[]>({
    queryKey: ["jobs"],
    queryFn: listJobs,
    staleTime: 10_000,
    enabled: !jobId,
  });

  // Full result for the selected job
  const {
    data: result,
    isLoading: resultLoading,
    isError,
    error,
  } = useQuery<AnalysisResult>({
    queryKey: ["result", jobId],
    queryFn: () => fetchResult(jobId!),
    enabled: !!jobId,
    retry: 1,
  });

  const handleDownload = async () => {
    if (!jobId) return;
    try {
      await downloadReport(jobId);
    } catch (err: unknown) {
      toast({
        title: "Download failed",
        description: err instanceof Error ? err.message : "PDF not available",
        variant: "destructive",
      });
    }
  };

  const final = result?.final;
  const modalities = result?.modalities ?? {};

  const explanation = useMemo(() => {
    if (!final) return null;
    return typeof final.explanation === "string" ? final.explanation : null;
  }, [final]);

  // ── No job selected — show job picker ───────────────────────────
  if (!jobId) {
    return (
      <DashboardLayout title="Reports" subtitle="Detailed analysis reports">
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Select an analysis to view its full report.
          </p>
          {jobsLoading ? (
            <div className="flex items-center gap-2 text-muted-foreground">
              <RefreshCw className="h-4 w-4 animate-spin" />
              <span className="text-sm">Loading jobs…</span>
            </div>
          ) : (
            <div className="space-y-2">
              {(jobs ?? [])
                .filter((j) => j.status === "done")
                .map((job) => (
                  <Card
                    key={job.job_id}
                    className="p-4 cursor-pointer hover:bg-accent transition-colors bg-card/50 border-border/60"
                    onClick={() => navigate(`/reports?job=${job.job_id}`)}
                  >
                    <div className="flex items-center gap-3">
                      <FileText className="h-5 w-5 text-muted-foreground" />
                      <div className="min-w-0">
                        <div className="font-medium text-foreground truncate">
                          {job.filename ?? job.job_id}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {job.job_id.slice(0, 12)}…
                        </div>
                      </div>
                      {job.final?.label !== undefined && (
                        <Badge
                          variant={job.final.label === 1 ? "destructive" : "secondary"}
                          className="ml-auto font-mono"
                        >
                          {job.final.label === 1 ? "FAKE" : "REAL"}
                        </Badge>
                      )}
                    </div>
                  </Card>
                ))}
              {(jobs ?? []).filter((j) => j.status === "done").length === 0 && (
                <Card className="p-8 text-center text-muted-foreground text-sm bg-card/50 border-border/60">
                  No completed analyses yet. Upload a file to get started.
                </Card>
              )}
            </div>
          )}
        </div>
      </DashboardLayout>
    );
  }

  // ── Loading ──────────────────────────────────────────────────────
  if (resultLoading) {
    return (
      <DashboardLayout title="Report" subtitle={`Job ${jobId.slice(0, 8)}…`}>
        <div className="flex items-center gap-2 text-muted-foreground py-12 justify-center">
          <RefreshCw className="h-5 w-5 animate-spin" />
          <span>Loading report…</span>
        </div>
      </DashboardLayout>
    );
  }

  // ── Error ────────────────────────────────────────────────────────
  if (isError || !result) {
    return (
      <DashboardLayout title="Report" subtitle="Error">
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-destructive">
            <AlertCircle className="h-5 w-5" />
            <span className="text-sm">
              {error instanceof Error ? error.message : "Could not load report."}
            </span>
          </div>
          <Button variant="outline" onClick={() => navigate("/reports")}>
            <ArrowLeft className="h-4 w-4 mr-2" /> Back to reports
          </Button>
        </div>
      </DashboardLayout>
    );
  }

  // ── Full report ──────────────────────────────────────────────────
  return (
    <DashboardLayout
      title="Forensic Report"
      subtitle={`Job ${jobId.slice(0, 8)}… · ${result.file_hash?.slice(0, 16)}…`}
    >
      <div className="space-y-6">
        {/* Back + Download */}
        <div className="flex items-center justify-between gap-3">
          <Button variant="ghost" size="sm" onClick={() => navigate("/reports")}>
            <ArrowLeft className="h-4 w-4 mr-2" /> All reports
          </Button>
          <Button onClick={handleDownload} className="gap-2">
            <Download className="h-4 w-4" />
            Download PDF
          </Button>
        </div>

        {/* Verdict Banner */}
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className={cn(
            "rounded-2xl p-6 flex flex-col sm:flex-row items-center justify-between gap-4",
            final?.label === 1
              ? "bg-destructive/10 border border-destructive/30"
              : "bg-green-500/10 border border-green-500/30"
          )}
        >
          <div className="flex items-center gap-4">
            {final?.label === 1
              ? <XCircle className="h-10 w-10 text-destructive" />
              : <CheckCircle className="h-10 w-10 text-green-500" />}
            <div>
              <div className="text-2xl font-bold tracking-tight">
                {final?.label === 1 ? "FAKE DETECTED" : "AUTHENTIC"}
              </div>
              <div className="text-sm text-muted-foreground">
                Confidence: {Math.round((final?.confidence ?? 0) * 100)}%
              </div>
            </div>
          </div>
          <div className="flex flex-col items-center gap-1">
            <ShieldAlert className={cn("h-8 w-8", final?.label === 1 ? "text-destructive" : "text-green-500")} />
            <div className="text-xs text-muted-foreground">
              Integrity: {result.integrity_verified ? "✓ Verified" : "✗ Not verified"}
            </div>
          </div>
        </motion.div>

        {/* Explanation */}
        {explanation && (
          <Card className="bg-card/50 border-border/60">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Analysis explanation
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-foreground leading-relaxed">
              {explanation}
            </CardContent>
          </Card>
        )}

        {/* Modality Breakdown */}
        <div>
          <h2 className="text-base font-semibold mb-3 text-foreground">Modality breakdown</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <ModalityCard title="Video" icon={Video} result={modalities.video} />
            <ModalityCard title="Audio" icon={Mic} result={modalities.audio} />
            <ModalityCard title="Text" icon={FileText} result={modalities.text} />
          </div>
        </div>

        {/* File hash */}
        <Card className="bg-card/50 border-border/60">
          <CardContent className="py-4">
            <div className="text-xs text-muted-foreground">
              <span className="font-semibold">SHA-256 hash: </span>
              <code className="font-mono break-all">{result.file_hash}</code>
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
