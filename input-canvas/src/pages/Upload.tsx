import { DashboardLayout } from "@/components/dashboard/DashboardLayout";
import { UploadAnalyzeCard } from "@/components/dashboard/UploadAnalyzeCard";

export default function UploadPage() {
  return (
    <DashboardLayout title="Upload" subtitle="Upload & analyze your inputs">
      <div className="space-y-8">
        <UploadAnalyzeCard />
      </div>
    </DashboardLayout>
  );
}
