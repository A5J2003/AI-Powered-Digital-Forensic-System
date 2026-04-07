import { useNavigate } from "react-router-dom";
import { DashboardLayout } from "@/components/dashboard/DashboardLayout";
import { RecentAnalyses } from "@/components/dashboard/RecentAnalyses";

export default function MyFilesPage() {
  const navigate = useNavigate();

  return (
    <DashboardLayout title="My Files" subtitle="Your recent uploads & analyses">
      <div className="space-y-8">
        <RecentAnalyses
          onViewReport={(id) => navigate(`/reports?job=${id}`)}
        />
      </div>
    </DashboardLayout>
  );
}
