import { useToast } from "@/hooks/use-toast";
import { useNavigate } from "react-router-dom";
import { UploadAnalyzeCard } from "@/components/dashboard/UploadAnalyzeCard";
import { RecentAnalyses } from "@/components/dashboard/RecentAnalyses";
import { DashboardLayout } from "@/components/dashboard/DashboardLayout";

const Dashboard = () => {
  const { toast } = useToast();
  const navigate = useNavigate();

  return (
    <DashboardLayout title="Dashboard" subtitle="Overview">
      <div className="space-y-8">
        <UploadAnalyzeCard />
        <RecentAnalyses
          onViewReport={(id) => navigate(`/reports?job=${id}`)}
        />
      </div>
    </DashboardLayout>
  );
};

export default Dashboard;
