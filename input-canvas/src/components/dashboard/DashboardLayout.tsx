import { ReactNode, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { LogOut } from "lucide-react";

import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { DashboardSidebar } from "@/components/dashboard/DashboardSidebar";

type DashboardLayoutProps = {
  title: string;
  subtitle?: string;
  children: ReactNode;
};

export function DashboardLayout({ title, subtitle, children }: DashboardLayoutProps) {
  const { user, loading, signOut } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!loading && !user) navigate("/auth");
  }, [loading, user, navigate]);

  const handleSignOut = async () => {
    await signOut();
    navigate("/auth");
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/20">
      <SidebarProvider defaultOpen>
        <div className="min-h-screen flex w-full">
          <DashboardSidebar />

          <SidebarInset>
            <header className="sticky top-0 z-10 border-b border-border/50 bg-card/40 backdrop-blur-sm">
              <div className="h-14 px-4 sm:px-6 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 min-w-0">
                  <SidebarTrigger className="mr-1" />
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-foreground truncate">{title}</div>
                    <div className="text-xs text-muted-foreground truncate">
                      {subtitle ?? user.email}
                    </div>
                  </div>
                </div>

                <div className="flex-1 max-w-xl hidden md:block">
                  <Input placeholder="Search files…" />
                </div>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleSignOut}
                  className="text-muted-foreground hover:text-foreground"
                >
                  <LogOut className="w-4 h-4 mr-2" />
                  Sign out
                </Button>
              </div>
            </header>

            <main className="relative px-4 sm:px-6 py-6 sm:py-10">
              <div className="max-w-5xl mx-auto">{children}</div>
            </main>
          </SidebarInset>
        </div>
      </SidebarProvider>
    </div>
  );
}
