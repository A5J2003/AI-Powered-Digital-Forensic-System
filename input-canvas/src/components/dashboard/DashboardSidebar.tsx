import { useMemo } from "react";
import { useLocation } from "react-router-dom";
import type { LucideIcon } from "lucide-react";
import { Upload, FolderOpen, FileText, Sparkles } from "lucide-react";

import { NavLink } from "@/components/NavLink";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
  useSidebar,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";

type Item = {
  title: string;
  to: string;
  icon: LucideIcon;
};

const items: Item[] = [
  { title: "Upload", to: "/upload", icon: Upload },
  { title: "My Files", to: "/files", icon: FolderOpen },
  { title: "Reports", to: "/reports", icon: FileText },
];



export function DashboardSidebar() {
  const location = useLocation();
  const { state } = useSidebar();
  const collapsed = state === "collapsed";

  const currentPath = location.pathname;
  const isActive = useMemo(() => (path: string) => currentPath === path, [currentPath]);

  return (
    <Sidebar variant="floating" collapsible="icon">
      <SidebarHeader className="px-3 py-3">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              "h-9 w-9 rounded-xl bg-primary text-primary-foreground",
              "flex items-center justify-center shadow-sm",
            )}
          >
            <Sparkles className="h-4 w-4" />
          </div>
          {!collapsed && (
            <div className="min-w-0">
              <div className="text-sm font-semibold text-sidebar-foreground truncate">InputHub</div>
              <div className="text-xs text-sidebar-foreground/70 truncate">Upload & Analyze</div>
            </div>
          )}
        </div>
      </SidebarHeader>

      <SidebarSeparator />

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild tooltip={item.title} isActive={isActive(item.to)}>
                    <NavLink
                      to={item.to}
                      end
                      className="gap-2"
                      activeClassName="bg-sidebar-accent text-sidebar-accent-foreground"
                    >
                      <item.icon className="h-4 w-4" />
                      {!collapsed && <span>{item.title}</span>}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>


      </SidebarContent>
    </Sidebar>
  );
}
