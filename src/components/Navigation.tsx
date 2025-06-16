
import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Brain, Home, Users, BarChart3, BookOpen, TrendingUp, MessageCircle } from "lucide-react";

const Navigation = () => {
  const location = useLocation();
  
  const navItems = [
    { path: "/", label: "Home", icon: Home },
    { path: "/certification-predictor", label: "Certification", icon: Brain },
    { path: "/risk-analyzer", label: "Risk Analysis", icon: TrendingUp },
    { path: "/course-recommendations", label: "Courses", icon: BookOpen },
    { path: "/members", label: "Members", icon: Users },
    { path: "/analytics", label: "Analytics", icon: BarChart3 },
    { path: "/chatbot", label: "Chatbot", icon: MessageCircle },
  ];

  return (
    <nav className="bg-white shadow-lg border-b">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-2">
            <Brain className="h-8 w-8 text-blue-600" />
            <span className="text-xl font-bold text-gray-900">CIEEEN AI-LMS</span>
          </div>
          
          <div className="flex items-center gap-2">
            {navItems.map((item) => (
              <Link key={item.path} to={item.path}>
                <Button
                  variant={location.pathname === item.path ? "default" : "ghost"}
                  size="sm"
                  className="flex items-center gap-2"
                >
                  <item.icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{item.label}</span>
                </Button>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
