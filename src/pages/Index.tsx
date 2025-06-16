
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { Brain, TrendingUp, BookOpen, Users, BarChart3, MessageCircle } from "lucide-react";

const Index = () => {
  const features = [
    {
      icon: Brain,
      title: "Certification Eligibility Predictor",
      description: "AI classification model to determine certification qualification based on training data, assessment scores, and compliance history.",
      link: "/certification-predictor",
      color: "bg-blue-50 border-blue-200 hover:bg-blue-100"
    },
    {
      icon: TrendingUp,
      title: "Performance Risk Analyzer",
      description: "Binary classifier that analyzes learning patterns and engagement to identify members at risk of non-compliance.",
      link: "/risk-analyzer",
      color: "bg-red-50 border-red-200 hover:bg-red-100"
    },
    {
      icon: BookOpen,
      title: "Course Recommendation System",
      description: "Intelligent recommendation engine suggesting upskilling courses based on learning history and career goals.",
      link: "/course-recommendations",
      color: "bg-green-50 border-green-200 hover:bg-green-100"
    },
    {
      icon: Users,
      title: "Member Dashboard",
      description: "Comprehensive view of member profiles, learning progress, and compliance status.",
      link: "/members",
      color: "bg-purple-50 border-purple-200 hover:bg-purple-100"
    },
    {
      icon: BarChart3,
      title: "Analytics & Insights",
      description: "Visual insights, confusion matrices, ROC curves, and compliance heatmaps for administrators.",
      link: "/analytics",
      color: "bg-orange-50 border-orange-200 hover:bg-orange-100"
    },
    {
      icon: MessageCircle,
      title: "AI Career Chatbot",
      description: "Intelligent chatbot providing career path suggestions and guidance for professional development.",
      link: "/chatbot",
      color: "bg-indigo-50 border-indigo-200 hover:bg-indigo-100"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            CIEEEN AI-Powered Learning Management System
          </h1>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto">
            AI-Based Systems for Automating Certification, Compliance Monitoring, and Professional Development Processes
          </p>
          <div className="mt-6 p-4 bg-blue-100 rounded-lg max-w-3xl mx-auto">
            <p className="text-blue-800 font-medium">
              Chartered Institute of Electrical and Electronic Engineering of Nigeria (CIEEEN)
            </p>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {features.map((feature, index) => (
            <Card key={index} className={`transition-all duration-300 hover:shadow-lg cursor-pointer ${feature.color}`}>
              <CardHeader className="pb-4">
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 bg-white rounded-lg shadow-sm">
                    <feature.icon className="h-6 w-6 text-gray-700" />
                  </div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <CardDescription className="text-gray-600 mb-4 leading-relaxed">
                  {feature.description}
                </CardDescription>
                <Link to={feature.link}>
                  <Button variant="outline" size="sm" className="w-full">
                    Access Feature
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Project Info */}
        <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">Project Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-medium text-gray-800 mb-3">Core AI Features</h3>
              <ul className="space-y-2 text-gray-600">
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  Certification Eligibility Prediction
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                  Performance Risk Analysis
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  Intelligent Course Recommendations
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  Predictive Analytics & Insights
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-medium text-gray-800 mb-3">Expected Outputs</h3>
              <ul className="space-y-2 text-gray-600">
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                  Confusion Matrices & ROC Curves
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-indigo-500 rounded-full"></div>
                  Chi-square Feature Analysis
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
                  Visual Prediction Insights
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-pink-500 rounded-full"></div>
                  Compliance Heatmaps
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-blue-600 text-white p-6 rounded-lg text-center">
            <div className="text-3xl font-bold mb-2">3</div>
            <div className="text-blue-100">Core AI Models</div>
          </div>
          <div className="bg-green-600 text-white p-6 rounded-lg text-center">
            <div className="text-3xl font-bold mb-2">6</div>
            <div className="text-green-100">System Features</div>
          </div>
          <div className="bg-purple-600 text-white p-6 rounded-lg text-center">
            <div className="text-3xl font-bold mb-2">∞</div>
            <div className="text-purple-100">Scalable Members</div>
          </div>
          <div className="bg-orange-600 text-white p-6 rounded-lg text-center">
            <div className="text-3xl font-bold mb-2">24/7</div>
            <div className="text-orange-100">AI Monitoring</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
