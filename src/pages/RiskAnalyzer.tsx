
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { TrendingUp, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import { toast } from "sonner";

const RiskAnalyzer = () => {
  const [formData, setFormData] = useState({
    learningHours: "",
    engagementScore: "",
    assessmentAttempts: "",
    coursesCompleted: "",
    lastActivityDays: "",
    complianceRating: ""
  });
  
  const [riskAnalysis, setRiskAnalysis] = useState<{
    riskLevel: "LOW" | "MEDIUM" | "HIGH";
    riskScore: number;
    factors: string[];
    interventions: string[];
    model_used?: string;
  } | null>(null);
  
  const [loading, setLoading] = useState(false);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleAnalyze = async () => {
    if (!formData.learningHours || !formData.engagementScore) {
      toast.error("Please fill in at least learning hours and engagement score");
      return;
    }

    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/risk/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to get risk analysis');
      }

      const result = await response.json();
      setRiskAnalysis(result);
      toast.success("Risk analysis completed successfully!");
    } catch (error) {
      console.error('Error:', error);
      toast.error("Failed to get risk analysis. Please ensure the API server is running.");
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      learningHours: "",
      engagementScore: "",
      assessmentAttempts: "",
      coursesCompleted: "",
      lastActivityDays: "",
      complianceRating: ""
    });
    setRiskAnalysis(null);
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case "LOW": return "text-green-600";
      case "MEDIUM": return "text-yellow-600";
      case "HIGH": return "text-red-600";
      default: return "text-gray-600";
    }
  };

  const getRiskBgColor = (level: string) => {
    switch (level) {
      case "LOW": return "bg-green-50 border-green-200";
      case "MEDIUM": return "bg-yellow-50 border-yellow-200";
      case "HIGH": return "bg-red-50 border-red-200";
      default: return "bg-gray-50 border-gray-200";
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <TrendingUp className="h-8 w-8 text-orange-600" />
          <h1 className="text-3xl font-bold text-gray-900">Performance Risk Analyzer</h1>
        </div>
        <p className="text-gray-600 max-w-3xl">
          Binary classifier that analyzes member learning patterns, assessment data, and engagement metrics 
          to identify professionals at risk of non-compliance or poor performance.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <Card className="h-fit">
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
            <CardDescription>
              Enter member's learning and engagement data for risk assessment analysis.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="learningHours">Learning Hours (Monthly)</Label>
                <Input
                  id="learningHours"
                  type="number"
                  placeholder="e.g., 15"
                  value={formData.learningHours}
                  onChange={(e) => handleInputChange("learningHours", e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="engagementScore">Engagement Score (%)</Label>
                <Input
                  id="engagementScore"
                  type="number"
                  placeholder="e.g., 75"
                  value={formData.engagementScore}
                  onChange={(e) => handleInputChange("engagementScore", e.target.value)}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="assessmentAttempts">Assessment Attempts</Label>
                <Input
                  id="assessmentAttempts"
                  type="number"
                  placeholder="e.g., 2"
                  value={formData.assessmentAttempts}
                  onChange={(e) => handleInputChange("assessmentAttempts", e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="coursesCompleted">Courses Completed</Label>
                <Input
                  id="coursesCompleted"
                  type="number"
                  placeholder="e.g., 3"
                  value={formData.coursesCompleted}
                  onChange={(e) => handleInputChange("coursesCompleted", e.target.value)}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="lastActivityDays">Days Since Last Activity</Label>
                <Input
                  id="lastActivityDays"
                  type="number"
                  placeholder="e.g., 7"
                  value={formData.lastActivityDays}
                  onChange={(e) => handleInputChange("lastActivityDays", e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="complianceRating">Compliance Rating (%)</Label>
                <Input
                  id="complianceRating"
                  type="number"
                  placeholder="e.g., 85"
                  value={formData.complianceRating}
                  onChange={(e) => handleInputChange("complianceRating", e.target.value)}
                />
              </div>
            </div>

            <div className="flex gap-3 pt-4">
              <Button 
                onClick={handleAnalyze} 
                disabled={loading}
                className="flex-1"
              >
                {loading ? "Analyzing..." : "Analyze Risk"}
              </Button>
              <Button variant="outline" onClick={resetForm}>
                Reset
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        <Card className="h-fit">
          <CardHeader>
            <CardTitle>Risk Analysis Results</CardTitle>
            <CardDescription>
              AI-powered risk assessment and intervention recommendations.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!riskAnalysis ? (
              <div className="text-center py-12 text-gray-500">
                <TrendingUp className="h-16 w-16 mx-auto mb-4 opacity-30" />
                <p>Enter performance data and click "Analyze Risk" to get AI-powered risk assessment.</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Risk Level */}
                <div className={`text-center p-6 rounded-lg border-2 ${getRiskBgColor(riskAnalysis.riskLevel)}`}>
                  <div className="flex items-center justify-center gap-3 mb-3">
                    {riskAnalysis.riskLevel === "LOW" ? (
                      <CheckCircle className="h-12 w-12 text-green-600" />
                    ) : riskAnalysis.riskLevel === "MEDIUM" ? (
                      <AlertTriangle className="h-12 w-12 text-yellow-600" />
                    ) : (
                      <XCircle className="h-12 w-12 text-red-600" />
                    )}
                    <div>
                      <h3 className={`text-2xl font-bold ${getRiskColor(riskAnalysis.riskLevel)}`}>
                        {riskAnalysis.riskLevel} RISK
                      </h3>
                      <p className="text-gray-600">
                        Risk Score: {riskAnalysis.riskScore}%
                      </p>
                      {riskAnalysis.model_used && (
                        <p className="text-xs text-blue-600">
                          Model: {riskAnalysis.model_used}
                        </p>
                      )}
                    </div>
                  </div>
                  <Progress 
                    value={riskAnalysis.riskScore} 
                    className="w-full h-3 mb-3"
                  />
                  <Badge 
                    variant={riskAnalysis.riskLevel === "LOW" ? "default" : 
                           riskAnalysis.riskLevel === "MEDIUM" ? "secondary" : "destructive"}
                    className="text-sm px-4 py-1"
                  >
                    {riskAnalysis.riskLevel === "LOW" ? "Continue Current Path" : 
                     riskAnalysis.riskLevel === "MEDIUM" ? "Monitor Closely" : "Immediate Intervention Required"}
                  </Badge>
                </div>

                {/* Risk Factors */}
                <div>
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 text-orange-600" />
                    Risk Factors Identified
                  </h4>
                  <div className="space-y-2">
                    {riskAnalysis.factors.map((factor, index) => (
                      <div key={index} className="flex items-center gap-2 p-3 bg-orange-50 rounded-lg">
                        <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                        <span className="text-sm text-gray-700">{factor}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Interventions */}
                <div>
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                    Recommended Interventions
                  </h4>
                  <div className="space-y-2">
                    {riskAnalysis.interventions.map((intervention, index) => (
                      <div key={index} className="flex items-center gap-2 p-3 bg-green-50 rounded-lg">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="text-sm text-gray-700">{intervention}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default RiskAnalyzer;
