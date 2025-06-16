
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Brain, CheckCircle, XCircle, AlertCircle } from "lucide-react";
import { toast } from "sonner";

const CertificationPredictor = () => {
  const [formData, setFormData] = useState({
    trainingHours: "",
    assessmentScore: "",
    complianceHistory: "",
    continuingEducation: "",
    yearsExperience: "",
    previousCertifications: ""
  });
  
  const [prediction, setPrediction] = useState<{
    approved: boolean;
    confidence: number;
    factors: string[];
    model_used?: string;
  } | null>(null);
  
  const [loading, setLoading] = useState(false);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    if (!formData.trainingHours || !formData.assessmentScore) {
      toast.error("Please fill in at least training hours and assessment score");
      return;
    }

    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/certification/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const result = await response.json();
      setPrediction(result);
      toast.success("Prediction completed successfully!");
    } catch (error) {
      console.error('Error:', error);
      toast.error("Failed to get prediction. Please ensure the API server is running.");
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      trainingHours: "",
      assessmentScore: "",
      complianceHistory: "",
      continuingEducation: "",
      yearsExperience: "",
      previousCertifications: ""
    });
    setPrediction(null);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <Brain className="h-8 w-8 text-blue-600" />
          <h1 className="text-3xl font-bold text-gray-900">Certification Eligibility Predictor</h1>
        </div>
        <p className="text-gray-600 max-w-3xl">
          AI-powered classification model that determines whether a professional qualifies for certification 
          based on training data, assessment scores, compliance history, and continuing education activities.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <Card className="h-fit">
          <CardHeader>
            <CardTitle>Member Information</CardTitle>
            <CardDescription>
              Enter the member's professional development data for certification eligibility assessment.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="trainingHours">Training Hours Completed</Label>
                <Input
                  id="trainingHours"
                  type="number"
                  placeholder="e.g., 45"
                  value={formData.trainingHours}
                  onChange={(e) => handleInputChange("trainingHours", e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="assessmentScore">Assessment Score (%)</Label>
                <Input
                  id="assessmentScore"
                  type="number"
                  placeholder="e.g., 85"
                  value={formData.assessmentScore}
                  onChange={(e) => handleInputChange("assessmentScore", e.target.value)}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="complianceHistory">Compliance Score (%)</Label>
                <Input
                  id="complianceHistory"
                  type="number"
                  placeholder="e.g., 90"
                  value={formData.complianceHistory}
                  onChange={(e) => handleInputChange("complianceHistory", e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="continuingEducation">Continuing Education Hours</Label>
                <Input
                  id="continuingEducation"
                  type="number"
                  placeholder="e.g., 25"
                  value={formData.continuingEducation}
                  onChange={(e) => handleInputChange("continuingEducation", e.target.value)}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="yearsExperience">Years of Experience</Label>
                <Input
                  id="yearsExperience"
                  type="number"
                  placeholder="e.g., 5"
                  value={formData.yearsExperience}
                  onChange={(e) => handleInputChange("yearsExperience", e.target.value)}
                />
              </div>
              <div>
                <Label htmlFor="previousCertifications">Previous Certifications</Label>
                <Input
                  id="previousCertifications"
                  type="number"
                  placeholder="e.g., 2"
                  value={formData.previousCertifications}
                  onChange={(e) => handleInputChange("previousCertifications", e.target.value)}
                />
              </div>
            </div>

            <div className="flex gap-3 pt-4">
              <Button 
                onClick={handlePredict} 
                disabled={loading}
                className="flex-1"
              >
                {loading ? "Analyzing..." : "Predict Eligibility"}
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
            <CardTitle>Prediction Results</CardTitle>
            <CardDescription>
              AI model analysis and certification eligibility assessment.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!prediction ? (
              <div className="text-center py-12 text-gray-500">
                <Brain className="h-16 w-16 mx-auto mb-4 opacity-30" />
                <p>Enter member data and click "Predict Eligibility" to get AI-powered results.</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Main Result */}
                <div className="text-center p-6 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-center gap-3 mb-3">
                    {prediction.approved ? (
                      <CheckCircle className="h-12 w-12 text-green-600" />
                    ) : (
                      <XCircle className="h-12 w-12 text-red-600" />
                    )}
                    <div>
                      <h3 className="text-2xl font-bold">
                        {prediction.approved ? "APPROVED" : "NOT APPROVED"}
                      </h3>
                      <p className="text-gray-600">
                        Confidence: {prediction.confidence}%
                      </p>
                      {prediction.model_used && (
                        <p className="text-xs text-blue-600">
                          Model: {prediction.model_used}
                        </p>
                      )}
                    </div>
                  </div>
                  <Badge 
                    variant={prediction.approved ? "default" : "destructive"}
                    className="text-sm px-4 py-1"
                  >
                    {prediction.approved ? "Eligible for Certification" : "Requires Improvement"}
                  </Badge>
                </div>

                {/* Factors */}
                <div>
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <AlertCircle className="h-5 w-5 text-blue-600" />
                    Assessment Factors
                  </h4>
                  <div className="space-y-2">
                    {prediction.factors.map((factor, index) => (
                      <div key={index} className="flex items-center gap-2 p-2 bg-blue-50 rounded-lg">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        <span className="text-sm text-gray-700">{factor}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Recommendations */}
                {!prediction.approved && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <h4 className="font-semibold text-yellow-800 mb-2">Recommendations</h4>
                    <ul className="text-sm text-yellow-700 space-y-1">
                      <li>• Complete additional training hours to meet requirements</li>
                      <li>• Retake assessments to improve scores</li>
                      <li>• Engage in more continuing education activities</li>
                      <li>• Maintain better compliance with professional standards</li>
                    </ul>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default CertificationPredictor;
