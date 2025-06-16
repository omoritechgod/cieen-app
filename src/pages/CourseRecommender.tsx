
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { BookOpen, Star, Clock, TrendingUp } from "lucide-react";
import { toast } from "sonner";

interface Course {
  id: number;
  title: string;
  category: string;
  difficulty: string;
  duration: number;
  rating: number;
  match_score: number;
  description: string;
}

const CourseRecommender = () => {
  const [formData, setFormData] = useState({
    specialization: "",
    experienceLevel: "",
    learningGoals: ""
  });
  
  const [recommendations, setRecommendations] = useState<{
    recommendations: Course[];
    member_profile: any;
    total_matches: number;
    model_used: string;
  } | null>(null);
  
  const [loading, setLoading] = useState(false);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleGetRecommendations = async () => {
    if (!formData.specialization || !formData.experienceLevel) {
      toast.error("Please fill in specialization and experience level");
      return;
    }

    setLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/recommendations/courses', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to get recommendations');
      }

      const result = await response.json();
      setRecommendations(result);
      toast.success("Course recommendations generated successfully!");
    } catch (error) {
      console.error('Error:', error);
      toast.error("Failed to get recommendations. Please ensure the API server is running.");
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      specialization: "",
      experienceLevel: "",
      learningGoals: ""
    });
    setRecommendations(null);
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case "beginner": return "bg-green-100 text-green-800";
      case "intermediate": return "bg-yellow-100 text-yellow-800";
      case "advanced": return "bg-red-100 text-red-800";
      default: return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <BookOpen className="h-8 w-8 text-purple-600" />
          <h1 className="text-3xl font-bold text-gray-900">Course Recommendation System</h1>
        </div>
        <p className="text-gray-600 max-w-3xl">
          Intelligent recommendation engine that suggests upskilling or remedial courses based on 
          learning history, career goals, identified knowledge gaps, and AI model predictions.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Input Form */}
        <Card className="h-fit">
          <CardHeader>
            <CardTitle>Member Profile</CardTitle>
            <CardDescription>
              Enter your professional details to get personalized course recommendations.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="specialization">Specialization Area</Label>
              <Select value={formData.specialization} onValueChange={(value) => handleInputChange("specialization", value)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select specialization" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="power-engineering">Power Engineering</SelectItem>
                  <SelectItem value="electronics">Electronics & Communication</SelectItem>
                  <SelectItem value="control-systems">Control Systems</SelectItem>
                  <SelectItem value="renewable-energy">Renewable Energy</SelectItem>
                  <SelectItem value="telecommunications">Telecommunications</SelectItem>
                  <SelectItem value="computer-engineering">Computer Engineering</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="experienceLevel">Experience Level</Label>
              <Select value={formData.experienceLevel} onValueChange={(value) => handleInputChange("experienceLevel", value)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select experience level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="entry-level">Entry Level (0-2 years)</SelectItem>
                  <SelectItem value="intermediate">Intermediate (3-7 years)</SelectItem>
                  <SelectItem value="senior">Senior (8-15 years)</SelectItem>
                  <SelectItem value="expert">Expert (15+ years)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="learningGoals">Learning Goals</Label>
              <Input
                id="learningGoals"
                placeholder="e.g., Certification preparation, Skill upgrade"
                value={formData.learningGoals}
                onChange={(e) => handleInputChange("learningGoals", e.target.value)}
              />
            </div>

            <div className="flex gap-3 pt-4">
              <Button 
                onClick={handleGetRecommendations} 
                disabled={loading}
                className="flex-1"
              >
                {loading ? "Generating..." : "Get Recommendations"}
              </Button>
              <Button variant="outline" onClick={resetForm}>
                Reset
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Recommended Courses</CardTitle>
              <CardDescription>
                AI-powered course recommendations tailored to your profile and goals.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!recommendations ? (
                <div className="text-center py-12 text-gray-500">
                  <BookOpen className="h-16 w-16 mx-auto mb-4 opacity-30" />
                  <p>Enter your profile details and click "Get Recommendations" to see personalized course suggestions.</p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Summary */}
                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingUp className="h-5 w-5 text-purple-600" />
                      <h4 className="font-semibold text-purple-800">Recommendation Summary</h4>
                    </div>
                    <p className="text-sm text-purple-700">
                      Found {recommendations.total_matches} courses matching your profile. 
                      Model: {recommendations.model_used}
                    </p>
                  </div>

                  {/* Course List */}
                  <div className="grid gap-4">
                    {recommendations.recommendations.map((course) => (
                      <Card key={course.id} className="border border-gray-200 hover:border-purple-300 transition-colors">
                        <CardContent className="p-4">
                          <div className="flex justify-between items-start mb-3">
                            <div className="flex-1">
                              <h3 className="font-semibold text-lg text-gray-900 mb-1">
                                {course.title}
                              </h3>
                              <p className="text-sm text-gray-600 mb-2">
                                {course.description}
                              </p>
                              <div className="flex flex-wrap gap-2 mb-3">
                                <Badge variant="outline" className="text-xs">
                                  {course.category}
                                </Badge>
                                <Badge 
                                  className={`text-xs ${getDifficultyColor(course.difficulty)}`}
                                >
                                  {course.difficulty}
                                </Badge>
                              </div>
                            </div>
                            <div className="text-right ml-4">
                              <div className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm font-semibold mb-2">
                                {course.match_score.toFixed(1)}% Match
                              </div>
                            </div>
                          </div>
                          
                          <div className="flex items-center justify-between text-sm text-gray-600">
                            <div className="flex items-center gap-4">
                              <div className="flex items-center gap-1">
                                <Clock className="h-4 w-4" />
                                <span>{course.duration} hours</span>
                              </div>
                              <div className="flex items-center gap-1">
                                <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                                <span>{course.rating}</span>
                              </div>
                            </div>
                            <Button size="sm" variant="outline">
                              Enroll Now
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default CourseRecommender;
