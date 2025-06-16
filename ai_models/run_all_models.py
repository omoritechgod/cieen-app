
"""
Master script to run all AI models training and evaluation
"""
import os
import sys
import subprocess
import time

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        end_time = time.time()
        
        print(result.stdout)
        if result.stderr:
            print("WARNINGS/ERRORS:")
            print(result.stderr)
        
        print(f"\n✅ {description} completed successfully in {end_time - start_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}:")
        print(e.stdout)
        print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error running {description}: {str(e)}")
        return False

def main():
    """Run all AI model training scripts"""
    print("CIEEEN AI-LMS Model Training Pipeline")
    print("=====================================")
    
    # Ensure directories exist
    os.makedirs('ai_models/plots', exist_ok=True)
    os.makedirs('ai_models/trained_models', exist_ok=True)
    os.makedirs('ai_models/datasets', exist_ok=True)
    
    # Change to ai_models directory
    original_dir = os.getcwd()
    
    scripts_to_run = [
        ('datasets/download_data.py', 'Dataset Generation'),
        ('certification_predictor.py', 'Certification Eligibility Predictor'),
        ('risk_analyzer.py', 'Performance Risk Analyzer'),
        ('course_recommender.py', 'Course Recommendation System')
    ]
    
    successful_runs = 0
    total_scripts = len(scripts_to_run)
    
    start_total_time = time.time()
    
    for script_path, description in scripts_to_run:
        if run_script(script_path, description):
            successful_runs += 1
        else:
            print(f"⚠️  Continuing with next script despite error in {description}")
    
    end_total_time = time.time()
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Total scripts run: {total_scripts}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {total_scripts - successful_runs}")
    print(f"Total execution time: {end_total_time - start_total_time:.2f} seconds")
    
    if successful_runs == total_scripts:
        print("\n🎉 All AI models trained successfully!")
        print("\nNext steps:")
        print("1. Check 'ai_models/plots/' for all generated visualizations")
        print("2. Check 'ai_models/trained_models/' for saved model files")
        print("3. Review the individual model outputs above")
        print("4. Use the models in your web application")
    else:
        print(f"\n⚠️  {total_scripts - successful_runs} script(s) failed. Check the error messages above.")
    
    # Display available outputs
    print(f"\nGenerated files:")
    
    plots_dir = 'ai_models/plots'
    if os.path.exists(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        if plot_files:
            print(f"\nPlots ({len(plot_files)} files):")
            for plot_file in plot_files:
                print(f"  📊 {plot_file}")
    
    models_dir = 'ai_models/trained_models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if model_files:
            print(f"\nTrained Models ({len(model_files)} files):")
            for model_file in model_files:
                print(f"  🤖 {model_file}")
    
    datasets_dir = 'ai_models/datasets'
    if os.path.exists(datasets_dir):
        dataset_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
        if dataset_files:
            print(f"\nDatasets ({len(dataset_files)} files):")
            for dataset_file in dataset_files:
                print(f"  📈 {dataset_file}")

if __name__ == "__main__":
    main()
