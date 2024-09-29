from experiments.feature_sensitivity import FeatureSensitivityExperiment
from experiments.planning_ahead import PlanningAheadExperiment

def main():
    FeatureSensitivityExperiment.run_default()
    PlanningAheadExperiment.run_default()

if __name__ == "__main__":
    main()