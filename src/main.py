import argparse
import logging
from pathlib import Path

from src.train import run_cross_validation, run_training, run_evaluation
from src.generate_results import generate_all_plots
from src.utils import (
    load_config,
    save_cv_summary,
    save_train_results,
    save_eval_results,
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_experiment_dir(results_dir: Path, model_name: str, config_name: str) -> Path:
    return results_dir / model_name / config_name


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML models on stock market prediction."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to results directory.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run cross-validation to evaluate hyperparameters.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model on full training set and save.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate most recent trained model on eval set.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots for all evaluated models.",
    )
    args = parser.parse_args()

    if not any([args.tune, args.train, args.test, args.plot]):
        parser.error("At least one of --tune, --train, --test, or --plot is required.")

    config = load_config(args.config)
    config_name = config["config_name"]
    windows = config["windows"]
    models = config["models"]

    logger.info(f"Using config: {config_name} with {len(models)} model(s) and {len(windows)} window(s)")

    for i, model_conf in enumerate(models):
        model_name = model_conf["name"]
        logger.info(f"Processing model {i+1}/{len(models)}: {model_name}")
        experiment_dir = get_experiment_dir(args.results_dir, model_name, config_name)

        if args.tune:
            logger.info(f"Running cross-validation for {model_name}...")
            summary_df = run_cross_validation(model_conf, windows)
            save_cv_summary(summary_df, model_conf, experiment_dir)
            logger.info(f"CV summary saved for {model_name}:\n{summary_df}")

        if args.train:
            logger.info(f"Training {model_name} on full training set...")
            results_df, summary_df, model_paths = run_training(
                model_conf, windows, experiment_dir
            )
            save_train_results(results_df, summary_df, model_conf, experiment_dir)
            logger.info(f"Training completed for {model_name}. Summary:\n{summary_df}")
            logger.info(f"Models saved: {[str(p) for p in model_paths]}")

        if args.test:
            logger.info(f"Evaluating {model_name}...")
            results_df, summary_df = run_evaluation(model_conf, windows, experiment_dir)
            save_eval_results(results_df, summary_df, experiment_dir)
            logger.info(f"Evaluation completed for {model_name}. Summary:\n{summary_df}")

        logger.info(f"Completed processing for model: {model_name}")

    if args.plot:
        logger.info("Generating plots...")
        generate_all_plots(args.results_dir, config_name)
        logger.info("Plots generated")

    logger.info("All tasks completed!")


if __name__ == "__main__":
    main()
