from EDA.EDA import EDA
from Unsupervised.Unsupervised import UnsupervisedLearning
from Supervised.supervised import PHQ9ModelTrainer
from IndexScores.indexes_scores import IndexesScoresEvaluator

if __name__ == "__main__":
    # Step 1: Exploratory Data Analysis and Feature Engineering
    print("\nStep 1: Exploratory Data Analysis")
    eda = EDA()
    engineered_df = eda.run()
 
    # Step 2: Unsupervised Learning
    print("\nStep 2: Unsupervised Learning")
    unsupervised = UnsupervisedLearning(random_state=42)
    relabeled_df, clustering_metrics = unsupervised.run(engineered_df)
 
    # Step 3: Supervised Learning
    print("\nStep 3: Supervised Learning")
    trainer = PHQ9ModelTrainer(
        target_reg="phq9_total",
        target_clf="cluster_label",
        random_state=42,
    )
    supervised_metrics = trainer.run_pipeline(relabeled_df)
 
    # Step 4: Aggregate and export all metrics
    print("\nStep 4: Evaluating and Exporting Metrics")
    all_metrics = {**supervised_metrics, **clustering_metrics}
 
    evaluator = IndexesScoresEvaluator(
        metrics_dict=all_metrics,
        output_dir="output_metrics",
    )
    evaluator.run_evaluation()
 
    print("\nPipeline completed successfully.")
