"""
main.py — PHQ-9 ML Pipeline entry point (cloud-optimised).

Additions vs. original
-----------------------
• Structured logging (file + console) via logger.py.
• Per-step timing so you can see in the log exactly how long each stage took.
• Memory snapshots between steps.
• Graceful top-level exception handler: on any unhandled error the full
  traceback is written to the log file (not just stderr) before the process
  exits, so you can read it over SSH without needing to scroll the terminal.
• psutil install check at startup with a helpful message.
"""

import sys
import time
import traceback

from logger import get_logger
from memory_guard import log_memory

log = get_logger("main")


def _check_dependencies() -> None:
    """Warn about optional but strongly recommended packages."""
    try:
        import psutil  # noqa: F401
    except ImportError:
        log.warning(
            "psutil is not installed — memory monitoring will be disabled. "
            "Install it with:  pip install psutil"
        )


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def main() -> None:
    pipeline_start = time.time()
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║          PHQ-9 ML PIPELINE — CLOUD RUN STARTED          ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    _check_dependencies()
    log_memory("Pipeline start")

    # ── Step 1: EDA ───────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("━━━━━━━━━━━━━━━━━━━━  STEP 1: EDA  ━━━━━━━━━━━━━━━━━━━━━━")
    from EDA.EDA import EDA
    eda = EDA()
    engineered_df = eda.run()
    log.info("Step 1 completed in %s", _fmt_elapsed(time.time() - t0))
    log_memory("After EDA")

    # ── Step 2: Unsupervised Learning ─────────────────────────────────────────
    t0 = time.time()
    log.info("━━━━━━━━━━━  STEP 2: UNSUPERVISED LEARNING  ━━━━━━━━━━━━━━")
    from Unsupervised.Unsupervised import UnsupervisedLearning
    unsupervised = UnsupervisedLearning(random_state=42)
    relabeled_df, clustering_metrics = unsupervised.run(engineered_df)
    log.info("Step 2 completed in %s", _fmt_elapsed(time.time() - t0))
    log_memory("After Unsupervised")

    # Free the intermediate engineered_df to recover RAM before supervised step
    del engineered_df
    log.info("[main] Released engineered_df from memory.")
    log_memory("After engineered_df release")

    # ── Step 3: Supervised Learning ───────────────────────────────────────────
    t0 = time.time()
    log.info("━━━━━━━━━━━━  STEP 3: SUPERVISED LEARNING  ━━━━━━━━━━━━━━━")
    from Supervised.supervised import PHQ9ModelTrainer
    trainer = PHQ9ModelTrainer(
        target_reg="phq9_total",
        target_clf="cluster_label",
        random_state=42,
    )
    supervised_metrics = trainer.run_pipeline(relabeled_df)
    log.info("Step 3 completed in %s", _fmt_elapsed(time.time() - t0))
    log_memory("After Supervised")

    # ── Step 4: Metrics export ────────────────────────────────────────────────
    t0 = time.time()
    log.info("━━━━━━━━━━  STEP 4: EVALUATING & EXPORTING METRICS  ━━━━━━")
    from IndexScores.indexes_scores import IndexesScoresEvaluator
    all_metrics = {**supervised_metrics, **clustering_metrics}
    evaluator = IndexesScoresEvaluator(
        metrics_dict=all_metrics, output_dir="output_metrics",
    )
    evaluator.run_evaluation()
    log.info("Step 4 completed in %s", _fmt_elapsed(time.time() - t0))

    # ── Done ──────────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   PIPELINE COMPLETED SUCCESSFULLY in %-20s ║", _fmt_elapsed(total))
    log.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Write the full traceback to the log file so it's readable over SSH
        tb = traceback.format_exc()
        log.critical("PIPELINE FAILED — unhandled exception:\n%s", tb)
        sys.exit(1)
