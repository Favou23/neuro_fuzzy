"""training package init

This file makes the training folder importable as a package so scripts
can be executed from the project root (fuzzy_logic) using package imports
like `from training import data_ingestion`.
"""

__all__ = ["data_ingestion", "iec_rule_based", "unified_training", "predict_fault"]
