"""Pipeline adapter — periodic analysis, config persistence, drift tracking.

Designed for CI/CD and DVC pipelines.  Analyzes data groups, detects drift
against previous baselines, and saves configs to disk and/or S3.

Usage:
    adapter = EstimatorAdapter(output_dir="configs/")
    adapter.run(
        groups={"demand": demand_data, "latency": latency_data},
        reference={"demand": last_week_demand, "latency": last_week_latency},
    )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from estimator_detector.detector import DetectorConfig, UniversalEstimatorDetector
from estimator_detector.drift import DriftDetector

logger = logging.getLogger(__name__)


class EstimatorAdapter:
    """Pipeline adapter for periodic estimator detection and drift tracking.

    Encapsulates one or more detectors with different asymmetry ratios
    for different workload types.

    Args:
        output_dir: Local directory for config JSON files.
        s3_bucket: Optional S3 bucket for cloud persistence.
        s3_prefix: S3 key prefix.
        default_ratio: Default asymmetry ratio.
        group_ratios: Dict of {group_name: asymmetry_ratio} for per-group tuning.
        min_samples: Minimum samples for analysis.
        drift_threshold: Wasserstein drift threshold (IQR-relative).
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        s3_bucket: str | None = None,
        s3_prefix: str = "estimator_configs/",
        default_ratio: float = 3.0,
        group_ratios: dict[str, float] | None = None,
        min_samples: int = 15,
        drift_threshold: float = 0.10,
        # Legacy compat
        rt_penalty: float | None = None,
        training_penalty: float | None = None,
        batch_penalty: float | None = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.default_ratio = default_ratio
        self.group_ratios = group_ratios or {}
        self.min_samples = min_samples
        self.drift_detector = DriftDetector(threshold=drift_threshold)

        # Legacy support: build group_ratios from old-style penalty params
        if rt_penalty is not None:
            self.group_ratios.setdefault("rt", rt_penalty)
        if training_penalty is not None:
            self.group_ratios.setdefault("training", training_penalty)
        if batch_penalty is not None:
            self.group_ratios.setdefault("batch", batch_penalty)

        self._detectors: dict[str, UniversalEstimatorDetector] = {}

    def _get_detector(self, group: str) -> UniversalEstimatorDetector:
        if group not in self._detectors:
            ratio = self.group_ratios.get(group, self.default_ratio)
            self._detectors[group] = UniversalEstimatorDetector(
                config=DetectorConfig(
                    asymmetry_ratio=ratio,
                    min_samples=self.min_samples,
                )
            )
        return self._detectors[group]

    def run(
        self,
        groups: dict[str, np.ndarray | list] | None = None,
        reference: dict[str, np.ndarray] | None = None,
        # Legacy interface
        rt_data: dict[tuple, np.ndarray] | None = None,
        training_data: dict[str, np.ndarray] | None = None,
        batch_data: np.ndarray | None = None,
        custom_data: dict[str, np.ndarray] | None = None,
        custom_penalty: float = 3.0,
    ) -> dict[str, Any]:
        """Run full analysis pipeline.

        New API:
            adapter.run(groups={"demand": data, "latency": data2})

        Legacy API (still supported):
            adapter.run(rt_data=..., training_data=..., batch_data=...)

        Returns:
            Dict with all recommendations, serializable to JSON.
        """
        results: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
        }

        # New-style groups
        if groups is not None:
            for name, values in groups.items():
                detector = self._get_detector(name)
                ref = reference.get(name) if reference else None
                rec = detector.analyze(values, reference=ref)
                results[name] = rec.to_dict()
                logger.info(f"Group '{name}': {rec.summary()}")

        # Legacy: RT data (binned)
        if rt_data is not None:
            rt_results = {}
            detector = self._get_detector("rt")
            for (dow, hour), values in rt_data.items():
                rec = detector.analyze(values)
                rt_results[f"{dow}_{hour}"] = rec.to_dict()
            results["inference_rt"] = rt_results

        # Legacy: Training data
        if training_data is not None:
            train_results = {}
            detector = self._get_detector("training")
            for train_type, values in training_data.items():
                rec = detector.analyze(values)
                train_results[train_type] = rec.to_dict()
                logger.info(f"Training {train_type}: {rec.summary()}")
            results["training"] = train_results

        # Legacy: Batch
        if batch_data is not None:
            detector = self._get_detector("batch")
            rec = detector.analyze(batch_data)
            results["batch"] = rec.to_dict()

        # Legacy: Custom
        if custom_data is not None:
            custom_det = UniversalEstimatorDetector(asymmetry_ratio=custom_penalty)
            for name, values in custom_data.items():
                rec = custom_det.analyze(values)
                results.setdefault("custom", {})[name] = rec.to_dict()

        # Persist
        self._save(results)
        return results

    def _save(self, results: dict) -> None:
        payload = json.dumps(results, indent=2, ensure_ascii=False)

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "estimator_configs.json"
            path.write_text(payload, encoding="utf-8")
            logger.info(f"Saved to {path}")

        if self.s3_bucket:
            self._upload_s3(payload)

    def _upload_s3(self, payload: str) -> None:
        try:
            import boto3

            s3 = boto3.client("s3")
            key = f"{self.s3_prefix}estimator_configs.json"
            s3.put_object(Bucket=self.s3_bucket, Key=key, Body=payload.encode("utf-8"))
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_key = f"{self.s3_prefix}history/{ts}.json"
            s3.put_object(
                Bucket=self.s3_bucket, Key=backup_key, Body=payload.encode("utf-8")
            )
        except ImportError:
            logger.warning("boto3 not installed. S3 upload skipped.")
        except Exception as e:
            logger.error(f"S3 upload error: {e}")
