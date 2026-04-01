from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import cv2


_FILENAME_PATTERN = re.compile(
    r"^(?P<camera_id>[^_]+)_(?P<index>\d+)_(?P<timestamp>[^.]+)\.png$"
)


@dataclass(frozen=True)
class FrameRecord:
    """Metadata for one image frame in the local dataset."""

    camera_id: str
    modality: str
    index: int
    timestamp: str
    path: Path


class LocalMultiCameraDataset:
    """Read and index local multi-camera image data.

    Expected folder layout:
        root/
          CameraA/
            Color/
              CameraA_0001_1712130000.png
            Depth/
              CameraA_0001_1712130000.png
          CameraB/
            Color/
            Depth/
    """

    VALID_MODALITIES = ("Color", "Depth")

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        self._records: Dict[str, Dict[str, List[FrameRecord]]] = {}
        self._build_index()

    def _build_index(self) -> None:
        for camera_dir in sorted(self.root_dir.iterdir()):
            if not camera_dir.is_dir():
                continue

            camera_id = camera_dir.name
            camera_records: Dict[str, List[FrameRecord]] = {}

            for modality in self.VALID_MODALITIES:
                modality_dir = camera_dir / modality
                if not modality_dir.exists() or not modality_dir.is_dir():
                    continue

                records: List[FrameRecord] = []
                for image_path in sorted(modality_dir.glob("*.png")):
                    match = _FILENAME_PATTERN.match(image_path.name)
                    if not match:
                        continue

                    file_camera_id = match.group("camera_id")
                    if file_camera_id != camera_id:
                        continue

                    records.append(
                        FrameRecord(
                            camera_id=file_camera_id,
                            modality=modality,
                            index=int(match.group("index")),
                            timestamp=match.group("timestamp"),
                            path=image_path,
                        )
                    )

                records.sort(key=lambda x: x.index)
                camera_records[modality] = records

            if camera_records:
                self._records[camera_id] = camera_records

    def cameras(self) -> List[str]:
        return sorted(self._records.keys())

    def modalities(self, camera_id: str) -> List[str]:
        if camera_id not in self._records:
            raise KeyError(f"Camera not found: {camera_id}")
        return sorted(self._records[camera_id].keys())

    def size(self, camera_id: str, modality: str) -> int:
        return len(self._get_records(camera_id, modality))

    def get_record(self, camera_id: str, modality: str, index: int) -> FrameRecord:
        records = self._get_records(camera_id, modality)
        if index < 0 or index >= len(records):
            raise IndexError(
                f"Index out of range for {camera_id}/{modality}: {index}, total={len(records)}"
            )
        return records[index]

    def read_image(self, camera_id: str, modality: str, index: int):
        record = self.get_record(camera_id, modality, index)
        img = cv2.imread(str(record.path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {record.path}")
        return img

    def find_by_timestamp(
        self, camera_id: str, modality: str, timestamp: str
    ) -> Optional[FrameRecord]:
        for record in self._get_records(camera_id, modality):
            if record.timestamp == timestamp:
                return record
        return None

    def iter_records(
        self, camera_id: Optional[str] = None, modality: Optional[str] = None
    ) -> Iterable[FrameRecord]:
        camera_ids = [camera_id] if camera_id else self.cameras()

        for cid in camera_ids:
            if cid not in self._records:
                continue
            modalities = [modality] if modality else sorted(self._records[cid].keys())
            for m in modalities:
                for record in self._records[cid].get(m, []):
                    yield record

    def _get_records(self, camera_id: str, modality: str) -> List[FrameRecord]:
        if camera_id not in self._records:
            raise KeyError(f"Camera not found: {camera_id}")
        if modality not in self._records[camera_id]:
            raise KeyError(f"Modality not found: {camera_id}/{modality}")
        return self._records[camera_id][modality]


def demo_access_dataset(root_dir: str | Path) -> None:
    """Demo: index dataset and access one Color + one Depth frame per camera."""

    dataset = LocalMultiCameraDataset(root_dir)

    print(f"Dataset root: {Path(root_dir).resolve()}")
    print(f"Cameras: {dataset.cameras()}")

    for camera_id in dataset.cameras():
        print(f"\\nCamera: {camera_id}")
        for modality in dataset.modalities(camera_id):
            n = dataset.size(camera_id, modality)
            print(f"  - {modality}: {n} frames")
            if n == 0:
                continue

            record = dataset.get_record(camera_id, modality, 0)
            image = dataset.read_image(camera_id, modality, 0)
            print(
                f"    first frame => index={record.index}, "
                f"timestamp={record.timestamp}, "
                f"shape={tuple(image.shape)}, "
                f"path={record.path}"
            )


if __name__ == "__main__":
    # Example:
    # python -m util.local_dataset /path/to/dataset_root
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m util.local_dataset <dataset_root>")
        raise SystemExit(1)

    demo_access_dataset(sys.argv[1])
