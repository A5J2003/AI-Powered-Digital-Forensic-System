import os
import json
import hashlib
import datetime


class CaseManager:
    def __init__(self, base_dir="output"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.case_id = f"case_{timestamp}"
        self.base_path = os.path.join(base_dir, self.case_id)

        self.paths = {
            "input":    os.path.join(self.base_path, "input"),
            "frames":   os.path.join(self.base_path, "extracted/frames"),
            "audio":    os.path.join(self.base_path, "extracted/audio"),
            "text":     os.path.join(self.base_path, "extracted/text"),
            "metadata": os.path.join(self.base_path, "metadata"),
            "results":  os.path.join(self.base_path, "results"),
            "explain":  os.path.join(self.base_path, "explainability"),
            "logs":     os.path.join(self.base_path, "logs"),
            "report":   os.path.join(self.base_path, "report"),
        }

        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

        # CoC log lives here
        self._coc_path = os.path.join(self.paths["logs"], "chain_of_custody.json")
        self._coc_log = []

    # --------------------------------------------------
    # PATH HELPER
    # --------------------------------------------------
    def get_path(self, key):
        return self.paths[key]

    # --------------------------------------------------
    # SHA-256
    # --------------------------------------------------
    @staticmethod
    def compute_sha256(file_path):
        if not file_path or not os.path.isfile(file_path):
            return None

        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # --------------------------------------------------
    # CoC ENTRY LOGGER
    # --------------------------------------------------
    def log_coc(
        self,
        stage,
        file_path,
        modality=None,
        action=None,
        handler="system",
        notes=None,
        extra=None
    ):
        entry = {
            "sequence":  len(self._coc_log) + 1,
            "case_id":   self.case_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "stage":     stage,
            "modality":  modality,
            "action":    action,
            "handler":   handler,
            "file": {
                "path":   file_path,
                "sha256": self.compute_sha256(file_path),
            },
            "notes": notes,
        }

        if extra:
            entry["extra"] = extra

        self._coc_log.append(entry)
        self._flush_coc()
        return entry

    # --------------------------------------------------
    # PERSIST (FIXED)
    # --------------------------------------------------
    def _flush_coc(self):
        """Writes the full CoC log to disk after every entry (safe for crashes)."""

        import numpy as np

        def convert(obj):
            # ✅ Fix: handle numpy types
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)  # fallback (safe)

        with open(self._coc_path, "w") as f:
            json.dump(
                {
                    "case_id":           self.case_id,
                    "generated_at":      datetime.datetime.utcnow().isoformat() + "Z",
                    "total_entries":     len(self._coc_log),
                    "chain_of_custody":  self._coc_log,
                },
                f,
                indent=4,
                default=convert   # ✅ CRITICAL FIX
            )

    # --------------------------------------------------
    # CONVENIENCE: hash-only (no log entry)
    # --------------------------------------------------
    def hash_file(self, file_path):
        return self.compute_sha256(file_path)