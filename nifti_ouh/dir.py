from pathlib import Path
import os
import logging
from nifti_ouh.file import File

log = logging.getLogger(__name__)


class Dir:
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        if not path.exists():
            log.error(f'The path {path} does not exist')

        self.files = []

    def scan_dir(self):
        log.info(f'Scanning {self.path}')

        self.files = []
        for root, dirs, files in os.walk(self.path, topdown=False):
            for name in files:
                if name.endswith('.nii.gz'):
                    self.files.append(File(Path(root).joinpath(name)))

        log.info(f'Found {len(self.files)} nifti files in {self.path}')

    def load_all_headers(self):
        for f in self.files:
            f.load_header()

    def get_file(self, index: int) -> File:
        return self.files[index]
