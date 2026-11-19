"""Verify wheel integrity and required dependencies for Linux Python 3.10."""
import hashlib
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

from pip._vendor.packaging.markers import default_environment
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name

root = Path(sys.argv[1])
packages = {}
checksums = []
for path in sorted(root.glob("*.whl")):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, path
        metadata = BytesParser().parsebytes(archive.read(next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )))
    packages[canonicalize_name(metadata["Name"])] = metadata
    with path.open("rb") as source:
        checksums.append(f"{hashlib.file_digest(source, 'sha256').hexdigest()}  {path.name}")

environment = default_environment()
environment.update(python_version="3.10", python_full_version="3.10.19",
                   sys_platform="linux", platform_system="Linux",
                   platform_machine="x86_64", os_name="posix", extra="")
for name, metadata in packages.items():
    for value in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        dependency = packages.get(canonicalize_name(requirement.name))
        assert dependency is not None, (name, str(requirement), "missing")
        assert requirement.specifier.contains(dependency["Version"]), (name, str(requirement))

(root / "SHA256SUMS").write_bytes(("\n".join(checksums) + "\n").encode("utf-8"))
print(f"Verified {len(packages)} wheels and Linux Python 3.10 dependency closure")
