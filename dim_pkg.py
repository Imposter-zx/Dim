# dim_pkg.py — Package Manager for Dim
#
# Manages dependencies, package resolution, and versioning for Dim projects.

import os
import json
import hashlib
import subprocess
from typing import Optional, Dict, List, Any, Set
from pathlib import Path


class Package:
    def __init__(self, name: str, version: str, source: Optional[str] = None):
        self.name = name
        self.version = version
        self.source = source
        self.dependencies: Dict[str, str] = {}
        self.path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "dependencies": self.dependencies,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Package":
        pkg = Package(data["name"], data["version"], data.get("source"))
        pkg.dependencies = data.get("dependencies", {})
        return pkg


class PackageManifest:
    def __init__(self, name: str, version: str = "0.1.0"):
        self.name = name
        self.version = version
        self.dependencies: Dict[str, str] = {}
        self.dev_dependencies: Dict[str, str] = {}
        self.main: str = "main.dim"
        self.description: str = ""
        self.author: str = ""

    def save(self, path: str):
        data = {
            "name": self.name,
            "version": self.version,
            "main": self.main,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "dev_dependencies": self.dev_dependencies,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> "PackageManifest":
        with open(path, "r") as f:
            data = json.load(f)
        manifest = PackageManifest(data["name"], data.get("version", "0.1.0"))
        manifest.main = data.get("main", "main.dim")
        manifest.description = data.get("description", "")
        manifest.author = data.get("author", "")
        manifest.dependencies = data.get("dependencies", {})
        manifest.dev_dependencies = data.get("dev_dependencies", {})
        return manifest


class PackageRegistry:
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".dim", "packages")
        self.cache_dir = cache_dir
        self.packages: Dict[str, Package] = {}

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def resolve(self, name: str, version: str) -> Optional[Package]:
        key = f"{name}@{version}"
        if key in self.packages:
            return self.packages[key]

        pkg_path = os.path.join(self.cache_dir, name, version)
        if os.path.exists(pkg_path):
            pkg = Package(name, version)
            pkg.path = pkg_path
            self.packages[key] = pkg
            return pkg

        return None

    def install(self, name: str, version: str, source: Optional[str] = None) -> bool:
        pkg_path = os.path.join(self.cache_dir, name, version)

        if os.path.exists(pkg_path):
            print(f"Package {name}@{version} already installed")
            return True

        os.makedirs(pkg_path, exist_ok=True)

        if source:
            if source.startswith("http://") or source.startswith("https://"):
                return self._install_from_url(name, version, source, pkg_path)
            elif os.path.isdir(source):
                return self._install_from_local(name, version, source, pkg_path)

        # Try to fetch from public registry
        registry_url = "https://registry.dimlang.dev"
        print(f"Installing {name}@{version} from registry...")

        try:
            import urllib.request

            url = f"{registry_url}/packages/{name}/{version}/package.zip"
            return self._install_from_url(name, version, url, pkg_path)
        except Exception as e:
            # Fall back to creating a stub package
            pass

        # Create a default/stub package if nothing else works
        manifest_path = os.path.join(pkg_path, "package.json")
        default_manifest = {"name": name, "version": version, "dependencies": {}}
        with open(manifest_path, "w") as f:
            json.dump(default_manifest, f, indent=2)

        dim_path = os.path.join(pkg_path, f"{name}.dim")
        with open(dim_path, "w") as f:
            f.write(f"# {name} v{version}\n# Auto-generated package\n\n")

        return True

    def _install_from_url(self, name: str, version: str, url: str, dest: str) -> bool:
        try:
            import urllib.request
            import zipfile
            import io

            zip_url = (
                url if url.endswith(".zip") else f"{url}/archive/refs/heads/main.zip"
            )

            response = urllib.request.urlopen(zip_url, timeout=30)
            data = response.read()

            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(dest)

            print(f"Installed {name}@{version} from {url}")
            return True
        except Exception as e:
            print(f"Failed to install from URL: {e}")
            return False

    def _install_from_local(self, name: str, version: str, src: str, dest: str) -> bool:
        import shutil

        try:
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dest, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            print(f"Installed {name}@{version} from {src}")
            return True
        except Exception as e:
            print(f"Failed to install from local: {e}")
            return False


class DimPackageManager:
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.manifest: Optional[PackageManifest] = None
        self.registry = PackageRegistry()

        self._load_manifest()

    def _load_manifest(self):
        manifest_path = os.path.join(self.project_path, "package.dim.json")
        if os.path.exists(manifest_path):
            self.manifest = PackageManifest.load(manifest_path)

    def init(self, name: str, main: str = "main.dim"):
        self.manifest = PackageManifest(name)
        self.manifest.main = main

        manifest_path = os.path.join(self.project_path, "package.dim.json")
        self.manifest.save(manifest_path)

        print(f"Initialized package: {name}")
        print(f"  Main file: {main}")
        print(f"  Manifest: {manifest_path}")

    def add_dep(self, name: str, version: str = "*"):
        if self.manifest is None:
            print("Error: Not a Dim package. Run 'dim pkg init' first.")
            return False

        if self.manifest.dependencies.get(name):
            print(f"Dependency {name} already exists")
            return False

        self.manifest.dependencies[name] = version

        manifest_path = os.path.join(self.project_path, "package.dim.json")
        self.manifest.save(manifest_path)

        return self.registry.install(name, version)

    def remove_dep(self, name: str):
        if self.manifest is None:
            print("Error: Not a Dim package.")
            return False

        if name in self.manifest.dependencies:
            del self.manifest.dependencies[name]

            manifest_path = os.path.join(self.project_path, "package.dim.json")
            self.manifest.save(manifest_path)

            print(f"Removed dependency: {name}")
            return True

        print(f"Dependency {name} not found")
        return False

    def install_all(self):
        if self.manifest is None:
            print("Error: Not a Dim package.")
            return

        print("Installing dependencies...")

        for name, version in self.manifest.dependencies.items():
            self.registry.install(name, version)

    def resolve_deps(self) -> Dict[str, Package]:
        if self.manifest is None:
            return {}

        resolved: Dict[str, Package] = {}

        for name, version in self.manifest.dependencies.items():
            pkg = self.registry.resolve(name, version)
            if pkg:
                resolved[name] = pkg
            else:
                if self.registry.install(name, version):
                    pkg = self.registry.resolve(name, version)
                    if pkg:
                        resolved[name] = pkg

        return resolved


def run_pkg(args: List[str]):
    if not args:
        print("Usage: dim pkg <command> [options]")
        print("Commands:")
        print("  init <name>       Initialize a new package")
        print("  add <name> [ver]  Add a dependency")
        print("  remove <name>     Remove a dependency")
        print("  install           Install all dependencies")
        print("  list              List installed packages")
        return

    command = args[0]
    project_path = os.getcwd()

    mgr = DimPackageManager(project_path)

    if command == "init":
        name = args[1] if len(args) > 1 else os.path.basename(project_path)
        main = args[2] if len(args) > 2 else "main.dim"
        mgr.init(name, main)

    elif command == "add":
        if len(args) < 2:
            print("Usage: dim pkg add <name> [version]")
            return
        name = args[1]
        version = args[2] if len(args) > 2 else "*"
        mgr.add_dep(name, version)

    elif command == "remove":
        if len(args) < 2:
            print("Usage: dim pkg remove <name>")
            return
        name = args[1]
        mgr.remove_dep(name)

    elif command == "install":
        mgr.install_all()

    elif command == "list":
        deps = mgr.resolve_deps()
        if deps:
            print("Dependencies:")
            for name, pkg in deps.items():
                print(f"  {name}@{pkg.version}")
        else:
            print("No dependencies")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    import sys

    run_pkg(sys.argv[1:] if len(sys.argv) > 1 else [])
