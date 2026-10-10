# Only the gate launcher puts this directory on PYTHONPATH. Fail closed during interpreter start.
import os
import sys

try:
    from guard import Guard

    if os.environ.get("MB_GATE_ROOT"):
        Guard.from_environment().install()

    if os.environ.get("MB_GATE_CATALOG"):
        from pathlib import Path

        from catalog_double import install

        install(Path(os.environ["MB_GATE_CATALOG"]))

    if os.environ.get("MB_GATE_BOOTSTRAP") == "1":
        import errno
        import fcntl
        from pathlib import Path

        from filelock._unix import UnixFileLock

        from mainboard.engines.compile.backend import Pixi, Process

        _projects = Path(os.environ["MB_GATE_PROJECTS"])
        _lock = _projects / ".mainboard/.sync.lock"
        _acquire = UnixFileLock._acquire
        _run = Pixi.run

        def _readonly_acquire(self):
            if Path(self.lock_file).resolve() != _lock:
                return _acquire(self)
            descriptor = os.open(_lock, os.O_RDONLY)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                os.close(descriptor)
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                return
            metadata = os.fstat(descriptor)
            self._mark_descriptor_owned(descriptor, (metadata.st_dev, metadata.st_ino))

        def _no_install(self, command, env="default"):
            if self.manifest.is_relative_to(_projects / ".mainboard"):
                return self.within_cwd(
                    Process.passthrough, "run", "--frozen", "--no-install", "-e", env, *command
                )
            return _run(self, command, env)

        UnixFileLock._acquire = _readonly_acquire
        Pixi.run = _no_install
except Exception as error:
    # Python normally prints and ignores sitecustomize errors. That would run an unguarded CLI.
    sys.stderr.write(f"M00 bootstrap refused: {error}\n")
    sys.stderr.flush()
    os._exit(70)
