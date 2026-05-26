import os
import tempfile
import builtins
import inspect

IS_WINDOWS = os.name == "nt"

temp_dir = os.path.abspath("tmp") if IS_WINDOWS else "/tmp"

_original_mkdir    = os.mkdir
_original_makedirs = os.makedirs
_original_chmod    = os.chmod
_original_os_open  = os.open
_original_open     = builtins.open

_original_makedirs(temp_dir, exist_ok=True)

if IS_WINDOWS:
    os.environ["TMPDIR"] = temp_dir
    os.environ["TEMP"]   = temp_dir
    os.environ["TMP"]    = temp_dir

tempfile.tempdir    = temp_dir
tempfile.gettempdir = lambda: temp_dir

def _redirect(path):
    if IS_WINDOWS and isinstance(path, (str, bytes)):
        p = path.decode() if isinstance(path, bytes) else path
        for prefix in ("/tmp", "\\tmp"):
            if p.startswith(prefix):
                redirected = os.path.normpath(p.replace(prefix, temp_dir, 1))
                return redirected.encode() if isinstance(path, bytes) else redirected
    return path

if IS_WINDOWS:
    def _patched_mkdir(path, mode=0o777):
        path = _redirect(path)
        if not os.path.exists(path):
            parent = os.path.dirname(path)
            if parent and not os.path.exists(parent):
                _patched_mkdir(parent, mode)
            if not os.path.exists(path):
                _original_mkdir(path, mode)

    def _patched_makedirs(path, mode=0o777, exist_ok=False):
        _original_makedirs(_redirect(path), mode=mode, exist_ok=exist_ok)

    def _patched_chmod(path, mode, **kwargs):
        p = path.decode() if isinstance(path, bytes) else str(path)
        if p.startswith("/tmp") or p.startswith("\\tmp"):
            return
        try:
            _original_chmod(path, mode, **kwargs)
        except (NotImplementedError, OSError):
            pass

    def _patched_os_open(path, flags, mode=0o777, **kwargs):
        path = _redirect(path)
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            _original_makedirs(parent, exist_ok=True)
        return _original_os_open(path, flags, mode, **kwargs)

    def _patched_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes)):
            file = _redirect(file)
            mode = args[0] if args else kwargs.get("mode", "r")
            if any(c in mode for c in ("w", "a", "x")):
                parent = os.path.dirname(file)
                if parent and not os.path.exists(parent):
                    _original_makedirs(parent, exist_ok=True)
        return _original_open(file, *args, **kwargs)

    os.mkdir      = _patched_mkdir
    os.makedirs   = _patched_makedirs
    os.chmod      = _patched_chmod
    os.open       = _patched_os_open
    builtins.open = _patched_open


def apply_hopsworks_patches():
    _SSL_KEYS = [
        "ssl.ca.location", "ssl.certificate.location",
        "ssl.key.location", "ssl.keystore.location",
    ]

    try:
        import hopsworks_common.client.external as _ext
        for _, obj in inspect.getmembers(_ext, inspect.isclass):
            if hasattr(obj, "_makedirs_with_sticky_bit"):
                def _patched_makedirs_with_sticky_bit(self):
                    try:
                        raw_dir = self._get_jks_dir_path()
                    except AttributeError:
                        parts = [self._host]
                        if hasattr(self, '_project_name') and self._project_name:
                            parts.append(self._project_name)
                        if hasattr(self, '_username') and self._username:
                            parts.append(self._username)
                        raw_dir = "/tmp/" + "/".join(parts)
                    _original_makedirs(_redirect(raw_dir), exist_ok=True)
                obj._makedirs_with_sticky_bit = _patched_makedirs_with_sticky_bit
                break
    except Exception as e:
        print(f"[warn] _makedirs_with_sticky_bit patch failed: {e}")

    try:
        import hsfs.storage_connector as _sc
        for _, obj in inspect.getmembers(_sc, inspect.isclass):
            if hasattr(obj, "confluent_options"):
                _orig_co = obj.confluent_options
                def _patched_confluent_options(self, _orig=_orig_co):
                    config = _orig(self)
                    for k in _SSL_KEYS:
                        if k in config:
                            config[k] = _redirect(str(config[k]))
                    return config
                obj.confluent_options = _patched_confluent_options
                break
    except Exception as e:
        print(f"[warn] confluent_options patch failed: {e}")

    try:
        import hsfs.core.kafka_engine as _ke
        _orig_gkc = _ke.get_kafka_config
        def _patched_get_kafka_config(*args, **kwargs):
            config = _orig_gkc(*args, **kwargs)
            for k in _SSL_KEYS:
                if k in config:
                    p = str(config[k])
                    redirected = _redirect(p)
                    if not IS_WINDOWS and not os.path.isfile(redirected):
                        import time
                        for _ in range(10):
                            if os.path.isfile(redirected):
                                break
                            time.sleep(0.5)
                    config[k] = redirected
            return config
        _ke.get_kafka_config = _patched_get_kafka_config
    except Exception as e:
        print(f"[warn] get_kafka_config patch failed: {e}")