"""`datasets` fingerprinting compat shim for Python 3.14.

`datasets` fingerprints a dataset config by dill-pickling it. Its custom
``datasets.utils._dill.Pickler`` overrides ``_batch_setitems(self, items)`` to sort
dict items, so the fingerprint is independent of key insertion order. Python 3.14
changed ``pickle._Pickler.save_dict`` to call ``_batch_setitems(items, obj)`` — the
extra ``obj`` crashes that old two-arg override:

    TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given

The broken override lives in `datasets` 3.6.0 (the last ``<4`` release, pinned for
MSWC's loading script), so no in-range version bump fixes it. This re-wraps the
override to accept and forward ``obj`` while keeping its sort-for-stable-hash
behavior. Idempotent and self-disabling: a no-op once `datasets` ships its own fix
or on any Python whose pickler uses the old signature.
"""

import inspect

_patched = False


def patch_datasets_py314() -> None:
    """Make ``datasets``' ``_batch_setitems`` override tolerate Python 3.14's extra arg."""
    global _patched
    if _patched:
        return

    import datasets.utils._dill as _dd
    import dill

    override_params = inspect.signature(_dd.Pickler._batch_setitems).parameters
    parent_params = inspect.signature(dill.Pickler._batch_setitems).parameters
    if "obj" in override_params or "obj" not in parent_params:
        # datasets already 3.14-compatible, or this Python predates the change.
        _patched = True
        return

    from datasets.fingerprint import Hasher

    def _batch_setitems(self, items, obj=None):
        if self._legacy_no_dict_keys_sorting:
            return super(_dd.Pickler, self)._batch_setitems(items, obj)
        # Ignore the order of keys in a dict (mirrors the upstream override).
        try:
            items = sorted(items)
        except Exception:  # unorderable elements
            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        dill.Pickler._batch_setitems(self, items, obj)

    _dd.Pickler._batch_setitems = _batch_setitems
    _patched = True
