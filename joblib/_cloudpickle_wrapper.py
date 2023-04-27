"""
Small shim of loky's cloudpickle_wrapper to avoid failure when
multiprocessing is not available.
"""


from ._multiprocessing_helpers import mp


if mp is not None:
    from .externals.loky import wrap_non_picklable_objects
else:
    def wrap_non_picklable_objects(obj, keep_wrapper=True):
        return obj

__all__ = ["wrap_non_picklable_objects"]
