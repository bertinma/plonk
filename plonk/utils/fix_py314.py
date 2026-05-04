"""
Script used to centralise all fix, patches and hook to be able to use lastest packages version (Python 3.14, Torch 3.10)
with older and/or not maintained packages.
Current packages with the need of patch :
    - Hydra
    - Torch_audiomentations
    - Multiprocessing (Python 3.14 changed start method from 'fork' to 'forkserver')
"""

"""
Fix Hydra compatibility with Python 3.14
Problem : Hydra uses an object, 'LazyCompletionHelp', not compatible with last version of argparse, included in Python 3.14
          The problem is due to format incompatibility (ValueError: badly formed help string)
Fix : Disable compatibility check for the object 'LazyCompletionHelp' in argparse.
      The only impact will be the possible lack of help string for CLI call. It does not impact pipeline running.
"""
import argparse

_original_check_help = argparse.ArgumentParser._check_help  # type: ignore


# Create the patched 'check_help' method
def _patched_check_help(self, action):
    # Detect the object LazyCompletionHelp. Do not apply any verification when this object is processed by argparse
    if action.help is not None and type(action.help).__name__ == "LazyCompletionHelp":
        return
    else:
        # When other object is detected, apply original check method
        _original_check_help(self, action)


# Replace 'check_help' method in argparse
argparse.ArgumentParser._check_help = _patched_check_help  # type: ignore


"""
Fix multiprocessing compatibility with Python 3.14
Problem : Python 3.14 changed the default multiprocessing start method from 'fork' to 'forkserver' on Linux.
          'forkserver' requires all DataLoader worker arguments to be picklable, which breaks with lambda functions,
          nested functions, and local classes.
Fix : Set the start method back to 'fork' for backward compatibility.
      Note: 'fork' can have issues with multithreading, but is faster and more compatible.
      Alternative: Make all code picklable (no lambdas, no nested functions).
"""
import multiprocessing
import sys

if sys.platform != "darwin":  # Only needed on Linux, macOS already uses 'fork'
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        # Start method already set, ignore
        pass
