# SPDX-License-Identifier: Apache-2.0
"""M5 mutation check: installs the M5_MUTATION hook in every Python process.

Put this directory FIRST on PYTHONPATH (sm70/scripts/m5_mutation_check.sh
does). Python imports the first ``sitecustomize`` on sys.path at startup, so
this file shadows any other one (the Nix python wrapper ships one that
handles NIX_PYTHONPATH); it therefore runs the shadowed one first, then
installs the hook. Without M5_MUTATION it only chains. See m5_mutation.py.
"""


def _chain_shadowed_sitecustomize():
    import importlib.machinery
    import importlib.util
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m5_shadowed_sitecustomize"] = mod
    spec.loader.exec_module(mod)


def _install():
    import os
    import sys

    mutation = os.environ.get("M5_MUTATION")
    if not mutation:
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import m5_mutation
    finally:
        sys.path.pop(0)
    m5_mutation.install_import_hook(mutation)


_chain_shadowed_sitecustomize()
_install()
