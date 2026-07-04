"""Remote deployment orchestration for Vultr GPU instances.

This subpackage runs on the *developer machine*. It provisions a Vultr GPU
instance with attached persistent block storage, bootstraps it (clone repo,
mount storage, install the CUDA + TRIBE stack), runs the end-to-end pipeline
(``ysp-pipeline``) on the instance, and optionally halts/destroys it when done —
so no dataset, tensor or checkpoint ever needs to live on the laptop.
"""
