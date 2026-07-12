"""Optional Jupyter compatibility for OpenAI4S scientific workers.

Importing this package never imports Jupyter, IPython, ZeroMQ, or ``ipykernel``.
KernelSpec generation and installation stay stdlib-only; the wire dependency is
loaded only when a Jupyter frontend launches :mod:`.bridge`.
"""

from openai4s.adapters.jupyter.kernelspec import (
    KERNEL_NAMES,
    adapter_status,
    install_kernelspecs,
    kernel_spec,
    write_kernelspecs,
)

__all__ = [
    "KERNEL_NAMES",
    "adapter_status",
    "install_kernelspecs",
    "kernel_spec",
    "write_kernelspecs",
]
