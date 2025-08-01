# mypy: allow-untyped-defs
import functools
import hashlib
import itertools
from typing import Any, Optional, TYPE_CHECKING, Union
from unittest.mock import patch

import sympy

import torch
from torch._inductor import config
from torch._inductor.utils import Placeholder
from torch._logging import getArtifactLogger

from ...autotune_process import CUDABenchmarkRequest, TensorMeta
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate

if TYPE_CHECKING:
    from ...scheduler import BaseSchedulerNode
else:
    BaseSchedulerNode = Any

log = getArtifactLogger(__name__, "cutedsl_template")


class CuteDSLTemplate(KernelTemplate):
    """
    Template for generating CuteDSL (CUTLASS Python DSL) kernels.
    Similar to TritonTemplate but generates Python CuteDSL code instead of Triton code.
    """
    
    def __init__(
        self,
        name: str,
        source: str,
        grid: Any = None,
    ) -> None:
        super().__init__(name)
        self.source = source
        self.grid = grid
        self.template = self._template_from_string(source)

    @functools.lru_cache(None)
    def _template_from_string(self, source: str) -> Any:
        return KernelTemplate._template_from_string(source)

    def generate(self, input_nodes, layout, **kwargs):
        """Generate the CuteDSL kernel caller."""
        kernel_name = f"cutedsl_{self.name}_{next(itertools.count())}"
        
        # Render the template
        if self.template is None:
            raise RuntimeError("Template compilation failed (Jinja2 required)")
            
        # Create a fake output node for rendering
        output_node = Buffer(name="buf_out", layout=layout)
        
        # Render the kernel code with Jinja2
        code = self.template.render(
            kernel_name=kernel_name,
            input_nodes=input_nodes,
            output_node=output_node,
            **kwargs
        )
        
        log.debug("Generated CuteDSL Code:\n%s", code)

        # Create benchmark request
        bmreq = CUDABenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(output_node),
            extra_args=tuple(),
            source_code=code,
        )

        # Return a simple caller
        return CuteDSLCaller(
            kernel_name,
            input_nodes,
            layout,
            code,
            bmreq,
        )


class CuteDSLCaller:
    """Simple caller for CuteDSL templates."""
    
    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        code: str,
        bmreq: CUDABenchmarkRequest,
    ):
        self.name = name
        self.input_nodes = input_nodes
        self.layout = layout
        self.code = code
        self.bmreq = bmreq

    def __str__(self) -> str:
        return f"CuteDSLCaller({self.name})"

    def benchmark(self, *args, out) -> float:
        """Benchmark the kernel execution."""
        return self.bmreq.benchmark(*args, out=out)

    def info_dict(self) -> dict[str, Any]:
        """Return information about this kernel."""
        return {
            "name": self.name,
            "backend": "CuteDSL",
        }