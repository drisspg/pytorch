# mypy: allow-untyped-defs
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
from torch._inductor import config
from torch._inductor.codegen.common import IndentedBuffer, KernelTemplate
from torch._inductor.ir import Buffer, CUDATemplateBuffer, IRNode, Layout, TensorBox
from torch._inductor.virtualized import V

if TYPE_CHECKING:
    from ...autotune_process import CUDABenchmarkRequest
    from .cutedsl_template import CuteDSLTemplate

log = logging.getLogger(__name__)


class CuteDSLKernelBase(ABC):
    """
    Base class for all CuteDSL kernel implementations.
    Subclasses should implement the generate_kernel_code method.
    """

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
    ) -> None:
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.args = ArgumentManager()

    @abstractmethod
    def generate_kernel_code(self, code: IndentedBuffer, **kwargs) -> None:
        """Generate the CuteDSL kernel implementation."""
        pass

    def header(self) -> IndentedBuffer:
        """Generate header imports for CuteDSL kernel."""
        res = IndentedBuffer()
        res.splice("""
            import torch
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack
            from typing import Optional
        """)
        return res

    def render(self, **kwargs) -> str:
        """Render the complete kernel code."""
        code = self.header()
        code.writeline("")
        self.generate_kernel_code(code, **kwargs)
        return code.getvalue()


class CuteDSLTemplateKernel:
    """
    Kernel implementation for CuteDSL templates.
    Handles code generation and argument management for Python-based CUDA kernels.
    """

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
        subgraph_fn: Optional[Any] = None,
        mask_fn: Optional[Any] = None,
    ) -> None:
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.subgraph_fn = subgraph_fn
        self.mask_fn = mask_fn
        self.args = ArgumentManager()

    def add_input_arg(self, name: str, tensor: Buffer) -> None:
        """Add an input tensor argument."""
        self.args.add_arg(name, tensor)

    def add_output_arg(self, name: str, tensor: Buffer) -> None:
        """Add an output tensor argument."""
        self.args.add_arg(name, tensor)

    def render(self, template, kwargs) -> str:
        """Render the kernel using the template."""
        return template.render(
            kernel_name=self.kernel_name,
            input_nodes=self.input_nodes,
            output_node=self.output_node,
            subgraph_fn=self.subgraph_fn,
            mask_fn=self.mask_fn,
            **kwargs
        )


class ArgumentManager:
    """Manages kernel arguments for CuteDSL kernels."""

    def __init__(self):
        self.args = []
        self.arg_names = []

    def add_arg(self, name: str, tensor: Buffer) -> None:
        """Add a tensor argument."""
        self.args.append(tensor)
        self.arg_names.append(name)

    def call_args(self) -> list[str]:
        """Get the list of argument names for kernel calls."""
        return self.arg_names


class CuteDSLTemplateCaller:
    """
    Caller implementation for CuteDSL templates.
    Handles kernel invocation and benchmarking.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_kernel_render: Callable,
        bmreq: "CUDABenchmarkRequest",
        template: "CuteDSLTemplate",
    ) -> None:
        self.name = name
        self.category = category
        self.input_nodes = input_nodes
        self.layout = layout
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template

    def __str__(self) -> str:
        return f"CuteDSLTemplateCaller({self.name}, {self.category})"

    def benchmark(self, *args, out) -> float:
        """Benchmark the kernel execution."""
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, out=out)

    def __call__(self, input_nodes: list[TensorBox]) -> TensorBox:
        """Call the kernel with given inputs."""
        # Create output buffer
        output = TensorBox.create(
            CUDATemplateBuffer(
                layout=self.layout,
                inputs=input_nodes,
                make_kernel_render=self.make_kernel_render,
                template=self.template,
            )
        )
        return output

    def info_dict(self) -> dict[str, Any]:
        """Return information about this kernel for logging/debugging."""
        return {
            "name": self.name,
            "category": self.category,
            "kernel_name": self.template.name,
            "has_subgraph": self.template.subgraph_fn is not None,
            "has_mask": self.template.mask_fn is not None,
        }