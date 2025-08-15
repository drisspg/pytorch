# mypy: allow-untyped-defs
"""
CuteDSL-specific operation overrides for pointwise operations.

This module provides CuteDSL implementations of common operations used in
template kernels, particularly for flex attention modifications.
"""

import math
from typing import Union

import torch
from torch._inductor.codegen.common import OpOverrides


import cutlass.cute as cute

class CuteDSLOpOverrides(OpOverrides):
    """
    CuteDSL-specific operation overrides that generate code using CuteDSL syntax.

    CuteDSL TensorSSA objects have built-in operator overloads (__add__, __mul__, etc.)
    and math functions (cute.math.exp, cute.math.sqrt, etc.), so our overrides
    generate Python expressions that leverage these native capabilities.
    """

    @staticmethod
    def constant(value: Union[bool, float, int], dtype: torch.dtype) -> str:
        """Generate CuteDSL constant representation."""
        if value == float("-inf"):
            return "float('-inf')"
        elif value == float("inf"):
            return "float('inf')"
        elif math.isnan(value):
            return "float('nan')"
        return repr(value)

    @staticmethod
    def add(a: str, b: str) -> str:
        """Addition using CuteDSL TensorSSA __add__ operator."""
        return f"({a} + {b})"

    @staticmethod
    def mul(a: str, b: str) -> str:
        """Multiplication using CuteDSL TensorSSA __mul__ operator."""
        return f"({a} * {b})"

    @staticmethod
    def sub(a: str, b: str) -> str:
        """Subtraction using CuteDSL TensorSSA __sub__ operator."""
        return f"({a} - {b})"

    @staticmethod
    def truediv(a: str, b: str) -> str:
        """Division using CuteDSL TensorSSA __truediv__ operator."""
        return f"({a} / {b})"

    @staticmethod
    def exp(x: str) -> str:
        """Exponential using CuteDSL cute.math.exp function."""
        return f"cute.math.exp({x})"

    @staticmethod
    def sqrt(x: str) -> str:
        """Square root using CuteDSL cute.math.sqrt function."""
        return f"cute.math.sqrt({x})"

    @staticmethod
    def log(x: str) -> str:
        """Natural logarithm using CuteDSL cute.math.log function."""
        return f"cute.math.log({x})"

    @staticmethod
    def cos(x: str) -> str:
        """Cosine using CuteDSL cute.math.cos function."""
        return f"cute.math.cos({x})"

    @staticmethod
    def sin(x: str) -> str:
        """Sine using CuteDSL cute.math.sin function."""
        return f"cute.math.sin({x})"

    @staticmethod
    def erf(x: str) -> str:
        """Error function using CuteDSL cute.math.erf function."""
        return f"cute.math.erf({x})"

    @staticmethod
    def maximum(a: str, b: str) -> str:
        """Maximum using CuteDSL cute.math.maximum function."""
        return f"cute.math.maximum({a}, {b})"

    @staticmethod
    def minimum(a: str, b: str) -> str:
        """Minimum using CuteDSL cute.math.minimum function."""
        return f"cute.math.minimum({a}, {b})"

    @staticmethod
    def where(condition, a, b) -> str:
        """Conditional selection - handles both CSEVariable and string inputs."""
        return f"cute.where({condition}, {a}, {b})"

    @staticmethod
    def pow(a: str, b: str) -> str:
        """Power operation using CuteDSL TensorSSA __pow__ operator."""
        return f"({a} ** {b})"

    @staticmethod
    def abs(x: str) -> str:
        """Absolute value using CuteDSL cute.math.abs function."""
        return f"cute.math.abs({x})"

    @staticmethod
    def neg(x: str) -> str:
        """Negation using CuteDSL TensorSSA __neg__ operator."""
        return f"(-{x})"

    @staticmethod
    def to_dtype(x: str, dtype: torch.dtype, src_dtype=None, use_compute_types=True) -> str:
        """Type conversion using CuteDSL TensorSSA.to(Type[Numeric]).

        Maps torch dtypes to cutlass.cute.typing numeric types and emits
        `{x}.to(cute.typing.<Type>)`.

        Raises NotImplementedError for unsigned integer and unsupported dtypes.
        """
        # Map torch dtypes to CuteDSL type strings
        torch2cute_dtype_map = {
            torch.float16: "cutlass.Float16",
            torch.bfloat16: "cutlass.BFloat16",
            torch.float32: "cutlass.Float32",
            torch.float64: "cutlass.Float64",
            torch.int8: "cutlass.Int8",
            torch.int16: "cutlass.Int16",
            torch.int32: "cutlass.Int32",
            torch.int64: "cutlass.Int64",
            torch.bool: "cutlass.Boolean",
            torch.float8_e4m3fn: "cutlass.Float8E4M3FN",
            torch.float8_e5m2: "cutlass.Float8E5M2",

        }

        cute_type = torch2cute_dtype_map.get(dtype)
        if cute_type is None:
            raise NotImplementedError(f"CuteDSL dtype cast not implemented for torch dtype: {dtype}")

        return f"{x}.to({cute_type})"

    @staticmethod
    def sigmoid(x: str) -> str:
        """Sigmoid activation function."""
        # Could use cute.math.sigmoid if available, or implement as 1/(1+exp(-x))
        return f"cute.math.sigmoid({x})"

    @staticmethod
    def relu(x: str) -> str:
        """ReLU activation function."""
        return f"cute.math.maximum({x}, 0.0)"

    def tanh(self, x0: str) -> str:
        """Hyperbolic tangent using CuteDSL cute.math.tanh function."""
        return f"cute.math.tanh({x0})"

    # Logical operations
    def logical_and(self, x0: str, x1: str) -> str:
        """Logical AND."""
        return f"({x0} & {x1})"

    def logical_or(self, x0: str, x1: str) -> str:
        """Logical OR."""
        return f"({x0} | {x1})"

    @staticmethod
    def logical_not(a: str) -> str:
        """Logical NOT."""
        return f"({a} == 0)"

    # Comparison operations
    @staticmethod
    def eq(a: str, b: str) -> str:
        """Equality comparison."""
        return f"({a} == {b})"

    @staticmethod
    def ne(a: str, b: str) -> str:
        """Not equal comparison."""
        return f"({a} != {b})"

    @staticmethod
    def lt(a: str, b: str) -> str:
        """Less than comparison."""
        return f"({a} < {b})"

    @staticmethod
    def le(a: str, b: str) -> str:
        """Less than or equal comparison."""
        return f"({a} <= {b})"

    @staticmethod
    def gt(a: str, b: str) -> str:
        """Greater than comparison."""
        return f"({a} > {b})"

    @staticmethod
    def ge(a: str, b: str) -> str:
        """Greater than or equal comparison."""
        return f"({a} >= {b})"
