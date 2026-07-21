# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
import inspect
import logging
import re
import textwrap
from contextlib import contextmanager
from typing import Any, Type, TYPE_CHECKING, ForwardRef

import dspy
from dspy.adapters.base import Adapter
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError
from pydantic import BaseModel
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from dspy.clients.lm import LM

logger = logging.getLogger(__name__)


def _get_pydantic_types(
    annotation, visited: set | None = None
) -> dict[str, Type[BaseModel]]:
    if visited is None:
        visited = set()

    types = {}

    # traverse pydantic fields
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        type_repr = repr(annotation)
        if type_repr in visited:
            return types

        visited.add(type_repr)
        types[type_repr] = annotation

        for field in annotation.model_fields.values():  # type: ignore
            annot = field.annotation
            if isinstance(annot, ForwardRef):
                if annot.__forward_evaluated__:
                    annot = annot.__forward_value__
                else:
                    try:
                        annot = annot._evaluate(
                            globals(), locals(), recursive_guard=set()
                        )
                    except (NameError, RecursionError):
                        print(f"Could not evaluate ForwardRef: {annot}")
                        continue
            types.update(_get_pydantic_types(annot, visited))

    # traverse parent classes
    for base in getattr(annotation, "__bases__", []):
        if issubclass(base, BaseModel) and base is not BaseModel:
            types.update(_get_pydantic_types(base, visited))

    # traverse composite type annotations
    if hasattr(annotation, "__args__"):
        for arg in getattr(annotation, "__args__"):
            types.update(_get_pydantic_types(arg, visited))

    return types


# This is a hacky solution. We probably shouldn't be using the repr to format the field value but this works for now
# We have to temporarily override the __repr__ method of dspy.Image (and probably other dspy types)
# This is probably not thread-safe but who cares about repr's
@contextmanager
def _temporary_repr(cls, fn):
    original = getattr(cls, "__repr__", None)
    setattr(cls, "__repr__", fn)
    try:
        yield
    finally:
        if original is None:
            delattr(cls, "__repr__")
        else:
            setattr(cls, "__repr__", original)


def _format_field_value(field: FieldInfo, value: Any) -> str:
    with _temporary_repr(dspy.Image, lambda im: im.serialize_model()):
        r = repr(value)
    return r


class PydanticAdapter(Adapter):
    async def acall(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Adds simple error correction to the default acall method"""
        processed_signature: Type[Signature] = self._call_preprocess(
            lm, lm_kwargs, signature, inputs
        )  # type: ignore
        messages = self.format(processed_signature, demos, inputs)
        outputs = await lm.acall(messages=messages, **lm_kwargs)
        try:
            parsed_output = self._call_postprocess(
                processed_signature, signature, outputs
            )
        except AdapterParseError as e:
            # TODO: adapt max_tokens based on previous response length
            error = e.__cause__ or e.__context__ or repr(e)
            messages.append({"role": "assistant", "content": outputs[0]})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous output failed to parse. See the error details below:\n\n"
                        f"{error}\n\n"
                        "Please fix the output format and try again."
                    ),
                }
            )
            outputs = await lm.acall(messages=messages, **lm_kwargs)
            parsed_output = self._call_postprocess(
                processed_signature, signature, outputs
            )
        return parsed_output

    def __call__(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return asyncio.run(self.acall(lm, lm_kwargs, signature, demos, inputs))

    def format_field_structure(self, signature: type[Signature]) -> str:
        return ""

    def format_field_description(self, signature: type[Signature]) -> str:
        sections = []

        sections.append(
            "This interaction will be structured using pythonic objects. "
            "You will receive an object of type `Input` and respond with an object of type `Output` as shown below:"
        )

        sections.append("\n```py\nInput(")
        if signature.input_fields:
            for name in signature.input_fields.keys():
                sections.append(f"\t{name}={{{name}}},")
        sections.append(")\n```\n")

        types_to_define = {}
        sections.append("\n```py\nOutput(")
        if signature.output_fields:
            for name, field in signature.output_fields.items():
                sections.append(
                    f"\t{name}={{{name}}}, # type: {repr(field.annotation)}"
                )
                types_to_define.update(_get_pydantic_types(field.annotation))
        sections.append(")\n```\n")

        if types_to_define:
            sections.append("Where the types are defined as follows:\n```py\n")
            for type_name, type_def in types_to_define.items():
                sections.append(
                    f"# Definition of '{type_name}'\n\n{inspect.getsource(type_def)}"
                )
            sections.append("```\n")

        return "\n".join(sections)

    def format_task_description(self, signature: type[Signature]) -> str:
        instructions = textwrap.dedent(signature.instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        return f"In adhering to this structure, your objective is: {objective}"

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        inputs_fmt = [
            f"\t{field_name}={_format_field_value(field, inputs[field_name])}"
            for field_name, field in signature.input_fields.items()
            if field_name in inputs
        ]
        return f"{prefix}\n```py\nInput(\n{',\n'.join(inputs_fmt)}\n)\n```\n{suffix}".strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        outputs_fmt = [
            f"\t{field_name}={_format_field_value(field, outputs[field_name])}"
            for field_name, field in signature.output_fields.items()
            if field_name in outputs
        ]
        return f"Output(\n{',\n'.join(outputs_fmt)}\n)"

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        try:
            last_output_start = completion.rfind("Output(")
            matches = list(
                re.finditer(
                    r"(Output\(.*\))", completion[last_output_start:], re.DOTALL
                )
            )
            if not matches:
                raise ValueError("No Output pattern found in completion")

            # TODO: use lightweight sandbox for execution
            # TODO: check that types are correct. This relies on pydantic validation but does not fail if the model does not use them at all
            result = eval(
                matches[-1].group(1),
                {},
                {
                    "Output": dspy.Prediction,
                    **{
                        t.__name__: t
                        for output_field in signature.output_fields.values()
                        for t in _get_pydantic_types(output_field.annotation).values()
                    },
                },
            )
            return result
        except Exception as e:  # type: ignore
            logger.error(f"Error parsing completion: {e}")
            raise AdapterParseError(
                adapter_name="PydanticAdapter",
                signature=signature,  # type: ignore
                lm_response=completion,
                parsed_result=None,
            )
