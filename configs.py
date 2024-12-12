import argparse
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Tuple, Union, cast, List

from omegaconf import OmegaConf

DataClass = Any
DataClassType = Any


import dataclasses
from typing import Any, Dict, Type, TypeVar, Union

T = TypeVar("T")


class DataclassArgParser:
    """Utility class to populate dataclasses from dictionaries."""

    @staticmethod
    def _populate_dataclass_from_dict(cls: Type[T], inputs: Dict[str, Any]) -> T:
        """
        Populates a dataclass instance from a dictionary.
        Handles nested dataclasses by recursively populating them.
        """
        if not dataclasses.is_dataclass(cls):
            raise ValueError(f"{cls} is not a dataclass")

        # Extract field names and types
        field_names = {field.name: field.type for field in dataclasses.fields(cls)}
        
        # Create an instance of the dataclass
        obj = cls()
        
        for key, value in inputs.items():
            if key in field_names:
                field_type = field_names[key]
                if dataclasses.is_dataclass(field_type):
                    # Recursively populate nested dataclasses
                    nested_obj = DataclassArgParser._populate_dataclass_from_dict(
                        field_type, value
                    )
                    setattr(obj, key, nested_obj)
                else:
                    # Assign value directly for non-dataclass fields
                    setattr(obj, key, value)
        return obj

    @staticmethod
    def _populate_dataclass_from_flat_dict(cls: Type[T], inputs: Dict[str, Any]) -> T:
        """
        Populates a dataclass instance from a flat dictionary.
        Nested fields are expected to use dot notation (e.g., 'field.subfield').
        """
        if not dataclasses.is_dataclass(cls):
            raise ValueError(f"{cls} is not a dataclass")

        # Extract field names and types
        field_names = {field.name: field.type for field in dataclasses.fields(cls)}
        
        # Create an instance of the dataclass
        obj = cls()
        
        for key, value in inputs.items():
            parts = key.split(".")
            current_obj = obj
            for i, part in enumerate(parts):
                if i == len(parts) - 1:  # Last part is the actual value
                    if part in field_names:
                        setattr(current_obj, part, value)
                else:
                    # Navigate or create nested objects
                    if hasattr(current_obj, part):
                        next_obj = getattr(current_obj, part)
                    else:
                        next_obj_type = field_names.get(part)
                        next_obj = next_obj_type()
                        setattr(current_obj, part, next_obj)
                    current_obj = next_obj
        return obj


@dataclass
class ConfigBase:
    """Base class that should handle parsing from command line,
    json, dicts.
    """

    @classmethod
    def parse_from_command_line(cls):
        return omegaconf_parse(cls)

    @classmethod
    def parse_from_file(cls, path: str):
        oc = OmegaConf.load(path)
        return cls.parse_from_dict(OmegaConf.to_container(oc))

    @classmethod
    def parse_from_command_line_deprecated(cls):
        result = DataclassArgParser(
            cls, fromfile_prefix_chars="@"
        ).parse_args_into_dataclasses()
        if len(result) > 1:
            raise RuntimeError(
                f"The following arguments were not recognized: {result[1:]}"
            )
        return result[0]

    @classmethod
    def parse_from_dict(cls, inputs):
        return DataclassArgParser._populate_dataclass_from_dict(cls, inputs.copy())

    @classmethod
    def parse_from_flat_dict(cls, inputs):
        return DataclassArgParser._populate_dataclass_from_flat_dict(cls, inputs.copy())

    def save(self, path: str):
        with open(path, "w") as f:
            OmegaConf.save(config=self, f=f)

@dataclass
class VicRegConfig:
    lambda_invariance: float = 25.0
    mu_variance: float = 25.0
    nu_covariance: float = 1.0

# New JEPAConfig class
@dataclass
class JEPAConfig(ConfigBase):
    embed_dim: int = 256
    action_dim: int = 2
    in_c: int = 2
    out_c: int = 64
    epochs: int = 20
    batch_size: int = 1024
    optimizer_type: str = 'adamw'
    scheduler_type: str = 'linear'
    learning_rate: float = 0.001
    model_type: str = 'JEPA'
    data_path: str = '/scratch/DL24FA/train'
    vicreg_loss: VicRegConfig = VicRegConfig()
    action_reg_hidden_dim: str = '32'
    lambda_reg: int = 0.2
    delta_gen: int = 1
    encoder_backbone: str = "resnet18.a1_in1k"
    teacher_forcing: str = 'True'
    # You can add more configuration parameters as needed