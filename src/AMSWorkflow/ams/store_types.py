from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from typing import Optional


class UQType(Enum):
    Random = "random"
    DeltaUQ = "deltaUQ"
    Faiss = "faiss"


class UQAggregate(Enum):
    max = "max"
    mean = "mean"


@dataclass
class AMSModelDescr:
    path: str
    threshold: float
    uq_type: UQType
    uq_aggregate: Optional[UQAggregate] = None

    def __post_init__(self):

        if not isinstance(self.uq_type, UQType):
            raise TypeError(f"Field uq_type must be of type {type(UQType)} but is of type {type(self.uq_type)}")

        if not Path(self.path).exists():
            raise RuntimeError("Trying to generate model descr from path {self.path} but file does not exist")
        print(self.uq_type, type(self.uq_type))
        if self.uq_type != UQType.Random and self.uq_aggregate is None:
            raise ValueError(f"Cannot create model of {self.uq_type} without specifying aggregation type")

        if self.uq_type == UQType.Faiss:
            raise NotImplementedError("ams-python does not support 'faiss' please request this feature")

    def to_dict(self):
        tmp = {"path": str(self.path), "threshold": self.threshold, "uq_type": self.uq_type.value}
        if self.uq_type != UQType.Random:
            tmp["uq_aggregate"] = self.uq_aggregate.value
        return tmp
