from uuid import uuid4
from enum import Enum
import time
import random
import torch
import os
import json

from cutegen.util import debug_print

class ErrorType(Enum):
    NONE = 0
    PASS = 1
    COMPILE = 2
    CORRECT = 3
    HACKED = 4

class Correction:
    def __init__(self, src, error_type, msg, fix=None):
        self.src = src
        self.error_type = error_type
        self.msg = msg
        self.fix = fix

    def to_dict(self):
        return {
            "src": self.src,
            "error_type": str(self.error_type.name),
            "msg": self.msg,
            "fix": self.fix, # TODO: figure out format and how to serialize this
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            src=str(data.get("src")),
            error_type=ErrorType[data.get("error_type", "NONE")],
            msg=str(data.get("msg")),
            fix=data.get("fix")
        )

    def __str__(self):
        return str(self.to_dict())

class Node:
    def __init__(self, ref, src="", prev_src="", ref_time=None, save_folder_path="", depth=0):
        #self.uuid = str(uuid4()) + "-" + str(int(time.time()))
        self.ref = ref
        self.ref_time = ref_time

        self.prev_src = prev_src
        self.src = src # current source code
        self.history = [] # history of corrections
        
        # for evaluating the code
        self.error_type = ErrorType.NONE
        self.time = {}
        self.perf = None
        self.metadata = {}

        self.save_folder_path = save_folder_path

        self.depth = depth
        self.uuid = str(depth)+"_"+time.strftime("%Y%m%d-%H%M%S", time.localtime())

        debug_print(f"Node {self.uuid} initialized")

    @classmethod
    def from_dict(cls, data):
        node = cls(
            ref=str(data["ref"]),
            src=str(data.get("src")),
            ref_time=dict(data.get("ref_time")),
            prev_src=str(data.get("prev_src", "")),
            save_folder_path=str(data.get("save_folder_path", ""))
        )
        # Restore additional properties
        node.uuid = str(data.get("uuid", node.uuid))
        node.error_type = ErrorType[data.get("error_type", "NONE")]
        node.time = dict(data.get("time", {}))
        node.perf = data.get("perf")
        node.depth = int(data.get("depth", 0))
        node.metadata = dict(data.get("metadata", {}))
        node.history = [Correction.from_dict(correction) for correction in data.get("history", [])]
        return node

    def to_dict(self):
        return {
            "uuid": str(self.uuid),
            "ref": self.ref,
            "ref_time": self.ref_time,
            "depth": self.depth,
            "src": self.src,
            "prev_src": self.prev_src,
            "history": [correction.to_dict() for correction in self.history],
            
            "error_type": str(self.error_type.name),
            "time": self.time,
            "perf": self.perf,
            "metadata": self.metadata,

            "save_folder_path": self.save_folder_path,
        }

    def __str__(self):
        return str(self.to_dict())

    def update_history(self, src, fix):
        msg = ""
        if self.error_type == ErrorType.COMPILE:
            msg = self.metadata["compile"]
        elif self.error_type == ErrorType.CORRECT:
            msg = self.metadata["correct"]
        self.history.append(Correction(self.src, self.error_type, msg, fix))
        self.src = src

    def save_as_json_local(self, file_dir):
        debug_print(f"Saving node {self.uuid} to {file_dir}")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        with open(file_dir + f"/{str(self.uuid)}.json", "w") as f:
            json.dump(self.to_dict(), f)