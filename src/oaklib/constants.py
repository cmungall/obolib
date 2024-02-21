"""Constants for use across OAK."""

import pystow

__all__ = [
    "OAKLIB_MODULE",
]

OAKLIB_MODULE = pystow.module("oaklib")

NODE_RENAME = "NodeRename"
CLASS_CREATION = "ClassCreation"
NODE_CREATION = "NodeCreation"
NODE_DELETION = "NodeDeletion"

NODE_TEXT_DEFINITION_CHANGE = "NodeTextDefinitionChange"
NODE_UNOBSOLETION = "NodeUnobsoletion"
NODE_DIRECT_MERGE = "NodeDirectMerge"
NODE_OBSOLETION_WITH_DIRECT_REPLACEMENT = "NodeObsoletionWithDirectReplacement"
NODE_OBSOLETION = "NodeObsoletion"
