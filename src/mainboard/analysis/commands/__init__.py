"""Commands of the analysis CLI that read a project's import graph, definitions and papers.

Importing the package registers the relations they share (`modules`, `experiments`).
"""

from . import gate as gate
from .graphs import modules as modules
from .papers import experiments as experiments
