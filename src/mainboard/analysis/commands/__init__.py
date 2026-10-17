"""Commands of the analysis CLI that read a project's import graph, definitions and papers.

Importing the package registers the relations they share (`modules`, `experiments`).
"""

from . import experiments as experiments
from . import gate as gate
from . import modules as modules
