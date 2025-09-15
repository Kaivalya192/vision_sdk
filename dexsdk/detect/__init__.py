"""Detection package (relocated).

Provides:
- primitive: SIFTMatcher and primitives-based detection utilities
- multi_template: MultiTemplateMatcher for managing multiple templates
"""

from .primitive import *  # noqa: F401,F403
from .multi_template import *  # noqa: F401,F403

__all__ = []  # populated by star imports above

