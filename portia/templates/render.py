"""Render templates."""

import importlib.resources
from typing import Any

from jinja2 import Environment, FileSystemLoader


def render_template(file_name: str, **kwargs: Any) -> str:  # noqa: ANN401
    """Render a jinja template from the file system in to a string."""
    from portia import templates

    with importlib.resources.path(templates, file_name) as template_path:
        env = Environment(loader=FileSystemLoader(template_path.parent), autoescape=True)
        template = env.get_template(file_name)
        return template.render(**kwargs)
