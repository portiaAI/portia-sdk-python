"""Render templates.

This module provides a utility function to render Jinja templates. It loads a template from the file
system and renders it to a string, allowing for dynamic generation of content with provided
keyword arguments.
"""

import importlib.resources
from typing import Any

from jinja2 import Environment, FileSystemLoader


def render_template(file_name: str, **kwargs: Any) -> str:
    """Render a Jinja template from the file system into a string.

    This function loads a template file from the `portia.templates` package,
    and using Jinja2 renders the template with the provided keyword arguments.

    Args:
        file_name (str): The name of the template file to be rendered.
        **kwargs (Any): Keyword arguments that will be passed to the template for rendering.

    Returns:
        str: The rendered template as a string.

    Example:
        rendered = render_template("example_template.html", user_name="Alice")

    """
    from portia import templates

    source = importlib.resources.files(templates).joinpath(file_name)
    with importlib.resources.as_file(source) as template_path:
        env = Environment(loader=FileSystemLoader(template_path.parent), autoescape=True)
        template = env.get_template(file_name)
        return template.render(**kwargs)
