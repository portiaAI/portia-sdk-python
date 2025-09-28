"""CLI entrypoint for Portia SDK."""
from __future__ import annotations

import builtins
import sys
from enum import Enum
from functools import wraps
from types import UnionType
from typing import TYPE_CHECKING, Any, get_args, get_origin

import click
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined
from portia.clarification_handler import ClarificationHandler  
from portia.config import Config, GenerativeModelsConfig
from portia.errors import InvalidConfigError
from portia.logger import logger
from portia.portia import ExecutionHooks, Portia
from portia.tool_registry import DefaultToolRegistry
from portia.version import get_version
import toml  
from portia.config_loader import (
    ConfigLoader, 
    ensure_config_directory, 
    get_config_file_path,
    get_config
)
if TYPE_CHECKING:
    from collections.abc import Callable
    from portia.cli_clarification_handler import CLIClarificationHandler
    from pydantic.fields import FieldInfo


DEFAULT_FILE_PATH = ".portia"
PORTIA_API_KEY = "portia_api_key"


class EnvLocation(Enum):
    """The location of the environment variables."""

    ENV_FILE = "ENV_FILE"
    ENV_VARS = "ENV_VARS"


class CLIConfig(BaseModel):
    """Config for the CLI."""

    env_location: EnvLocation = Field(
        default=EnvLocation.ENV_VARS,
        description="The location of the environment variables.",
    )

    end_user_id: str = Field(
        default="",
        description="The end user id to use in the execution context.",
    )

    confirm: bool = Field(
        default=True,
        description="Whether to confirm plans before running them.",
    )

    tool_id: str | None = Field(
        default=None,
        description="The tool ID to use. If not provided, all tools will be used.",
    )


def generate_cli_option_from_pydantic_field(
    f: Callable[..., Any],
    field: str,
    info: FieldInfo,
) -> Callable[..., Any]:
    """Generate a click option from a pydantic field."""
    option_name = field.replace("_", "-")

    
    if option_name.endswith("api-key"):
        return f
    field_type = _annotation_to_click_type(info.annotation)
    if field_type is None:
        return f

    optional_kwargs = {}
    default = info.default if info.default_factory is None else info.default_factory()  # type: ignore reportCallIssue
    if isinstance(default, Enum):
        optional_kwargs["default"] = default.name
    elif default is not None and default is not PydanticUndefined:
        optional_kwargs["default"] = default

    field_help = info.description or f"Set the value for {option_name}"

    return click.option(
        f"--{option_name}",
        type=field_type,
        help=field_help,
        **optional_kwargs,
    )(f)


def _annotation_to_click_type(  # noqa: PLR0911
    annotation: type[Any] | None,
) -> click.ParamType | None:
    """Convert a type annotation to a click type."""
    match annotation:
        case builtins.int:
            return click.INT
        case builtins.float:
            return click.FLOAT
        case builtins.bool:
            return click.BOOL
        case builtins.str:
            return click.STRING
        case builtins.list:
            return click.Tuple([str])
        case _ if isinstance(annotation, type) and issubclass(annotation, Enum):
            return click.Choice(
                [e.value for e in annotation],
                case_sensitive=False,
            )
        case _ if get_origin(annotation) is UnionType:
            args = get_args(annotation)
            for arg in args:
                if (click_type := _annotation_to_click_type(arg)) is not None:
                    return click_type
            return None
        case _:
            return None


def common_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """Define common options for CLI commands."""
    for field, info in Config.model_fields.items():
        generate_cli_option_from_pydantic_field(f, field, info)
    for field, info in GenerativeModelsConfig.model_fields.items():
        generate_cli_option_from_pydantic_field(f, field, info)
    for field, info in CLIConfig.model_fields.items():
        generate_cli_option_from_pydantic_field(f, field, info)

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return f(*args, **kwargs)

    return wrapper


class CLIExecutionHooks(ExecutionHooks):
    """Execution hooks for the CLI."""

    clarification_handler: ClarificationHandler | None = CLIClarificationHandler()


@click.group(context_settings={"max_content_width": 240})
@click.option('--profile', '-p', default=None, help='Configuration profile to use')
@click.pass_context
def cli(ctx, profile) -> None:
    """Portia CLI."""
    ctx.ensure_object(dict)
    ctx.obj['profile'] = profile



@click.command()
def version() -> None:
    """Print the CLI tool version."""
    click.echo(get_version())


@click.command()
@common_options
@click.argument("query")
@click.pass_context
def run(
    ctx,
    query: str,
    **kwargs,  
) -> None:
    """Run a query."""
    profile = ctx.obj.get('profile') if ctx.obj else None
    cli_config, config = _get_config(profile=profile, **kwargs)

   
    registry = DefaultToolRegistry(config)

   
    portia = Portia(
        config=config,
        tools=(
            registry.match_tools(tool_ids=[cli_config.tool_id]) if cli_config.tool_id else registry
        ),
        execution_hooks=CLIExecutionHooks(),
    )

    plan = portia.plan(query, end_user=cli_config.end_user_id)

    if cli_config.confirm:
        click.echo(plan.pretty_print())
        if not click.confirm("Do you want to execute the plan?"):
            return

    plan_run = portia.run_plan(plan, end_user=cli_config.end_user_id)
    click.echo(plan_run.model_dump_json(indent=4))


@click.command()
@common_options
@click.argument("query")
@click.pass_context
def plan(
    ctx,
    query: str,
    **kwargs,  
) -> None:
    """Plan a query."""
    profile = ctx.obj.get('profile') if ctx.obj else None
    cli_config, config = _get_config(profile=profile, **kwargs)

    portia = Portia(config=config)

    output = portia.plan(query, end_user=cli_config.end_user_id)

    click.echo(output.model_dump_json(indent=4))


@click.command()
@common_options
@click.pass_context
def list_tools(
    ctx,
    **kwargs,  
) -> None:
    """List tools."""
    profile = ctx.obj.get('profile') if ctx.obj else None
    cli_config, config = _get_config(profile=profile, **kwargs)


    for tool in DefaultToolRegistry(config).get_tools():
        click.echo(tool.pretty() + "\n")


@click.group()
def config():
    """Manage Portia configuration profiles."""
    pass

@config.command()
@click.argument('profile_name')
@click.option('--template', '-t', 
              type=click.Choice(['openai', 'anthropic', 'gemini', 'azure', 'mixed']),
              help='Use a predefined template')
def create(profile_name: str, template: str | None = None) -> None:
    """Create a new configuration profile."""
    config_file = get_config_file_path()
    ensure_config_directory()
    
    
    if not config_file.exists():
        initial_config = {"profile": {}}
        with open(config_file, "w") as f:
            toml.dump(initial_config, f)
    
    
    with open(config_file, "r") as f:
        data = toml.load(f)
    
    if "profile" not in data:
        data["profile"] = {}
    
    
    if profile_name in data["profile"]:
        if not click.confirm(f"Profile '{profile_name}' already exists. Overwrite?"):
            return
    
    
    if template:
        profile_config = _get_template_config(template)
    else:
        profile_config = {
            "llm_provider": "",
            "default_model": "",
            "storage_class": "MEMORY",
            "default_log_level": "INFO"
        }
    
    data["profile"][profile_name] = profile_config
    
   
    with open(config_file, "w") as f:
        toml.dump(data, f)
    
    click.echo(f"✅ Created profile \"{profile_name}\"")
    if template:
        click.echo(f"   Using template: {template}")

@config.command('set-default')
@click.argument('profile_name')
def set_default(profile_name: str) -> None:
    """Set the default profile."""
    loader = ConfigLoader()
    profiles = loader.list_profiles()
    
    if profile_name not in profiles:
        click.echo(f"❌ Profile '{profile_name}' not found. Available: {', '.join(profiles)}")
        return
    
    
    import os
    os.environ["PORTIA_DEFAULT_PROFILE"] = profile_name
    
    
    config_dir = ensure_config_directory()
    default_file = config_dir / "default_profile"
    with open(default_file, "w") as f:
        f.write(profile_name)
    
    click.echo(f"✅ Set default profile to '{profile_name}'")

@config.command()
@click.argument('profile_name')
@click.argument('assignments', nargs=-1, required=True)
def set(profile_name: str, assignments: tuple[str, ...]) -> None:
    """Set configuration values for a profile.
    
    Example: portia-cli config set gemini execution_model=gemini default_model=google/gemini-2.5-pro
    """
    config_file = get_config_file_path()
    
    if not config_file.exists():
        click.echo("❌ No config file found. Run 'portia-cli config create <profile>' first.")
        return
    
   
    updates = {}
    for assignment in assignments:
        if '=' not in assignment:
            click.echo(f"❌ Invalid assignment: {assignment}. Use format key=value")
            return
        key, value = assignment.split('=', 1)
        
        
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        
        elif value.isdigit():
            value = int(value)
        
        elif value.lower() == 'null':
            value = None
        
        updates[key] = value
    
   
    with open(config_file, "r") as f:
        data = toml.load(f)
    
    if "profile" not in data or profile_name not in data["profile"]:
        click.echo(f"❌ Profile '{profile_name}' not found. Create it first.")
        return
    
    
    data["profile"][profile_name].update(updates)
    
    
    with open(config_file, "w") as f:
        toml.dump(data, f)
    
    click.echo(f"✅ Updated profile '{profile_name}':")
    for key, value in updates.items():
        click.echo(f"   {key} = {value}")

@config.command()
@click.argument('profile_name')
@click.argument('key', required=False)
def get(profile_name: str, key: str | None = None) -> None:
    """Get configuration values from a profile."""
    try:
        loader = ConfigLoader()
        profile_config = loader.load_config_from_toml(profile_name)
        
        if key:
            if key in profile_config:
                click.echo(f"{key} = {profile_config[key]}")
            else:
                click.echo(f"❌ Key '{key}' not found in profile '{profile_name}'")
        else:
            click.echo(f"Profile '{profile_name}' configuration:")
            for k, v in profile_config.items():
                click.echo(f"  {k} = {v}")
                
    except Exception as e:
        click.echo(f"❌ Error loading profile '{profile_name}': {e}")

@config.command('list')
def list_profiles() -> None:
    """List all configuration profiles."""
    loader = ConfigLoader()
    profiles = loader.list_profiles()
    
    if not profiles:
        click.echo("No profiles found. Create one with 'portia-cli config create <name>'")
        return
    
    # Get default profile
    default_profile = loader.get_default_profile()
    
    click.echo("Available profiles:")
    for profile in profiles:
        marker = " (default)" if profile == default_profile else ""
        click.echo(f"  • {profile}{marker}")

@config.command()
@click.argument('profile_name')
@click.option('--force', is_flag=True, help='Delete without confirmation')
def delete(profile_name: str, force: bool = False) -> None:
    """Delete a configuration profile."""
    config_file = get_config_file_path()
    
    if not config_file.exists():
        click.echo("❌ No config file found.")
        return
    
    with open(config_file, "r") as f:
        data = toml.load(f)
    
    if "profile" not in data or profile_name not in data["profile"]:
        click.echo(f"❌ Profile '{profile_name}' not found.")
        return
    
    if not force and not click.confirm(f"Delete profile '{profile_name}'?"):
        return
    
    del data["profile"][profile_name]
    
    with open(config_file, "w") as f:
        toml.dump(data, f)
    
    click.echo(f"✅ Deleted profile '{profile_name}'")

@config.command()
def path() -> None:
    """Show the path to the configuration file."""
    config_file = get_config_file_path()
    click.echo(f"Config file: {config_file}")
    click.echo(f"Exists: {'Yes' if config_file.exists() else 'No'}")

@config.command()
@click.argument('profile_name', required=False)
def validate(profile_name: str | None = None) -> None:
    """Validate configuration profiles."""
    loader = ConfigLoader()
    
    if profile_name:
        profiles = [profile_name]
    else:
        profiles = loader.list_profiles()
    
    if not profiles:
        click.echo("No profiles to validate.")
        return
    
    for profile in profiles:
        try:
            config_dict = get_config(profile)
            # Try to create a Config object to validate
            from portia.config import Config
            Config.from_local_config(profile)
            click.echo(f"✅ Profile '{profile}' is valid")
        except Exception as e:
            click.echo(f"❌ Profile '{profile}' is invalid: {e}")

def _get_template_config(template: str) -> dict:
    """Get predefined template configurations using Config defaults."""
    provider_map = {
        'openai': LLMProvider.OPENAI,
        'anthropic': LLMProvider.ANTHROPIC,
        'gemini': LLMProvider.GOOGLE,
        'azure': LLMProvider.AZURE_OPENAI,
    }
    if template in provider_map:
        provider = provider_map[template]
        # Use Config logic to get default models for this provider
        config = Config(llm_provider=provider)
        return {
            'llm_provider': provider.value,
            'default_model': config.get_agent_default_model('default_model', provider),
            'planning_model': config.get_agent_default_model('planning_model', provider),
            'introspection_model': config.get_agent_default_model('introspection_model', provider),
            'storage_class': 'CLOUD',
            'default_log_level': 'INFO',
            'execution_agent_type': 'ONE_SHOT'
        }
    elif template == 'mixed':
        return {
            'default_model': Config(llm_provider=LLMProvider.OPENAI).get_agent_default_model('default_model', LLMProvider.OPENAI),
            'planning_model': Config(llm_provider=LLMProvider.ANTHROPIC).get_agent_default_model('planning_model', LLMProvider.ANTHROPIC),
            'execution_model': Config(llm_provider=LLMProvider.OPENAI).get_agent_default_model('execution_model', LLMProvider.OPENAI),
            'introspection_model': Config(llm_provider=LLMProvider.GOOGLE).get_agent_default_model('introspection_model', LLMProvider.GOOGLE),
            'storage_class': 'CLOUD',
            'default_log_level': 'INFO'
        }
    else:
        return {
            "llm_provider": "",
            "default_model": "",
            "storage_class": "MEMORY",
            "default_log_level": "INFO"
        }



def _get_config(
    profile: str | None = None,
    **kwargs,  
) -> tuple[CLIConfig, Config]:
    """Init config."""
    cli_config = CLIConfig(**kwargs)
    if cli_config.env_location == EnvLocation.ENV_FILE:
        load_dotenv(override=True)
    
    try:
        if profile:
            config = Config.from_local_config(profile=profile, **kwargs)
        else:
            config = Config.from_default(**kwargs)
    except InvalidConfigError as e:
        logger().error(e.message)
        sys.exit(1)

    return (cli_config, config)

cli.add_command(version)
cli.add_command(run)
cli.add_command(plan)
cli.add_command(list_tools)
cli.add_command(config)

if __name__ == "__main__":
    cli(obj={})
