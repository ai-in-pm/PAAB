#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Command Line Interface
Main CLI entry point for the agent builder system
"""

import click
import json
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.specialized import ResearcherAgent, AnalystAgent, WriterAgent, CoderAgent
from tasks.base import BaseTask, TaskType, TaskPriority
from crews.base import BaseCrew, ProcessType
from runtime.executor import RuntimeExecutor, ExecutionMode
from tools.base import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx, verbose, config):
    """PsychoPy AI Agent Builder - Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        click.echo("Verbose mode enabled")
    
    if config:
        ctx.obj['config_data'] = load_config(config)
        click.echo(f"Loaded configuration from {config}")


@cli.group()
def agent():
    """Agent management commands"""
    pass


@agent.command()
@click.option('--name', '-n', required=True, help='Agent name')
@click.option('--role', '-r', required=True, 
              type=click.Choice(['researcher', 'analyst', 'writer', 'coder']),
              help='Agent role')
@click.option('--goal', '-g', required=True, help='Agent goal')
@click.option('--backstory', '-b', required=True, help='Agent backstory')
@click.option('--output', '-o', type=click.Path(), help='Output file for agent configuration')
def create(name, role, goal, backstory, output):
    """Create a new agent"""
    try:
        # Create agent based on role
        agent_classes = {
            'researcher': ResearcherAgent,
            'analyst': AnalystAgent,
            'writer': WriterAgent,
            'coder': CoderAgent
        }
        
        agent_class = agent_classes[role]
        agent = agent_class(goal=goal, backstory=backstory)
        
        # Create agent configuration
        agent_config = {
            'name': name,
            'role': role,
            'goal': goal,
            'backstory': backstory,
            'id': agent.id,
            'created_at': agent.metrics.last_updated
        }
        
        if output:
            # Save to file
            with open(output, 'w') as f:
                yaml.dump(agent_config, f, default_flow_style=False)
            click.echo(f"Agent configuration saved to {output}")
        else:
            # Print to stdout
            click.echo(yaml.dump(agent_config, default_flow_style=False))
        
        click.echo(f"✅ Agent '{name}' created successfully with ID: {agent.id}")
        
    except Exception as e:
        click.echo(f"❌ Error creating agent: {str(e)}", err=True)
        sys.exit(1)


@agent.command()
@click.argument('agent_file', type=click.Path(exists=True))
def validate(agent_file):
    """Validate an agent configuration file"""
    try:
        with open(agent_file, 'r') as f:
            config = yaml.safe_load(f)
        
        required_fields = ['name', 'role', 'goal', 'backstory']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            click.echo(f"❌ Missing required fields: {', '.join(missing_fields)}", err=True)
            sys.exit(1)
        
        click.echo(f"✅ Agent configuration '{agent_file}' is valid")
        
    except Exception as e:
        click.echo(f"❌ Error validating agent: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def task():
    """Task management commands"""
    pass


@task.command()
@click.option('--description', '-d', required=True, help='Task description')
@click.option('--expected-output', '-e', required=True, help='Expected output description')
@click.option('--type', '-t', type=click.Choice(['research', 'analysis', 'writing', 'coding', 'custom']),
              default='custom', help='Task type')
@click.option('--priority', '-p', type=click.Choice(['low', 'medium', 'high', 'critical']),
              default='medium', help='Task priority')
@click.option('--output', '-o', type=click.Path(), help='Output file for task configuration')
def create(description, expected_output, type, priority, output):
    """Create a new task"""
    try:
        # Map string values to enums
        task_type_map = {
            'research': TaskType.RESEARCH,
            'analysis': TaskType.ANALYSIS,
            'writing': TaskType.WRITING,
            'coding': TaskType.CODING,
            'custom': TaskType.CUSTOM
        }
        
        priority_map = {
            'low': TaskPriority.LOW,
            'medium': TaskPriority.MEDIUM,
            'high': TaskPriority.HIGH,
            'critical': TaskPriority.CRITICAL
        }
        
        task = BaseTask(
            description=description,
            expected_output=expected_output,
            task_type=task_type_map[type],
            priority=priority_map[priority]
        )
        
        task_config = {
            'id': task.id,
            'description': description,
            'expected_output': expected_output,
            'type': type,
            'priority': priority,
            'created_at': task.created_at
        }
        
        if output:
            with open(output, 'w') as f:
                yaml.dump(task_config, f, default_flow_style=False)
            click.echo(f"Task configuration saved to {output}")
        else:
            click.echo(yaml.dump(task_config, default_flow_style=False))
        
        click.echo(f"✅ Task created successfully with ID: {task.id}")
        
    except Exception as e:
        click.echo(f"❌ Error creating task: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def crew():
    """Crew management commands"""
    pass


@crew.command()
@click.option('--name', '-n', required=True, help='Crew name')
@click.option('--agents', '-a', multiple=True, help='Agent configuration files')
@click.option('--tasks', '-t', multiple=True, help='Task configuration files')
@click.option('--process', '-p', type=click.Choice(['sequential', 'parallel', 'hierarchical']),
              default='sequential', help='Process type')
@click.option('--output', '-o', type=click.Path(), help='Output file for crew configuration')
def create(name, agents, tasks, process, output):
    """Create a new crew"""
    try:
        # Load agents
        agent_objects = []
        for agent_file in agents:
            with open(agent_file, 'r') as f:
                agent_config = yaml.safe_load(f)
            
            # Create agent based on role
            agent_classes = {
                'researcher': ResearcherAgent,
                'analyst': AnalystAgent,
                'writer': WriterAgent,
                'coder': CoderAgent
            }
            
            agent_class = agent_classes[agent_config['role']]
            agent = agent_class(
                goal=agent_config['goal'],
                backstory=agent_config['backstory']
            )
            agent_objects.append(agent)
        
        # Load tasks
        task_objects = []
        for task_file in tasks:
            with open(task_file, 'r') as f:
                task_config = yaml.safe_load(f)
            
            task_type_map = {
                'research': TaskType.RESEARCH,
                'analysis': TaskType.ANALYSIS,
                'writing': TaskType.WRITING,
                'coding': TaskType.CODING,
                'custom': TaskType.CUSTOM
            }
            
            priority_map = {
                'low': TaskPriority.LOW,
                'medium': TaskPriority.MEDIUM,
                'high': TaskPriority.HIGH,
                'critical': TaskPriority.CRITICAL
            }
            
            task = BaseTask(
                description=task_config['description'],
                expected_output=task_config['expected_output'],
                task_type=task_type_map[task_config['type']],
                priority=priority_map[task_config['priority']]
            )
            task_objects.append(task)
        
        # Create crew
        process_map = {
            'sequential': ProcessType.SEQUENTIAL,
            'parallel': ProcessType.PARALLEL,
            'hierarchical': ProcessType.HIERARCHICAL
        }
        
        crew_obj = BaseCrew(
            name=name,
            agents=agent_objects,
            tasks=task_objects,
            process_type=process_map[process]
        )
        
        crew_config = {
            'name': name,
            'id': crew_obj.id,
            'process_type': process,
            'agents': [agent.id for agent in agent_objects],
            'tasks': [task.id for task in task_objects],
            'created_at': crew_obj.created_at
        }
        
        if output:
            with open(output, 'w') as f:
                yaml.dump(crew_config, f, default_flow_style=False)
            click.echo(f"Crew configuration saved to {output}")
        else:
            click.echo(yaml.dump(crew_config, default_flow_style=False))
        
        click.echo(f"✅ Crew '{name}' created successfully with ID: {crew_obj.id}")
        
    except Exception as e:
        click.echo(f"❌ Error creating crew: {str(e)}", err=True)
        sys.exit(1)


@crew.command()
@click.argument('crew_file', type=click.Path(exists=True))
@click.option('--mode', '-m', type=click.Choice(['sync', 'async', 'parallel']),
              default='async', help='Execution mode')
@click.option('--inputs', '-i', type=click.Path(exists=True), help='Input data file (JSON/YAML)')
@click.option('--timeout', '-t', type=int, help='Execution timeout in seconds')
def execute(crew_file, mode, inputs, timeout):
    """Execute a crew"""
    try:
        # Load crew configuration
        with open(crew_file, 'r') as f:
            crew_config = yaml.safe_load(f)
        
        # Load input data if provided
        input_data = None
        if inputs:
            with open(inputs, 'r') as f:
                if inputs.endswith('.json'):
                    input_data = json.load(f)
                else:
                    input_data = yaml.safe_load(f)
        
        # For demo purposes, create a simple crew
        # In a real implementation, you would reconstruct the crew from config
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        
        research_task = BaseTask(
            description="Research the given topic",
            expected_output="Research findings and insights",
            task_type=TaskType.RESEARCH
        )
        
        analysis_task = BaseTask(
            description="Analyze the research findings",
            expected_output="Analysis report with recommendations",
            task_type=TaskType.ANALYSIS
        )
        
        research_task.assign_agent(researcher)
        analysis_task.assign_agent(analyst)
        analysis_task.add_dependency(research_task)
        
        crew_obj = BaseCrew(
            name=crew_config['name'],
            agents=[researcher, analyst],
            tasks=[research_task, analysis_task],
            process_type=ProcessType.SEQUENTIAL
        )
        
        # Execute crew
        executor = RuntimeExecutor()
        executor.start()
        
        mode_map = {
            'sync': ExecutionMode.SYNCHRONOUS,
            'async': ExecutionMode.ASYNCHRONOUS,
            'parallel': ExecutionMode.PARALLEL
        }
        
        click.echo(f"🚀 Executing crew '{crew_config['name']}' in {mode} mode...")
        
        result = executor.execute_crew(
            crew_obj,
            inputs=input_data,
            mode=mode_map[mode],
            timeout=timeout
        )
        
        # Handle async result
        if mode != 'sync':
            click.echo("⏳ Waiting for execution to complete...")
            result = result.result()  # Wait for Future to complete
        
        executor.stop()
        
        if result.success:
            click.echo("✅ Crew execution completed successfully!")
            click.echo(f"📊 Execution time: {result.execution_time:.2f} seconds")
            click.echo(f"📋 Results: {json.dumps(result.outputs, indent=2)}")
        else:
            click.echo("❌ Crew execution failed!")
            click.echo(f"🔍 Errors: {result.errors}")
        
    except Exception as e:
        click.echo(f"❌ Error executing crew: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--port', '-p', default=8501, help='Port for the studio interface')
@click.option('--host', '-h', default='localhost', help='Host for the studio interface')
def studio(port, host):
    """Launch the visual agent builder studio"""
    try:
        click.echo(f"🎨 Launching PsychoPy AI Agent Builder Studio...")
        click.echo(f"🌐 Access the studio at: http://{host}:{port}")
        
        # Import and launch Streamlit app
        import subprocess
        import sys
        
        studio_path = Path(__file__).parent.parent / "studio" / "main.py"
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(studio_path),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "false"
        ]
        
        subprocess.run(cmd)
        
    except Exception as e:
        click.echo(f"❌ Error launching studio: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def version():
    """Show version information"""
    click.echo("PsychoPy AI Agent Builder v1.0.0")
    click.echo("Built with ❤️ by the AI in PM team")


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='paab_config.yaml',
              help='Output configuration file')
def init(output):
    """Initialize a new PAAB project"""
    try:
        config = {
            'project': {
                'name': 'My AI Agent Project',
                'version': '1.0.0',
                'description': 'A new AI agent project built with PAAB'
            },
            'agents': {
                'default_config': {
                    'max_iterations': 10,
                    'memory_enabled': True,
                    'collaboration_enabled': True
                }
            },
            'crews': {
                'default_process_type': 'sequential',
                'max_concurrent_tasks': 5
            },
            'runtime': {
                'max_concurrent_crews': 10,
                'execution_timeout': 3600,
                'monitoring_enabled': True
            }
        }
        
        with open(output, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        click.echo(f"✅ PAAB project initialized!")
        click.echo(f"📁 Configuration saved to: {output}")
        click.echo(f"🚀 Get started by editing the configuration and creating your first agent:")
        click.echo(f"   paab agent create --name 'MyAgent' --role researcher --goal 'Research AI trends' --backstory 'Expert researcher'")
        
    except Exception as e:
        click.echo(f"❌ Error initializing project: {str(e)}", err=True)
        sys.exit(1)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            else:
                return yaml.safe_load(f)
    except Exception as e:
        click.echo(f"❌ Error loading configuration: {str(e)}", err=True)
        sys.exit(1)


def main():
    """Main CLI entry point"""
    cli()


if __name__ == '__main__':
    main()
