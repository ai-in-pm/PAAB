# Getting Started with PsychoPy AI Agent Builder

Welcome to PsychoPy AI Agent Builder (PAAB) - a powerful framework for building AI agent teams inspired by CrewAI and built on the solid foundation of PsychoPy.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Your First Agent](#your-first-agent)
5. [Building a Crew](#building-a-crew)
6. [Running Workflows](#running-workflows)
7. [Next Steps](#next-steps)

## Quick Start

Get up and running with PAAB in under 5 minutes:

```bash
# Install PAAB
pip install psychopy-ai-agent-builder

# Set up your environment
export OPENAI_API_KEY="your-api-key-here"

# Launch the visual studio
paab studio

# Or create your first agent via CLI
paab agent create --name "ResearchBot" --role researcher --goal "Research AI trends" --backstory "Expert AI researcher"
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least one LLM API key (OpenAI, Anthropic, etc.)

### Install from PyPI

```bash
pip install psychopy-ai-agent-builder
```

### Install from Source

```bash
git clone https://github.com/ai-in-pm/psychopy-ai-agent-builder.git
cd psychopy-ai-agent-builder
pip install -e .
```

### Docker Installation

```bash
# Pull the official image
docker pull paab:latest

# Run with Docker Compose
docker-compose up -d
```

### Development Installation

```bash
git clone https://github.com/ai-in-pm/psychopy-ai-agent-builder.git
cd psychopy-ai-agent-builder
pip install -e ".[dev]"
pre-commit install
```

## Core Concepts

PAAB transforms PsychoPy's experimental psychology concepts into AI agent equivalents:

### 🤖 Agents
Specialized AI entities with defined roles, goals, and capabilities.

```python
from src.agents.specialized import ResearcherAgent

researcher = ResearcherAgent(
    goal="Research the latest AI trends",
    backstory="You are an expert AI researcher with 10 years of experience",
    expertise_domains=["machine learning", "natural language processing"]
)
```

### 📝 Tasks
Individual assignments with clear objectives and success criteria.

```python
from src.tasks.base import BaseTask, TaskType

research_task = BaseTask(
    description="Research current trends in AI agent frameworks",
    expected_output="Comprehensive report with key findings and trends",
    task_type=TaskType.RESEARCH
)
```

### 👥 Crews
Teams of agents working together on complex objectives.

```python
from src.crews.base import BaseCrew, ProcessType

crew = BaseCrew(
    name="AI Research Team",
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process_type=ProcessType.SEQUENTIAL
)
```

### 🛠️ Tools
Extensible capabilities that agents can use to accomplish their tasks.

```python
from src.tools.llm_tools import OpenAITool

llm_tool = OpenAITool(model="gpt-4")
researcher.add_tool(llm_tool)
```

## Your First Agent

Let's create a simple research agent:

```python
from src.agents.specialized import ResearcherAgent
from src.tools.llm_tools import get_default_llm_tools

# Create a research agent
researcher = ResearcherAgent(
    goal="Research and analyze AI technology trends",
    backstory="""You are a senior AI researcher with expertise in machine learning, 
                 natural language processing, and emerging AI technologies. You excel at 
                 finding relevant information and synthesizing insights.""",
    expertise_domains=["AI", "machine learning", "NLP", "computer vision"]
)

# Add tools to the agent
for tool in get_default_llm_tools():
    researcher.add_tool(tool)

# Test the agent
status = researcher.get_status()
print(f"Agent {researcher.role} is ready with {len(researcher.tools)} tools")
```

## Building a Crew

Now let's create a complete research crew:

```python
from src.agents.specialized import ResearcherAgent, AnalystAgent, WriterAgent
from src.tasks.base import BaseTask, TaskType, TaskPriority
from src.crews.base import BaseCrew, ProcessType

# Create specialized agents
researcher = ResearcherAgent(
    goal="Conduct thorough research on assigned topics",
    backstory="Expert researcher with deep analytical skills"
)

analyst = AnalystAgent(
    goal="Analyze research findings and extract insights",
    backstory="Data analyst with expertise in pattern recognition"
)

writer = WriterAgent(
    goal="Create clear, engaging reports from research and analysis",
    backstory="Technical writer with excellent communication skills"
)

# Define tasks
research_task = BaseTask(
    description="Research the current state of AI agent frameworks",
    expected_output="Detailed research findings with sources",
    task_type=TaskType.RESEARCH,
    priority=TaskPriority.HIGH
)

analysis_task = BaseTask(
    description="Analyze the research findings for trends and insights",
    expected_output="Analysis report with key insights and recommendations",
    task_type=TaskType.ANALYSIS,
    priority=TaskPriority.HIGH
)

writing_task = BaseTask(
    description="Write a comprehensive report combining research and analysis",
    expected_output="Professional report suitable for stakeholders",
    task_type=TaskType.WRITING,
    priority=TaskPriority.MEDIUM
)

# Assign tasks to agents
research_task.assign_agent(researcher)
analysis_task.assign_agent(analyst)
writing_task.assign_agent(writer)

# Set up dependencies
analysis_task.add_dependency(research_task)
writing_task.add_dependency(analysis_task)

# Create the crew
crew = BaseCrew(
    name="AI Research Crew",
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process_type=ProcessType.SEQUENTIAL,
    verbose=True
)

print(f"Crew '{crew.name}' created with {len(crew.agents)} agents and {len(crew.tasks)} tasks")
```

## Running Workflows

Execute your crew using the runtime executor:

```python
from src.runtime.executor import RuntimeExecutor, ExecutionMode

# Initialize the executor
executor = RuntimeExecutor(
    max_concurrent_crews=5,
    max_workers=10,
    enable_monitoring=True
)

# Start the executor
executor.start()

# Execute the crew
result = executor.execute_crew(
    crew=crew,
    inputs={"topic": "AI agent frameworks", "deadline": "urgent"},
    mode=ExecutionMode.ASYNCHRONOUS,
    timeout=1800  # 30 minutes
)

# Wait for completion and get results
if hasattr(result, 'result'):  # Async execution
    final_result = result.result()
else:  # Sync execution
    final_result = result

# Display results
if final_result.success:
    print("✅ Crew execution completed successfully!")
    print(f"⏱️ Execution time: {final_result.execution_time:.2f} seconds")
    print("📋 Results:")
    for key, value in final_result.outputs.items():
        print(f"  {key}: {str(value)[:100]}...")
else:
    print("❌ Crew execution failed!")
    print(f"Errors: {final_result.errors}")

# Stop the executor
executor.stop()
```

## Using the Visual Studio

PAAB includes a powerful visual interface built with Streamlit:

```bash
# Launch the studio
paab studio

# Or specify custom host/port
paab studio --host 0.0.0.0 --port 8501
```

The studio provides:

- **Visual Agent Builder**: Drag-and-drop agent creation
- **Task Designer**: Visual task definition and dependency management
- **Crew Orchestrator**: Team composition and workflow design
- **Execution Monitor**: Real-time execution tracking
- **Performance Analytics**: Detailed metrics and insights

## Command Line Interface

PAAB provides a comprehensive CLI for all operations:

```bash
# Initialize a new project
paab init

# Create agents
paab agent create --name "DataScientist" --role analyst --goal "Analyze data" --backstory "Expert analyst"

# Create tasks
paab task create --description "Analyze sales data" --expected-output "Sales insights report" --type analysis

# Create crews
paab crew create --name "DataTeam" --agents agent1.yaml --tasks task1.yaml --process sequential

# Execute crews
paab crew execute crew_config.yaml --mode async --timeout 3600

# Get help
paab --help
paab agent --help
paab crew --help
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Runtime Configuration
PAAB_LOG_LEVEL=INFO
PAAB_MAX_CONCURRENT_CREWS=10
PAAB_MAX_WORKERS=50

# Storage
PAAB_DATA_PATH=./data
PAAB_CACHE_PATH=./cache
PAAB_LOGS_PATH=./logs

# Optional: Database
DATABASE_URL=postgresql://user:pass@localhost:5432/paab
REDIS_URL=redis://localhost:6379/0
```

### Configuration File

Create `config/config.yaml`:

```yaml
project:
  name: "My AI Agent Project"
  version: "1.0.0"

runtime:
  max_concurrent_crews: 10
  max_workers: 50
  execution_timeout: 3600
  monitoring_enabled: true

agents:
  default_max_iterations: 10
  default_memory_enabled: true
  default_collaboration_enabled: true

llm:
  default_provider: "openai"
  default_model: "gpt-4"
  default_temperature: 0.7
```

## Best Practices

### 1. Agent Design
- Give agents clear, specific roles and goals
- Provide detailed backstories for better context
- Use appropriate expertise domains for specialized agents

### 2. Task Definition
- Write clear, actionable task descriptions
- Define specific expected outputs
- Set appropriate priorities and dependencies

### 3. Crew Organization
- Choose the right process type for your workflow
- Consider agent capabilities when assigning tasks
- Use collaboration patterns effectively

### 4. Performance Optimization
- Monitor execution metrics
- Use caching for repeated operations
- Optimize resource allocation

### 5. Error Handling
- Implement proper error handling in custom tools
- Use retry mechanisms for unreliable operations
- Monitor and log errors for debugging

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure PAAB is properly installed
pip install --upgrade psychopy-ai-agent-builder

# Check Python path
python -c "import sys; print(sys.path)"
```

**API Key Issues**
```bash
# Verify API keys are set
echo $OPENAI_API_KEY

# Test API connectivity
paab test-api --provider openai
```

**Memory Issues**
```bash
# Check system resources
paab system-info

# Reduce concurrent crews
export PAAB_MAX_CONCURRENT_CREWS=5
```

**Permission Issues**
```bash
# Check directory permissions
ls -la data/ cache/ logs/

# Fix permissions
chmod 755 data cache logs
```

## Next Steps

Now that you have the basics, explore these advanced features:

1. **[Custom Tools](tools-guide.md)** - Build specialized tools for your agents
2. **[Memory Systems](memory-guide.md)** - Implement learning and adaptation
3. **[Flow Management](flows-guide.md)** - Create complex workflows
4. **[Monitoring](monitoring-guide.md)** - Set up comprehensive monitoring
5. **[Deployment](deployment-guide.md)** - Deploy to production environments

## Examples

Check out the `examples/` directory for complete working examples:

- `basic_research_crew.py` - Simple research workflow
- `data_analysis_pipeline.py` - Data processing crew
- `content_creation_team.py` - Content generation workflow
- `software_development_crew.py` - Code generation and review
- `customer_service_bot.py` - Customer support automation

## Community and Support

- **Documentation**: [Full documentation](https://psychopy-ai-agent-builder.readthedocs.io)
- **GitHub**: [Source code and issues](https://github.com/ai-in-pm/psychopy-ai-agent-builder)
- **Discussions**: [Community forum](https://github.com/ai-in-pm/psychopy-ai-agent-builder/discussions)
- **Examples**: [Example repository](https://github.com/ai-in-pm/paab-examples)

Welcome to the future of AI agent development! 🚀
