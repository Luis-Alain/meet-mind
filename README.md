# MeetMind

**AI-powered meeting intelligence and action-item extraction.**

MeetMind is an AI-powered application designed to transform meeting transcripts into structured, actionable tasks. It aims to identify decisions, extract action items, assign priorities and deadlines, and organize responsibilities based on the information discussed during a meeting.

The project is being developed in Python, using Amazon Bedrock for generative AI capabilities and exploring Strands Agents and local decision models for structured task classification.

> **Project status:** Early development.

## Features

- Meeting transcript ingestion.
- Action-item extraction from meeting conversations.
- Task prioritization and classification.
- Assignee and deadline extraction grounded in transcript evidence.
- Structured output validation using Pydantic.
- Human review and editing of generated tasks.
- Task persistence using SQLite.
- Export to JSON and CSV.
- Integrations with project management platforms such as Asana, Trello, and Jira.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| uv | Dependency and project management |
| boto3 | AWS SDK for Python |
| Amazon Bedrock | Generative AI inference |
| Strands Agents SDK | Agent orchestration and tool integration |
| FastAPI | REST API |
| Pydantic | Data validation and structured outputs |
| SQLite | local persistence |
| pytest | Automated testing |


## Prerequisites

Before running the project, make sure you have:

- Python 3.13 or a compatible version configured for the project.
- [uv](https://docs.astral.sh/uv/).
- An AWS account with access to Amazon Bedrock.
- AWS credentials configured locally.
- Access to a supported Bedrock model or inference profile in your selected AWS region.

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd meet-mind
```

Replace `<repository-url>` with the repository's actual URL.

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure AWS credentials

If you use the default AWS CLI profile:

```bash
aws configure
```

Verify your identity:

```bash
aws sts get-caller-identity
```

If you use a named profile, specify it when running AWS CLI commands and configure your application to use the same profile.

### 4. Configure environment variables

Create a local environment file:

```bash
cp .env.example .env
```

Add the required configuration:

```dotenv
AWS_REGION=us-east-1
AWS_PROFILE=default
AWS_BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
```

Use a model ID or inference profile that is available and authorized for your AWS account. The example above uses a regional inference profile; verify its availability before relying on it.

**Security:** Never commit `.env` or expose AWS access keys in source code. Prefer AWS CLI profiles, IAM roles, or another supported credential provider.

### 5. Run the application

```bash
uv run src/meet_mind/main.py
```

This currently runs the initial Bedrock Converse API example. As the project evolves, a package entry point or development server command may replace this command.

## Configuration

| Variable | Description |
|---|---|
| `AWS_REGION` | AWS region used by the application |
| `AWS_PROFILE` | Optional AWS CLI profile name |
| `AWS_BEDROCK_MODEL_ID` | Bedrock model ID or inference profile ID/ARN |

The application should use the standard AWS credential provider chain rather than storing credentials directly in configuration files.

## Testing

Run the test suite with:

```bash
uv run pytest
```

As tests are added, the project should cover data validation, task extraction, date and assignee handling, API behavior, persistence, and export functionality.

Model evaluation should also measure extraction accuracy, priority classification quality, unsupported claims, and latency.

## Design Principles

- **Evidence-based extraction:** Tasks, assignees, deadlines, and decisions should be supported by the meeting transcript.
- **Explicit uncertainty:** Missing or ambiguous information should remain unresolved rather than being invented.
- **Human oversight:** Users should be able to review and edit generated tasks before exporting them.
- **Deterministic validation:** Business rules and schema validation should be enforced in application code.
- **Evaluated model selection:** Generative and decision models should be selected based on measurable quality, cost, latency, and deployment requirements.
- **Modular architecture:** Model providers, task extraction, validation, persistence, and integrations should remain independently testable.

## Contributing

This project is under active development. Contributions should follow the existing project structure, include tests where appropriate, and keep credentials and sensitive meeting data out of version control.

## License

No license has been specified yet. Until a license is added to the repository, reuse and redistribution are subject to applicable copyright law.
