# LLM Rule Extractor Usage Guide

## Overview

`llm_rule_extractor` uses an LLM to extract safety policies from documents (PDF, TXT, DOC, etc.).

## Prerequisites

1. Configure LLM API in `config.yaml`: `rule_extractor_llm` (preferred) or `world_model_llm` (fallback).
2. Install dependencies.

## Command Line

### Basic

```bash
python -m SafePred.models.llm_rule_extractor -d document/wasp_risk.pdf
```

### Full example

```bash
python -m SafePred.models.llm_rule_extractor \
  -d document/wasp_risk.pdf \
  --organization 'WebArena' \
  --organization-description 'A web automation platform for testing AI agents' \
  --target-subject 'AI agent' \
  --context 'Safety policies for web automation agents' \
  -o /path/to/output/policies.json \
  --config config/config.yaml
```

### Parameters

**Required:** `-d, --document`: path to document (PDF, TXT, DOC, DOCX).

**Optional:** `-o, --output` (default: `policies/`), `-n, --name`, `--context`, `--organization`, `--organization-description`, `--target-subject`, `--config`.

## Python API

```python
from SafePred.models.llm_rule_extractor import LLMRuleExtractor
from SafePred.config.config import SafetyConfig

config = SafetyConfig.from_yaml("config/config.yaml")
llm_config = config.get_llm_config("rule_extractor")

extractor = LLMRuleExtractor(
    api_key=llm_config["api_key"],
    api_url=llm_config["api_url"],
    model_name=llm_config["model_name"],
    provider=llm_config["provider"],
    timeout=llm_config["timeout"],
    temperature=llm_config["temperature"],
    max_tokens=llm_config["max_tokens"],
)

policies = extractor.extract_rules_from_file(
    file_path="document/wasp_risk.pdf",
    organization="WebArena",
    organization_description="A web automation platform for testing AI agents",
    target_subject="AI agent",
    context="Safety policies for web automation agents"
)

import json
with open("output_policies.json", "w", encoding="utf-8") as f:
    json.dump({"policies": policies}, f, indent=2, ensure_ascii=False)
```

### From text

```python
policies = extractor.extract_rules_from_text(
    text="Your document text here...",
    organization="WebArena",
    target_subject="AI agent",
    organization_description="Web automation platform"
)
```

### From directory

```python
policies = extractor.extract_rules_from_directory(
    directory="documents/",
    pattern="*.pdf",
    organization="WebArena",
    target_subject="AI agent",
    organization_description="Web automation platform"
)
```

## Output format

JSON with `policies` array; each policy has `id`, `name`, `description`, `risk_patterns`, `severity`.

## Troubleshooting

- **ModuleNotFoundError**: Run from project root or set PYTHONPATH.
- **API errors**: Check `.env` and `config.yaml` (API keys by provider).
- **Extraction fails**: Verify document format and that it contains policy content.
