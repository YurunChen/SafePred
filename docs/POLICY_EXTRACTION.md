# Extracting Policies

SafePred uses **safety policies** to evaluate action risk. You can extract structured policies from your own documents (PDF, TXT, DOC, DOCX, or webpage URL) so that the guardrail aligns with your organization's guidelines.

## Prerequisites

- Set `rule_extractor_llm` (or reuse `world_model_llm`) and the corresponding API key/URL in `config/config.yaml` and `.env`.
- Run from the SafePred repo root. If the package is not installed, use `pip install -e .` first, or use `PYTHONPATH=. python models/llm_rule_extractor.py` instead of the `-m` form below.

## Basic usage

Run the policy extractor from the **SafePred repo root**:

```bash
# If SafePred is installed (pip install -e .):
python -m SafePred.models.llm_rule_extractor -d /path/to/your/document.pdf

# If not installed, from repo root:
PYTHONPATH=. python models/llm_rule_extractor.py -d /path/to/your/document.pdf

# With optional context and benchmark-specific action space
python -m SafePred.models.llm_rule_extractor -d /path/to/document.pdf \
  --organization "MyOrg" \
  --organization-description "Description of the organization" \
  --target-subject "User" \
  --bench stweb \
  --config config/config.yaml
```

## Options

| Option | Description |
|--------|-------------|
| `-d, --document` | **Required.** Path to the document (PDF, TXT, DOC, DOCX, etc.) or a webpage URL. |
| `-o, --output` | Output JSON path (default: save under `policies/` in a timestamped folder). |
| `--organization` | Organization name for context. |
| `--organization-description` | Short description of the organization. |
| `--target-subject` | Who the policies target (e.g., `User`, `Web Agent`). |
| `--bench` | `osharm` or `stweb` — use the benchmark's action space in the extraction prompt. |
| `--user-request` | Extra instructions for the extractor. |
| `--config` | Path to `config.yaml` (default: `config.yaml`; will search under `config/`). |

## Output

The script will (1) extract text from the document, (2) call the LLM to get structured policies, and (3) review and deduplicate. Outputs are written under `policies/<timestamp>/`, including a JSON policy file you can pass to `--policy` when running benchmarks or the wrapper.
