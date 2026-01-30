<p align="center">
  <h1><strong>SafePred</strong></h1>
</p>

## Abstract

With the widespread deployment of Computer-using Agents (CUAs) in complex real-world environments, long-term risks often lead to severe and irreversible consequences. Most existing guardrails adopt a *reactive* approach—constraining behavior only within the current observation space. They can prevent immediate risks (e.g., clicking a phishing link) but cannot avoid *long-term* risks: seemingly reasonable actions can yield high-risk outcomes that appear only later (e.g., cleaning logs makes future audits untraceable), which reactive guardrails cannot see in the current observation. We propose a **predictive guardrail** approach: align predicted future risks with current decisions. **SafePred** implements this via:

1. **Short- and long-term risk prediction** — Using safety policies as the basis, SafePred leverages a world model to produce semantic risk representations (short- and long-term), identifying and pruning actions that lead to high-risk states.
2. **Decision optimization** — Translating predicted risks into actionable guidance through step-level interventions and task-level re-planning.

Extensive experiments show that SafePred significantly reduces high-risk behaviors, achieving over 97.6% safety performance and improving task utility by up to 21.4% compared with reactive baselines.

---

## Status

> **Code will be released in one week.** 


