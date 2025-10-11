* [Evals and SteelThread](/evals-steel-thread)
* Introducing Steel Thread

On this page

# Introducing Steel Thread

Steel Thread is a lightweight, extensible framework for evaluating LLM agents — designed to help teams measure quality, catch regressions, and improve performance with minimal friction.

Steel Thread is built around two core abstractions:

* **Streams** are dynamic datasets built from sampling real plans and plan runs in production allowing you to monitor live agent behaviour.
* **Evals** are static data sets designed to be run multiple times to allow you to analyze the impact of changes to your agents before deploying them.

Note that Steel Thread is built to work specifically with Portia cloud and therefore requires a Portia API key.

---

## Why we built Steel Thread[​](#why-we-built-steel-thread "Direct link to Why we built Steel Thread")

Evaluating agents isn’t hard because the models are bad — it’s hard because:

* The output space is non-deterministic.
* The tool usage is complex and multi-step.
* The definition of "correct" can be subjective.
* And most of all: **curating test data is painful** ☠️.

We found that most eval frameworks fall down not on logic or metrics — but on data. They assume someone else is maintaining eval datasets that are 1) clean, 2) up-to-date with how their agents are behaving in production.

Instead of asking teams to build new datasets from scratch, Steel Thread **plugs directly into the data you already generate in Portia Cloud**:

* Plans
* Plan Runs
* Tool Calls
* User IDs
* Metadata and outputs

Now, every agent execution can become an eval — either retrospectively or in real time.

---

## What does it do?[​](#what-does-it-do "Direct link to What does it do?")

Steel Thread helps you figure out whether your agents are getting better or worse across runs. It does this by providing:

### 🌊 Streams[​](#-streams "Direct link to 🌊 Streams")

Run against your live or recent production runs. More commonly referred to as 'online evals', they are useful for:

* Monitoring quality in production usage.
* Tracking performance across time or model changes.
* Detecting silent failures.

### 📈 Evals[​](#-evals "Direct link to 📈 Evals")

Run against curated static datasets. More commonly referred to as 'offline evals', they are useful for:

* Iterating on prompts.
* Testing new chains of tool calls.
* Benchmarking models.
* Catching regressions on common use cases before deployment.

### 🎯 Custom metrics[​](#-custom-metrics "Direct link to 🎯 Custom metrics")

Use both determistic evaluators or LLMs-as-judge ones to compute:

* Accuracy
* Completeness
* Clarity
* Efficiency
* Latency
* Tool usage
* ...or domain-specific checks

---

## Built for Portia[​](#built-for-portia "Direct link to Built for Portia")

Steel Thread is deeply integrated with the Portia SDK and cloud. It works natively with:

* Plan and PlanRun IDs
* Tool call metadata
* End user context
* Agent outputs (e.g. final outputs, intermediate values)
* APIs and UI features in Portia Cloud

This means you don’t need to create new test harnesses or annotate synthetic datasets — you can evaluate what's already happening.

Just point Steel Thread at your Portia instance, and start measuring.

Last updated on **Sep 9, 2025** by **github-actions[bot]**