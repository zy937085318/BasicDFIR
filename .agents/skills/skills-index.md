# Skills Index

> **Do NOT read all SKILL.md files at once.** Use this index to find the right skill, then read only that one.

## Core Pipeline Skills

| Skill | Path | Description |
|-------|------|-------------|
| dataset-discovery | `.agents/skills/dataset-discovery/SKILL.md` | Multi-source ML dataset discovery. Search HuggingFace Hub, OpenML, GitHub, and paper cross-references for datasets re... |
| init-analysis | `.agents/skills/init-analysis/SKILL.md` | This skill should be used when the user asks to "run initial analysis", "analyze single-cell data", "QC my data", "ru... |
| inno-code-survey | `.agents/skills/inno-code-survey/SKILL.md` | Acquires missing code repositories for the selected idea (Phase A) and conducts comprehensive code survey mapping aca... |
| inno-deep-research | `.agents/skills/inno-deep-research/SKILL.md` | Comprehensive research assistant that synthesizes information from multiple sources with citations. Use when: conduct... |
| inno-experiment-analysis | `.agents/skills/inno-experiment-analysis/SKILL.md` | This skill should be used when the user asks to "analyze experimental results", "generate results section", "statisti... |
| inno-experiment-dev | `.agents/skills/inno-experiment-dev/SKILL.md` | Creates implementation plan, writes project code with judge feedback loop, and submits final experiment run. Use afte... |
| inno-figure-gen | `.agents/skills/inno-figure-gen/SKILL.md` | Generate/edit images with Gemini image models (default: gemini-3.1-flash-image-preview). Use for image create/modify ... |
| inno-grant-proposal | `.agents/skills/inno-grant-proposal/SKILL.md` | Help professors and researchers write, revise, adapt, and polish grant proposals for US agencies (NSF, NIH, DOE, DARP... |
| inno-humanizer | `.agents/skills/inno-humanizer/SKILL.md` | Remove signs of AI-generated writing from text. Use when editing or reviewing text to make it sound more natural and ... |
| inno-idea-eval | `.agents/skills/inno-idea-eval/SKILL.md` | Multi-persona idea evaluation with quality gate. Evaluates ideas across 5 InnoEval dimensions (Clarity, Novelty, Vali... |
| inno-idea-generation | `.agents/skills/inno-idea-generation/SKILL.md` | Facilitates structured brainstorming sessions, conducts comprehensive research, and generates creative solutions usin... |
| inno-paper-reviewer | `.agents/skills/inno-paper-reviewer/SKILL.md` | Structured manuscript/grant review with checklist-based evaluation. Use when writing formal peer reviews with specifi... |
| inno-paper-writing | `.agents/skills/inno-paper-writing/SKILL.md` | Creates formal academic research papers following IEEE/ACM formatting standards with proper structure, citations, and... |
| inno-pipeline-planner | `.agents/skills/inno-pipeline-planner/SKILL.md` | Guides the user through an interactive conversation to define their research project, then generates research_brief.j... |
| inno-prepare-resources | `.agents/skills/inno-prepare-resources/SKILL.md` | Loads the evaluation instance, searches GitHub for related repositories, builds a dataset description, queries the Pr... |
| inno-rclone-to-overleaf | `.agents/skills/inno-rclone-to-overleaf/SKILL.md` | Access Overleaf projects via CLI. Use for reading/writing LaTeX files, syncing local .tex files to Overleaf, download... |
| inno-rebuttal | `.agents/skills/inno-rebuttal/SKILL.md` | Drafting and refining academic rebuttals for top-tier AI/CS conferences (NeurIPS, ICML, ICLR, CVPR, ECCV, AAAI, ARR, ... |
| inno-reference-audit | `.agents/skills/inno-reference-audit/SKILL.md` | This skill provides reference guidance for citation verification in academic writing. Use when the user asks about "c... |

## Library Skills

| Skill | Path | Description |
|-------|------|-------------|
| academic-researcher | `.agents/skills/library/academic-researcher/SKILL.md` | Academic research assistant for literature reviews, paper analysis, and scholarly writing. Use when: reviewing academ... |
| huggingface-accelerate | `.agents/skills/library/accelerate/SKILL.md` | Simplest distributed training API. 4 lines to add distributed support to any PyTorch script. Unified API for DeepSpee... |
| aris-ablation-planner | `.agents/skills/library/aris-ablation-planner/SKILL.md` | Use when main results pass result-to-claim (claim_supported=yes or partial) and ablation studies are needed for paper... |
| aris-analyze-results | `.agents/skills/library/aris-analyze-results/SKILL.md` | Analyze ML experiment results, compute statistics, generate comparison tables and insights. Use when user says "analy... |
| aris-arxiv | `.agents/skills/library/aris-arxiv/SKILL.md` | Search, download, and summarize academic papers from arXiv. Use when user says "search arxiv", "download paper", "fet... |
| aris-auto-paper-improvement-loop | `.agents/skills/library/aris-auto-paper-improvement-loop/SKILL.md` | Autonomously improve a generated paper via GPT-5.4 xhigh review → implement fixes → recompile, for 2 rounds. Use when... |
| aris-auto-review-loop | `.agents/skills/library/aris-auto-review-loop/SKILL.md` | Autonomous multi-round research review loop. Repeatedly reviews via Codex MCP, implements fixes, and re-reviews until... |
| aris-comm-lit-review | `.agents/skills/library/aris-comm-lit-review/SKILL.md` | Communications-domain literature review with Claude-style knowledge-base-first retrieval. Use when the task is about ... |
| aris-dse-loop | `.agents/skills/library/aris-dse-loop/SKILL.md` | Autonomous design space exploration loop for computer architecture and EDA. Runs a program, analyzes results, tunes p... |
| aris-experiment-bridge | `.agents/skills/library/aris-experiment-bridge/SKILL.md` | Workflow 1.5: Bridge between idea discovery and auto review. Reads EXPERIMENT_PLAN.md, implements experiment code, de... |
| aris-experiment-plan | `.agents/skills/library/aris-experiment-plan/SKILL.md` | Turn a refined research proposal or method idea into a detailed, claim-driven experiment roadmap. Use after `aris-res... |
| aris-feishu-notify | `.agents/skills/library/aris-feishu-notify/SKILL.md` | Send notifications to Feishu/Lark. Internal utility used by other skills, or manually via /feishu-notify. Supports pu... |
| aris-formula-derivation | `.agents/skills/library/aris-formula-derivation/SKILL.md` | Structures and derives research formulas when the user wants to 推导公式, build a theory line, organize assumptions, turn... |
| aris-grant-proposal | `.agents/skills/library/aris-grant-proposal/SKILL.md` | Draft a structured grant proposal from research ideas and literature. Supports KAKENHI (Japan), NSF (US), NSFC (China... |
| aris-idea-creator | `.agents/skills/library/aris-idea-creator/SKILL.md` | Generate and rank research ideas given a broad direction. Use when user says "找idea", "brainstorm ideas", "generate r... |
| aris-idea-discovery | `.agents/skills/library/aris-idea-discovery/SKILL.md` | Workflow 1: Full idea discovery pipeline. Orchestrates research-lit → idea-creator → novelty-check → research-review ... |
| aris-infra | `.agents/skills/library/aris-infra/SKILL.md` | ARIS (Auto-claude-code-research-in-sleep) infrastructure setup and configuration. Configures MCP servers for cross-mo... |
| aris-mermaid-diagram | `.agents/skills/library/aris-mermaid-diagram/SKILL.md` | Generate Mermaid diagrams from user requirements. Saves .mmd and .md files to figures/ directory with syntax verifica... |
| aris-meta-optimize | `.agents/skills/library/aris-meta-optimize/SKILL.md` | Analyze ARIS usage logs and propose optimizations to SKILL.md files, reviewer prompts, and workflow defaults. Outer-l... |
| aris-monitor-experiment | `.agents/skills/library/aris-monitor-experiment/SKILL.md` | Monitor running experiments, check progress, collect results. Use when user says "check results", "is it done", "moni... |
| aris-novelty-check | `.agents/skills/library/aris-novelty-check/SKILL.md` | Verify research idea novelty against recent literature. Use when user says "查新", "novelty check", "有没有人做过", "check no... |
| aris-paper-compile | `.agents/skills/library/aris-paper-compile/SKILL.md` | Compile LaTeX paper to PDF, fix errors, and verify output. Use when user says "编译论文", "compile paper", "build PDF", "... |
| aris-paper-figure | `.agents/skills/library/aris-paper-figure/SKILL.md` | Generate publication-quality figures and tables from experiment results. Use when user says "画图", "作图", "generate fig... |
| aris-paper-illustration | `.agents/skills/library/aris-paper-illustration/SKILL.md` | Generate publication-quality AI illustrations for academic papers using Gemini image generation. Creates architecture... |
| aris-paper-plan | `.agents/skills/library/aris-paper-plan/SKILL.md` | Generate a structured paper outline from review conclusions and experiment results. Use when user says "写大纲", "paper ... |
| aris-paper-poster | `.agents/skills/library/aris-paper-poster/SKILL.md` | Generate a conference poster (article + tcbposter LaTeX → A0/A1 PDF + editable PPTX + SVG) from a compiled paper. Use... |
| aris-paper-slides | `.agents/skills/library/aris-paper-slides/SKILL.md` | Generate conference presentation slides (beamer LaTeX → PDF + editable PPTX) from a compiled paper, with speaker note... |
| aris-paper-write | `.agents/skills/library/aris-paper-write/SKILL.md` | Draft LaTeX paper section by section from an outline. Use when user says "写论文", "write paper", "draft LaTeX", "开始写", ... |
| aris-paper-writing | `.agents/skills/library/aris-paper-writing/SKILL.md` | Workflow 3: Full paper writing pipeline. Orchestrates paper-plan → paper-figure → paper-write → paper-compile → auto-... |
| aris-pixel-art | `.agents/skills/library/aris-pixel-art/SKILL.md` | Generate pixel art SVG illustrations for READMEs, docs, or slides. Use when user says "画像素图", "pixel art", "make an S... |
| aris-proof-writer | `.agents/skills/library/aris-proof-writer/SKILL.md` | Writes rigorous mathematical proofs for ML/AI theory. Use when asked to prove a theorem, lemma, proposition, or corol... |
| aris-rebuttal | `.agents/skills/library/aris-rebuttal/SKILL.md` | Workflow 4: Submission rebuttal pipeline. Parses external reviews, enforces coverage and grounding, drafts a safe tex... |
| aris-research-lit | `.agents/skills/library/aris-research-lit/SKILL.md` | Search and analyze research papers, find related work, summarize key ideas. Use when user says "find papers", "relate... |
| aris-research-pipeline | `.agents/skills/library/aris-research-pipeline/SKILL.md` | Full research pipeline: Workflow 1 (idea discovery) → implementation → Workflow 2 (auto review loop). Goes from a bro... |
| aris-research-refine | `.agents/skills/library/aris-research-refine/SKILL.md` | Turn a vague research direction into a problem-anchored, elegant, frontier-aware, implementation-oriented method plan... |
| aris-research-refine-pipeline | `.agents/skills/library/aris-research-refine-pipeline/SKILL.md` | Run an end-to-end workflow that chains `aris-research-refine` and `aris-experiment-plan`. Use when the user wants a o... |
| aris-research-review | `.agents/skills/library/aris-research-review/SKILL.md` | Get a deep critical review of research from GPT via Codex MCP. Use when user says "review my research", "help me revi... |
| aris-research-wiki | `.agents/skills/library/aris-research-wiki/SKILL.md` | Persistent research knowledge base that accumulates papers, ideas, experiments, claims, and their relationships acros... |
| aris-result-to-claim | `.agents/skills/library/aris-result-to-claim/SKILL.md` | Use when experiments complete to judge what claims the results support, what they don't, and what evidence is still m... |
| aris-run-experiment | `.agents/skills/library/aris-run-experiment/SKILL.md` | Deploy and run ML experiments on local, remote, Vast.ai, or Modal serverless GPU. Use when user says "run experiment"... |
| aris-semantic-scholar | `.agents/skills/library/aris-semantic-scholar/SKILL.md` | Search published venue papers (IEEE, ACM, Springer, etc.) via Semantic Scholar API. Complements /aris-arxiv (preprint... |
| aris-serverless-modal | `.agents/skills/library/aris-serverless-modal/SKILL.md` | Run GPU workloads on Modal — training, fine-tuning, inference, batch processing. Zero-config serverless: no SSH, no D... |
| aris-system-profile | `.agents/skills/library/aris-system-profile/SKILL.md` | Profile a target (script, process, GPU, memory, interconnect) using external tools and code instrumentation. Produces... |
| aris-training-check | `.agents/skills/library/aris-training-check/SKILL.md` | Periodically check WandB metrics during training to catch problems early (NaN, loss divergence, idle GPUs). Avoids wa... |
| aris-vast-gpu | `.agents/skills/library/aris-vast-gpu/SKILL.md` | Rent, manage, and destroy GPU instances on vast.ai. Use when user says "rent gpu", "vast.ai", "rent a server", "cloud... |
| audiocraft-audio-generation | `.agents/skills/library/audiocraft/SKILL.md` | PyTorch library for audio generation including text-to-music (MusicGen) and text-to-sound (AudioGen). Use when you ne... |
| autogpt-agents | `.agents/skills/library/autogpt/SKILL.md` | Autonomous AI agent platform for building and deploying continuous agents. Use when creating visual workflow agents, ... |
| autoresearch | `.agents/skills/library/autoresearch/SKILL.md` |  |
| awq-quantization | `.agents/skills/library/awq/SKILL.md` | Activation-aware weight quantization for 4-bit LLM compression with 3x speedup and minimal accuracy loss. Use when de... |
| axolotl | `.agents/skills/library/axolotl/SKILL.md` | Expert guidance for fine-tuning LLMs with Axolotl - YAML configs, 100+ models, LoRA/QLoRA, DPO/KTO/ORPO/GRPO, multimo... |
| evaluating-code-models | `.agents/skills/library/bigcode-evaluation-harness/SKILL.md` | Evaluates code generation models across HumanEval, MBPP, MultiPL-E, and 15+ benchmarks with pass@k metrics. Use when ... |
| biorxiv-database | `.agents/skills/library/biorxiv-database/SKILL.md` | Efficient database search tool for bioRxiv preprint server. Use this skill when searching for life sciences preprints... |
| quantizing-models-bitsandbytes | `.agents/skills/library/bitsandbytes/SKILL.md` | Quantizes LLMs to 8-bit or 4-bit for 50-75% memory reduction with minimal accuracy loss. Use when GPU memory is limit... |
| blip-2-vision-language | `.agents/skills/library/blip-2/SKILL.md` | Vision-language pre-training framework bridging frozen image encoders and LLMs. Use when you need image captioning, v... |
| brainstorming-research-ideas | `.agents/skills/library/brainstorming-research-ideas/SKILL.md` | Guides researchers through structured ideation frameworks to discover high-impact research directions. Use when explo... |
| chroma | `.agents/skills/library/chroma/SKILL.md` | Open-source embedding database for AI applications. Store embeddings and metadata, perform vector and full-text searc... |
| clip | `.agents/skills/library/clip/SKILL.md` | OpenAI's model connecting vision and language. Enables zero-shot image classification, image-text matching, and cross... |
| constitutional-ai | `.agents/skills/library/constitutional-ai/SKILL.md` | Anthropic's method for training harmless AI through self-improvement. Two-phase approach - supervised learning with s... |
| creative-thinking-for-research | `.agents/skills/library/creative-thinking-for-research/SKILL.md` | Applies cognitive science frameworks for creative thinking to CS and AI research ideation. Use when seeking genuinely... |
| crewai-multi-agent | `.agents/skills/library/crewai/SKILL.md` | Multi-agent orchestration framework for autonomous AI collaboration. Use when building teams of specialized agents wo... |
| deepspeed | `.agents/skills/library/deepspeed/SKILL.md` | Expert guidance for distributed training with DeepSpeed - ZeRO optimization stages, pipeline parallelism, FP16/BF16/F... |
| ds-analysis-campaign | `.agents/skills/library/ds-analysis-campaign/SKILL.md` | Use when a quest needs one or more follow-up runs such as ablations, robustness checks, error analysis, or failure an... |
| ds-baseline | `.agents/skills/library/ds-baseline/SKILL.md` | Use when a quest needs to attach, import, reproduce, repair, verify, compare, or publish a baseline and its metrics. |
| ds-decision | `.agents/skills/library/ds-decision/SKILL.md` | Use when the quest needs an explicit go, stop, branch, reuse-baseline, write, finalize, reset, or user-decision trans... |
| ds-experiment | `.agents/skills/library/ds-experiment/SKILL.md` | Use when a quest is ready for a concrete implementation pass or a main experiment run tied to a selected idea and an ... |
| ds-figure-polish | `.agents/skills/library/ds-figure-polish/SKILL.md` | Use when a quest needs a polished milestone chart, paper-facing figure, appendix figure, or a mandatory render-inspec... |
| ds-finalize | `.agents/skills/library/ds-finalize/SKILL.md` | Use when the quest is ready to consolidate final claims, limitations, recommendations, summary state, and graph expor... |
| ds-full-pipeline | `.agents/skills/library/ds-full-pipeline/SKILL.md` | Full DeepScientist research pipeline: scout → baseline → idea → experiment → analysis → optimize → write → review → f... |
| ds-idea | `.agents/skills/library/ds-idea/SKILL.md` | Use when a quest needs concrete hypotheses, limitation analysis, candidate directions, or a selected idea relative to... |
| ds-intake-audit | `.agents/skills/library/ds-intake-audit/SKILL.md` | Use when a quest does not start from a blank state and the agent must first audit, trust-rank, and reconcile existing... |
| ds-optimize | `.agents/skills/library/ds-optimize/SKILL.md` | Use when an algorithm-first quest should manage candidate briefs, optimization frontier, branch promotion, or fusion-... |
| ds-rebuttal | `.agents/skills/library/ds-rebuttal/SKILL.md` | Use when a quest already has a paper, draft, or review package and the task is to map reviewer feedback into experime... |
| ds-review | `.agents/skills/library/ds-review/SKILL.md` | Use when a draft, paper, or paper-like report is substantial enough for an independent skeptical audit before finaliz... |
| ds-scout | `.agents/skills/library/ds-scout/SKILL.md` | Use when a quest needs problem framing, literature scouting, dataset or metric clarification, or baseline discovery b... |
| ds-write | `.agents/skills/library/ds-write/SKILL.md` | Use when a quest has enough evidence to draft or refine a paper, report, or research summary without inventing missin... |
| dspy | `.agents/skills/library/dspy/SKILL.md` | Build complex AI systems with declarative programming, optimize prompts automatically, create modular RAG systems and... |
| faiss | `.agents/skills/library/faiss/SKILL.md` | Facebook's library for efficient similarity search and clustering of dense vectors. Supports billions of vectors, GPU... |
| optimizing-attention-flash | `.agents/skills/library/flash-attention/SKILL.md` | Optimizes transformer attention with Flash Attention for 2-4x speedup and 10-20x memory reduction. Use when training/... |
| gemini-deep-research | `.agents/skills/library/gemini-deep-research/SKILL.md` | Perform deep, multi-source research using Google Gemini's Deep Research Agent. Use this skill whenever the user asks ... |
| gguf-quantization | `.agents/skills/library/gguf/SKILL.md` | GGUF format and llama.cpp quantization for efficient CPU/GPU inference. Use when deploying models on consumer hardwar... |
| gptq | `.agents/skills/library/gptq/SKILL.md` | Post-training 4-bit quantization for LLMs with minimal accuracy loss. Use for deploying large models (70B, 405B) on c... |
| grpo-rl-training | `.agents/skills/library/grpo-rl-training/SKILL.md` | Expert guidance for GRPO/RL fine-tuning with TRL for reasoning and task-specific model training |
| guidance | `.agents/skills/library/guidance/SKILL.md` | Control LLM output with regex and grammars, guarantee valid JSON/XML/code generation, enforce structured formats, and... |
| hqq-quantization | `.agents/skills/library/hqq/SKILL.md` | Half-Quadratic Quantization for LLMs without calibration data. Use when quantizing models to 4/3/2-bit precision with... |
| huggingface-tokenizers | `.agents/skills/library/huggingface-tokenizers/SKILL.md` | Fast tokenizers optimized for research and production. Rust-based implementation tokenizes 1GB in <20 seconds. Suppor... |
| instructor | `.agents/skills/library/instructor/SKILL.md` | Extract structured data from LLM responses with Pydantic validation, retry failed extractions automatically, parse co... |
| knowledge-distillation | `.agents/skills/library/knowledge-distillation/SKILL.md` | Compress large language models using knowledge distillation from teacher to student models. Use when deploying smalle... |
| lambda-labs-gpu-cloud | `.agents/skills/library/lambda-labs/SKILL.md` | Reserved and on-demand GPU cloud instances for ML training and inference. Use when you need dedicated GPU instances w... |
| langchain | `.agents/skills/library/langchain/SKILL.md` | Framework for building LLM-powered applications with agents, chains, and RAG. Supports multiple providers (OpenAI, An... |
| langsmith-observability | `.agents/skills/library/langsmith/SKILL.md` | LLM observability platform for tracing, evaluation, and monitoring. Use when debugging LLM applications, evaluating m... |
| implementing-llms-litgpt | `.agents/skills/library/litgpt/SKILL.md` | Implements and trains LLMs using Lightning AI's LitGPT with 20+ pretrained architectures (Llama, Gemma, Phi, Qwen, Mi... |
| llama-cpp | `.agents/skills/library/llama-cpp/SKILL.md` | Runs LLM inference on CPU, Apple Silicon, and consumer GPUs without NVIDIA hardware. Use for edge deployment, M1/M2/M... |
| llama-factory | `.agents/skills/library/llama-factory/SKILL.md` | Expert guidance for fine-tuning LLMs with LLaMA-Factory - WebUI no-code, 100+ models, 2/3/4/5/6/8-bit QLoRA, multimod... |
| llamaguard | `.agents/skills/library/llamaguard/SKILL.md` | Meta's 7-8B specialized moderation model for LLM input/output filtering. 6 safety categories - violence/hate, sexual ... |
| llamaindex | `.agents/skills/library/llamaindex/SKILL.md` | Data framework for building LLM applications with RAG. Specializes in document ingestion (300+ connectors), indexing,... |
| llava | `.agents/skills/library/llava/SKILL.md` | Large Language and Vision Assistant. Enables visual instruction tuning and image-based conversations. Combines CLIP v... |
| evaluating-llms-harness | `.agents/skills/library/lm-evaluation-harness/SKILL.md` | Evaluates LLMs across 60+ academic benchmarks (MMLU, HumanEval, GSM8K, TruthfulQA, HellaSwag). Use when benchmarking ... |
| long-context | `.agents/skills/library/long-context/SKILL.md` | Extend context windows of transformer models using RoPE, YaRN, ALiBi, and position interpolation techniques. Use when... |
| making-academic-presentations | `.agents/skills/library/making-academic-presentations/SKILL.md` | Create academic presentation slide decks and optionally demo videos from research papers. Use when the user asks to "... |
| mamba-architecture | `.agents/skills/library/mamba/SKILL.md` | State-space model with O(n) complexity vs Transformers' O(n²). 5× faster inference, million-token sequences, no KV ca... |
| training-llms-megatron | `.agents/skills/library/megatron-core/SKILL.md` | Trains large language models (2B-462B parameters) using NVIDIA Megatron-Core with advanced parallelism strategies. Us... |
| miles-rl-training | `.agents/skills/library/miles/SKILL.md` | Provides guidance for enterprise-grade RL training using miles, a production-ready fork of slime. Use when training l... |
| ml-paper-writing | `.agents/skills/library/ml-paper-writing/SKILL.md` | Write publication-ready ML/AI papers for NeurIPS, ICML, ICLR, ACL, AAAI, COLM. Use when drafting papers from research... |
| mlflow | `.agents/skills/library/mlflow/SKILL.md` | Track ML experiments, manage model registry with versioning, deploy models to production, and reproduce experiments w... |
| modal-serverless-gpu | `.agents/skills/library/modal/SKILL.md` | Serverless GPU cloud platform for running ML workloads. Use when you need on-demand GPU access without infrastructure... |
| model-merging | `.agents/skills/library/model-merging/SKILL.md` | Merge multiple fine-tuned models using mergekit to combine capabilities without retraining. Use when creating special... |
| model-pruning | `.agents/skills/library/model-pruning/SKILL.md` | Reduce LLM size and accelerate inference using pruning techniques like Wanda and SparseGPT. Use when compressing mode... |
| moe-training | `.agents/skills/library/moe-training/SKILL.md` | Train Mixture of Experts (MoE) models using DeepSpeed or HuggingFace. Use when training large-scale models with limit... |
| nanogpt | `.agents/skills/library/nanogpt/SKILL.md` | Educational GPT implementation in ~300 lines. Reproduces GPT-2 (124M) on OpenWebText. Clean, hackable code for learni... |
| nemo-curator | `.agents/skills/library/nemo-curator/SKILL.md` | GPU-accelerated data curation for LLM training. Supports text/image/video/audio. Features fuzzy deduplication (16× fa... |
| nemo-evaluator-sdk | `.agents/skills/library/nemo-evaluator/SKILL.md` | Evaluates LLMs across 100+ benchmarks from 18+ harnesses (MMLU, HumanEval, GSM8K, safety, VLM) with multi-backend exe... |
| nemo-guardrails | `.agents/skills/library/nemo-guardrails/SKILL.md` | NVIDIA's runtime safety framework for LLM applications. Features jailbreak detection, input/output validation, fact-c... |
| nnsight-remote-interpretability | `.agents/skills/library/nnsight/SKILL.md` | Provides guidance for interpreting and manipulating neural network internals using nnsight with optional NDIF remote ... |
| openrlhf-training | `.agents/skills/library/openrlhf/SKILL.md` | High-performance RLHF framework with Ray+vLLM acceleration. Use for PPO, GRPO, RLOO, DPO training of large models (7B... |
| outlines | `.agents/skills/library/outlines/SKILL.md` | Guarantee valid JSON/XML/code structure during generation, use Pydantic models for type-safe outputs, support local m... |
| paper-analyzer | `.agents/skills/library/paper-analyzer/SKILL.md` | Deep analysis of a single paper — generate structured notes with figures, evaluation, and knowledge graph updates |
| paper-finder | `.agents/skills/library/paper-finder/SKILL.md` | Search existing paper notes by title, author, keyword, or research domain |
| paper-image-extractor | `.agents/skills/library/paper-image-extractor/SKILL.md` | Extract figures from papers — prioritizes arXiv source package for high-quality images |
| peft-fine-tuning | `.agents/skills/library/peft/SKILL.md` | Parameter-efficient fine-tuning for LLMs using LoRA, QLoRA, and 25+ methods. Use when fine-tuning large models (7B-70... |
| phoenix-observability | `.agents/skills/library/phoenix/SKILL.md` | Open-source AI observability platform for LLM tracing, evaluation, and monitoring. Use when debugging LLM application... |
| pinecone | `.agents/skills/library/pinecone/SKILL.md` | Managed vector database for production AI applications. Fully managed, auto-scaling, with hybrid search (dense + spar... |
| prompt-guard | `.agents/skills/library/prompt-guard/SKILL.md` | Meta's 86M prompt injection and jailbreak detector. Filters malicious prompts and third-party data for LLM apps. 99%+... |
| pytorch-fsdp2 | `.agents/skills/library/pytorch-fsdp2/SKILL.md` | Adds PyTorch FSDP2 (fully_shard) to training scripts with correct init, sharding, mixed precision/offload config, and... |
| pytorch-lightning | `.agents/skills/library/pytorch-lightning/SKILL.md` | High-level PyTorch framework with Trainer class, automatic distributed training (DDP/FSDP/DeepSpeed), callbacks syste... |
| pyvene-interventions | `.agents/skills/library/pyvene/SKILL.md` | Provides guidance for performing causal interventions on PyTorch models using pyvene's declarative intervention frame... |
| qdrant-vector-search | `.agents/skills/library/qdrant/SKILL.md` | High-performance vector similarity search engine for RAG and semantic search. Use when building production RAG system... |
| ray-data | `.agents/skills/library/ray-data/SKILL.md` | Scalable data processing for ML workloads. Streaming execution across CPU/GPU, supports Parquet/CSV/JSON/images. Inte... |
| ray-train | `.agents/skills/library/ray-train/SKILL.md` | Distributed training orchestration across clusters. Scales PyTorch/TensorFlow/HuggingFace from laptop to 1000s of nod... |
| research-news | `.agents/skills/library/research-news/SKILL.md` | Daily paper recommendation workflow — search arXiv and Semantic Scholar, score and recommend papers |
| rwkv-architecture | `.agents/skills/library/rwkv/SKILL.md` | RNN+Transformer hybrid with O(n) inference. Linear time, infinite context, no KV cache. Train like GPT (parallel), in... |
| sparse-autoencoder-training | `.agents/skills/library/saelens/SKILL.md` | Provides guidance for training and analyzing Sparse Autoencoders (SAEs) using SAELens to decompose neural network act... |
| scientific-writing | `.agents/skills/library/scientific-writing/SKILL.md` | Core skill for the deep research and writing tool. Write scientific manuscripts in full paragraphs (never bullet poin... |
| segment-anything-model | `.agents/skills/library/segment-anything/SKILL.md` | Foundation model for image segmentation with zero-shot transfer. Use when you need to segment any object in images us... |
| sentence-transformers | `.agents/skills/library/sentence-transformers/SKILL.md` | Framework for state-of-the-art sentence, text, and image embeddings. Provides 5000+ pre-trained models for semantic s... |
| sentencepiece | `.agents/skills/library/sentencepiece/SKILL.md` | Language-independent tokenizer treating text as raw Unicode. Supports BPE and Unigram algorithms. Fast (50k sentences... |
| sglang | `.agents/skills/library/sglang/SKILL.md` | Fast structured generation and serving for LLMs with RadixAttention prefix caching. Use for JSON/regex outputs, const... |
| simpo-training | `.agents/skills/library/simpo/SKILL.md` | Simple Preference Optimization for LLM alignment. Reference-free alternative to DPO with better performance (+6.4 poi... |
| skypilot-multi-cloud-orchestration | `.agents/skills/library/skypilot/SKILL.md` | Multi-cloud orchestration for ML workloads with automatic cost optimization. Use when you need to run training or bat... |
| slime-rl-training | `.agents/skills/library/slime/SKILL.md` | Provides guidance for LLM post-training with RL using slime, a Megatron+SGLang framework. Use when training GLM model... |
| speculative-decoding | `.agents/skills/library/speculative-decoding/SKILL.md` | Accelerate LLM inference using speculative decoding, Medusa multiple heads, and lookahead decoding techniques. Use wh... |
| stable-diffusion-image-generation | `.agents/skills/library/stable-diffusion/SKILL.md` | State-of-the-art text-to-image generation with Stable Diffusion models via HuggingFace Diffusers. Use when generating... |
| tensorboard | `.agents/skills/library/tensorboard/SKILL.md` | Visualize training metrics, debug models with histograms, compare experiments, visualize model graphs, and profile pe... |
| tensorrt-llm | `.agents/skills/library/tensorrt-llm/SKILL.md` | Optimizes LLM inference with NVIDIA TensorRT for maximum throughput and lowest latency. Use for production deployment... |
| torchforge-rl-training | `.agents/skills/library/torchforge/SKILL.md` | Provides guidance for PyTorch-native agentic RL using torchforge, Meta's library separating infra from algorithms. Us... |
| distributed-llm-pretraining-torchtitan | `.agents/skills/library/torchtitan/SKILL.md` | Provides PyTorch-native distributed LLM pretraining using torchtitan with 4D parallelism (FSDP2, TP, PP, CP). Use whe... |
| transformer-lens-interpretability | `.agents/skills/library/transformer-lens/SKILL.md` | Provides guidance for mechanistic interpretability research using TransformerLens to inspect and manipulate transform... |
| fine-tuning-with-trl | `.agents/skills/library/trl-fine-tuning/SKILL.md` | Fine-tune LLMs using reinforcement learning with TRL - SFT for instruction tuning, DPO for preference alignment, PPO/... |
| unsloth | `.agents/skills/library/unsloth/SKILL.md` | Expert guidance for fast fine-tuning with Unsloth - 2-5x faster training, 50-80% less memory, LoRA/QLoRA optimization |
| verl-rl-training | `.agents/skills/library/verl/SKILL.md` | Provides guidance for training LLMs with reinforcement learning using verl (Volcano Engine RL). Use when implementing... |
| serving-llms-vllm | `.agents/skills/library/vllm/SKILL.md` | Serves LLMs with high throughput using vLLM's PagedAttention and continuous batching. Use when deploying production L... |
| weights-and-biases | `.agents/skills/library/weights-and-biases/SKILL.md` | Track ML experiments with automatic logging, visualize training in real-time, optimize hyperparameters with sweeps, a... |
| whisper | `.agents/skills/library/whisper/SKILL.md` | OpenAI's general-purpose speech recognition model. Supports 99 languages, transcription, translation to English, and ... |
