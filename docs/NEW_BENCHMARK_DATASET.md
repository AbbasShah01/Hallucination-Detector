# New Benchmark Dataset Proposal: HaluBench-Multi

## Executive Summary

**HaluBench-Multi** is a comprehensive, multi-dimensional benchmark dataset for hallucination detection that addresses critical gaps in existing datasets through:
- Multi-turn conversational hallucinations
- Fine-grained hallucination typology
- Cross-domain coverage
- Temporal and causal reasoning errors
- Adversarial examples (subtle hallucinations)

## Analysis of Existing Datasets

### Current Dataset Landscape

#### HaluEval (Li et al., 2023)
**Strengths:**
- Large scale (10K+ examples)
- Multiple tasks (QA, summarization, dialogue)

**Gaps:**
- Binary labels only (hallucination/correct)
- Single-turn focus (no conversation context)
- Obvious errors (easy to detect)
- Limited domain coverage
- No temporal reasoning errors

#### TruthfulQA (Lin et al., 2021)
**Strengths:**
- Focus on truthfulness
- Multiple question types

**Gaps:**
- Question-answering only
- No fine-grained error types
- Limited to factual errors
- No multi-turn scenarios

#### FEVER (Thorne et al., 2018)
**Strengths:**
- Evidence-based verification
- Wikipedia-based

**Gaps:**
- Claim-level only (not full responses)
- No conversational context
- Limited to factual claims
- No temporal/causal errors

### Critical Gaps Identified

1. **No Multi-Turn Context**: All datasets test single responses
2. **Binary Classification Only**: No fine-grained typology
3. **Obvious Errors**: Easy to detect, not realistic
4. **Limited Domains**: Mostly general knowledge
5. **No Temporal Reasoning**: Missing time-based errors
6. **No Causal Errors**: Missing logical causality issues
7. **No Adversarial Examples**: Missing subtle, plausible errors
8. **No Cross-Lingual**: English-only
9. **No Uncertainty Labels**: Don't indicate when detection is hard
10. **No Root Cause Labels**: Don't explain why hallucinations occur

## Proposed Dataset: HaluBench-Multi

### Dataset Characteristics

- **Size**: 50,000 examples (train: 35K, val: 7.5K, test: 7.5K)
- **Format**: Multi-turn conversations with context
- **Granularity**: Fine-grained typology + binary labels
- **Domains**: 10 domains (medical, legal, technical, historical, etc.)
- **Difficulty Levels**: Easy, Medium, Hard, Adversarial
- **Languages**: English (primary), with multilingual extension planned

### Hallucination Categories

#### 1. **Factual Errors** (FACT)
- **Subtypes**:
  - Entity confusion (wrong person/place/thing)
  - Factual contradiction (contradicts known facts)
  - Numerical error (wrong numbers/dates)
  - Misattribution (wrong source/author)

**Example:**
```
Context: "Tell me about the first moon landing."
Response: "Neil Armstrong landed on the moon in 1972." 
Error: Wrong year (should be 1969)
Type: FACT-numerical_error
```

#### 2. **Temporal Errors** (TEMP)
- **Subtypes**:
  - Wrong time period
  - Temporal contradiction
  - Anachronism (future knowledge in past context)
  - Sequence error (wrong order of events)

**Example:**
```
Context: "What happened in 1995?"
Response: "The iPhone was released in 1995."
Error: iPhone released in 2007, not 1995
Type: TEMP-anachronism
```

#### 3. **Causal Errors** (CAUS)
- **Subtypes**:
  - False causation (A didn't cause B)
  - Reversed causation (effect before cause)
  - Missing cause (ignores actual cause)
  - Spurious correlation (correlation ≠ causation)

**Example:**
```
Context: "Why did the stock market crash in 2008?"
Response: "The stock market crashed because of high oil prices."
Error: Actual cause was subprime mortgage crisis
Type: CAUS-false_causation
```

#### 4. **Logical Inconsistencies** (LOGIC)
- **Subtypes**:
  - Self-contradiction (contradicts itself)
  - Cross-turn contradiction (contradicts earlier turn)
  - Logical fallacy (invalid reasoning)
  - Incomplete reasoning (missing steps)

**Example:**
```
Turn 1: "I've never been to Paris."
Turn 2: "When I visited Paris last year, I saw the Eiffel Tower."
Error: Contradicts previous statement
Type: LOGIC-cross_turn_contradiction
```

#### 5. **Entity Confusion** (ENTITY)
- **Subtypes**:
  - Person confusion (wrong person)
  - Place confusion (wrong location)
  - Organization confusion (wrong org)
  - Concept confusion (wrong concept)

**Example:**
```
Context: "Who wrote Romeo and Juliet?"
Response: "Charles Dickens wrote Romeo and Juliet."
Error: Wrong author (Shakespeare)
Type: ENTITY-person_confusion
```

#### 6. **Omission Errors** (OMIT)
- **Subtypes**:
  - Critical information missing
  - Incomplete answer
  - Missing qualifiers
  - Missing context

**Example:**
```
Context: "What are the side effects of this medication?"
Response: "This medication is safe and effective."
Error: Missing critical side effect information
Type: OMIT-critical_info_missing
```

#### 7. **Citation Errors** (CITE)
- **Subtypes**:
  - Wrong citation
  - Fabricated citation
  - Misattributed citation
  - Missing citation

**Example:**
```
Response: "According to a 2023 study in Nature, AI will replace all jobs by 2025."
Error: Fabricated citation (no such study exists)
Type: CITE-fabricated_citation
```

#### 8. **Adversarial (Subtle)** (ADV)
- **Subtypes**:
  - Plausible but false
  - Partially correct (mostly true, one error)
  - Domain-specific subtle error
  - Context-dependent error

**Example:**
```
Context: "What is the capital of Australia?"
Response: "Canberra is the capital of Australia, with a population of 2.5 million."
Error: Population is wrong (actually ~400K), but fact is correct
Type: ADV-partially_correct
```

### Dataset Schema

```json
{
  "dataset_version": "1.0",
  "example_id": "halu_001234",
  "conversation": {
    "turns": [
      {
        "turn_id": 0,
        "role": "user",
        "text": "What happened in 1995?",
        "timestamp": null
      },
      {
        "turn_id": 1,
        "role": "assistant",
        "text": "The iPhone was released in 1995.",
        "timestamp": null
      }
    ],
    "context": {
      "domain": "technology",
      "topic": "product_releases",
      "source": "synthetic"
    }
  },
  "hallucination_annotation": {
    "has_hallucination": true,
    "binary_label": 1,
    "hallucination_type": "TEMP",
    "hallucination_subtype": "anachronism",
    "severity": "high",
    "confidence": 0.95,
    "affected_spans": [
      {
        "turn_id": 1,
        "start_char": 0,
        "end_char": 40,
        "text": "The iPhone was released in 1995.",
        "error_text": "1995",
        "correct_text": "2007"
      }
    ],
    "root_cause": [
      "temporal_knowledge_gap",
      "model_training_cutoff"
    ],
    "detection_difficulty": "medium",
    "requires_reasoning": true,
    "requires_domain_knowledge": true
  },
  "metadata": {
    "generation_method": "llm_synthetic",
    "base_model": "gpt-4",
    "verification_method": "human_annotated",
    "annotator_id": "annotator_001",
    "annotation_date": "2024-11-29",
    "quality_score": 0.92
  }
}
```

### CSV Schema (Alternative)

```csv
example_id,conversation_id,turn_id,role,text,has_hallucination,binary_label,hallucination_type,hallucination_subtype,severity,confidence,affected_span_start,affected_span_end,error_text,correct_text,root_cause,detection_difficulty,domain,topic,generation_method,quality_score
halu_001234,conv_001,0,user,"What happened in 1995?",0,0,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,technology,product_releases,synthetic,0.95
halu_001234,conv_001,1,assistant,"The iPhone was released in 1995.",1,1,TEMP,anachronism,high,0.95,0,40,"1995","2007","temporal_knowledge_gap|model_training_cutoff",medium,technology,product_releases,llm_synthetic,0.92
```

## Synthetic Generation Strategy

### Generation Pipeline

```
┌─────────────────────────────────────────────────────────┐
│         HaluBench-Multi Generation Pipeline             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  1. Template Selection              │
        │     • Choose hallucination type     │
        │     • Select domain                 │
        │     • Pick difficulty level         │
        └─────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  2. Context Generation              │
        │     • Generate realistic prompt     │
        │     • Create conversation history   │
        │     • Add domain-specific context   │
        └─────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  3. Hallucination Injection         │
        │     • Generate response with error  │
        │     • Ensure error is realistic     │
        │     • Match difficulty level        │
        └─────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  4. Verification & Filtering        │
        │     • Verify error is correct       │
        │     • Check plausibility            │
        │     • Filter low-quality examples   │
        └─────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │  5. Annotation                     │
        │     • Label hallucination type      │
        │     • Mark affected spans           │
        │     • Identify root causes          │
        └─────────────────────────────────────┘
```

### Generation Methods

#### Method 1: LLM-Based Generation
- Use GPT-4/Claude to generate examples
- Provide detailed prompts for each hallucination type
- Use few-shot examples to guide generation

#### Method 2: Template-Based Generation
- Create templates for each error type
- Fill templates with real entities/facts
- Introduce controlled errors

#### Method 3: Adversarial Generation
- Start with correct responses
- Use adversarial prompts to introduce subtle errors
- Ensure errors are plausible

#### Method 4: Human-in-the-Loop
- LLM generates candidates
- Humans verify and refine
- Ensures high quality

## Quality Assurance

### Automatic Checks
1. **Fact Verification**: Verify errors are actually errors
2. **Plausibility Check**: Ensure errors are realistic
3. **Type Consistency**: Verify labels match errors
4. **Span Validation**: Check affected spans are correct

### Human Validation
1. **Expert Review**: Domain experts verify examples
2. **Inter-Annotator Agreement**: Multiple annotators
3. **Quality Scoring**: Rate each example
4. **Error Analysis**: Identify systematic issues

## Dataset Statistics

### Distribution by Type
- Factual Errors: 25%
- Temporal Errors: 15%
- Causal Errors: 10%
- Logical Inconsistencies: 15%
- Entity Confusion: 15%
- Omission Errors: 10%
- Citation Errors: 5%
- Adversarial: 5%

### Distribution by Difficulty
- Easy: 30%
- Medium: 40%
- Hard: 20%
- Adversarial: 10%

### Distribution by Domain
- General Knowledge: 20%
- Medical: 10%
- Legal: 10%
- Technical: 15%
- Historical: 15%
- Scientific: 10%
- Business: 10%
- Cultural: 10%

## Usage Guidelines

### Training
- Use train split (35K examples)
- Can filter by type/difficulty/domain
- Supports few-shot learning

### Validation
- Use val split (7.5K examples)
- For hyperparameter tuning
- For model selection

### Testing
- Use test split (7.5K examples)
- Final evaluation only
- Report metrics per type/difficulty

## Evaluation Metrics

### Overall Metrics
- Accuracy, Precision, Recall, F1
- Per-type metrics
- Per-difficulty metrics
- Per-domain metrics

### Advanced Metrics
- Detection difficulty analysis
- Root cause prediction accuracy
- Span-level F1 (for affected spans)

## Comparison with Existing Datasets

| Feature | HaluEval | TruthfulQA | FEVER | **HaluBench-Multi** |
|---------|----------|------------|-------|---------------------|
| Size | 10K | 817 | 185K | **50K** |
| Multi-turn | ❌ | ❌ | ❌ | ✅ |
| Fine-grained types | ❌ | ❌ | ❌ | ✅ |
| Temporal errors | ❌ | ❌ | ❌ | ✅ |
| Causal errors | ❌ | ❌ | ❌ | ✅ |
| Adversarial examples | ❌ | ❌ | ❌ | ✅ |
| Root cause labels | ❌ | ❌ | ❌ | ✅ |
| Difficulty levels | ❌ | ❌ | ❌ | ✅ |
| Cross-domain | Limited | Limited | Limited | ✅ |

## Future Extensions

1. **Multilingual**: Extend to 10+ languages
2. **Real-time**: Streaming conversation data
3. **Multimodal**: Add images/videos
4. **Dynamic**: Continuously updated with new errors
5. **Crowdsourced**: Community contributions

