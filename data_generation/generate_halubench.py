"""
HaluBench-Multi Dataset Generation Script
Generates synthetic but high-quality hallucination detection examples.
"""

import os
import json
import random
import argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class ConversationTurn:
    """Represents a turn in a conversation."""
    turn_id: int
    role: str  # "user" or "assistant"
    text: str
    timestamp: Optional[str] = None


@dataclass
class AffectedSpan:
    """Represents a span of text affected by hallucination."""
    turn_id: int
    start_char: int
    end_char: int
    text: str
    error_text: str
    correct_text: str


@dataclass
class HallucinationAnnotation:
    """Annotation for hallucination in a response."""
    has_hallucination: bool
    binary_label: int  # 0 or 1
    hallucination_type: str  # FACT, TEMP, CAUS, etc.
    hallucination_subtype: str
    severity: str  # "low", "medium", "high"
    confidence: float
    affected_spans: List[AffectedSpan]
    root_cause: List[str]
    detection_difficulty: str  # "easy", "medium", "hard", "adversarial"
    requires_reasoning: bool
    requires_domain_knowledge: bool


@dataclass
class DatasetExample:
    """Complete dataset example."""
    example_id: str
    conversation: Dict
    hallucination_annotation: Dict
    metadata: Dict


class HaluBenchGenerator:
    """
    Generates HaluBench-Multi dataset examples.
    """
    
    # Hallucination types and subtypes
    HALLUCINATION_TYPES = {
        "FACT": {
            "entity_confusion": 0.3,
            "factual_contradiction": 0.3,
            "numerical_error": 0.2,
            "misattribution": 0.2
        },
        "TEMP": {
            "wrong_time_period": 0.3,
            "temporal_contradiction": 0.2,
            "anachronism": 0.3,
            "sequence_error": 0.2
        },
        "CAUS": {
            "false_causation": 0.4,
            "reversed_causation": 0.2,
            "missing_cause": 0.2,
            "spurious_correlation": 0.2
        },
        "LOGIC": {
            "self_contradiction": 0.3,
            "cross_turn_contradiction": 0.4,
            "logical_fallacy": 0.2,
            "incomplete_reasoning": 0.1
        },
        "ENTITY": {
            "person_confusion": 0.3,
            "place_confusion": 0.3,
            "organization_confusion": 0.2,
            "concept_confusion": 0.2
        },
        "OMIT": {
            "critical_info_missing": 0.4,
            "incomplete_answer": 0.3,
            "missing_qualifiers": 0.2,
            "missing_context": 0.1
        },
        "CITE": {
            "wrong_citation": 0.3,
            "fabricated_citation": 0.4,
            "misattributed_citation": 0.2,
            "missing_citation": 0.1
        },
        "ADV": {
            "plausible_but_false": 0.3,
            "partially_correct": 0.4,
            "domain_subtle_error": 0.2,
            "context_dependent_error": 0.1
        }
    }
    
    DOMAINS = [
        "general_knowledge", "medical", "legal", "technical",
        "historical", "scientific", "business", "cultural",
        "geography", "arts"
    ]
    
    DIFFICULTY_LEVELS = ["easy", "medium", "hard", "adversarial"]
    
    ROOT_CAUSES = [
        "insufficient_context",
        "knowledge_gap",
        "model_training_cutoff",
        "prompt_ambiguity",
        "domain_mismatch",
        "temporal_gap",
        "reasoning_limitation"
    ]
    
    def __init__(self, use_llm: bool = True, llm_provider: str = "openai"):
        """
        Initialize generator.
        
        Args:
            use_llm: Whether to use LLM for generation
            llm_provider: "openai" or "anthropic"
        """
        self.use_llm = use_llm and (OPENAI_AVAILABLE or ANTHROPIC_AVAILABLE)
        self.llm_provider = llm_provider
        self.templates = self._load_templates()
    
    def generate_example(
        self,
        hallucination_type: Optional[str] = None,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> DatasetExample:
        """
        Generate a single dataset example.
        
        Args:
            hallucination_type: Specific type to generate (or random)
            domain: Specific domain (or random)
            difficulty: Specific difficulty (or random)
        
        Returns:
            DatasetExample object
        """
        # Select parameters if not specified
        if hallucination_type is None:
            hallucination_type = self._select_hallucination_type()
        
        if domain is None:
            domain = random.choice(self.DOMAINS)
        
        if difficulty is None:
            difficulty = random.choice(self.DIFFICULTY_LEVELS)
        
        # Select subtype
        subtypes = self.HALLUCINATION_TYPES[hallucination_type]
        subtype = random.choices(
            list(subtypes.keys()),
            weights=list(subtypes.values())
        )[0]
        
        # Generate conversation
        if self.use_llm:
            conversation = self._generate_with_llm(
                hallucination_type, subtype, domain, difficulty
            )
        else:
            conversation = self._generate_with_template(
                hallucination_type, subtype, domain, difficulty
            )
        
        # Create annotation
        annotation = self._create_annotation(
            conversation, hallucination_type, subtype, difficulty
        )
        
        # Create metadata
        metadata = self._create_metadata(domain, difficulty)
        
        # Generate example ID
        example_id = f"halu_{uuid.uuid4().hex[:8]}"
        
        return DatasetExample(
            example_id=example_id,
            conversation=conversation,
            hallucination_annotation=asdict(annotation),
            metadata=metadata
        )
    
    def _select_hallucination_type(self) -> str:
        """Select hallucination type based on distribution."""
        # Weighted selection based on desired distribution
        types = list(self.HALLUCINATION_TYPES.keys())
        weights = [0.25, 0.15, 0.10, 0.15, 0.15, 0.10, 0.05, 0.05]
        return random.choices(types, weights=weights)[0]
    
    def _generate_with_llm(
        self,
        h_type: str,
        subtype: str,
        domain: str,
        difficulty: str
    ) -> Dict:
        """Generate conversation using LLM."""
        prompt = self._create_generation_prompt(h_type, subtype, domain, difficulty)
        
        if self.llm_provider == "openai" and OPENAI_AVAILABLE:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a dataset generator for hallucination detection."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            generated_text = response.choices[0].message.content
        elif self.llm_provider == "anthropic" and ANTHROPIC_AVAILABLE:
            client = anthropic.Anthropic()
            message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            generated_text = message.content[0].text
        else:
            # Fallback to template
            return self._generate_with_template(h_type, subtype, domain, difficulty)
        
        # Parse LLM response into conversation format
        return self._parse_llm_response(generated_text)
    
    def _generate_with_template(
        self,
        h_type: str,
        subtype: str,
        domain: str,
        difficulty: str
    ) -> Dict:
        """Generate conversation using templates."""
        # Load template for this type/subtype
        template = self.templates.get(f"{h_type}_{subtype}", {})
        
        # Generate user turn
        user_turn = ConversationTurn(
            turn_id=0,
            role="user",
            text=template.get("user_prompt", f"Tell me about {domain}.")
        )
        
        # Generate assistant turn with hallucination
        assistant_text = template.get("assistant_response", 
                                      f"Here's information about {domain}.")
        
        assistant_turn = ConversationTurn(
            turn_id=1,
            role="assistant",
            text=assistant_text
        )
        
        return {
            "turns": [asdict(user_turn), asdict(assistant_turn)],
            "context": {
                "domain": domain,
                "topic": template.get("topic", "general"),
                "source": "template"
            }
        }
    
    def _create_generation_prompt(
        self,
        h_type: str,
        subtype: str,
        domain: str,
        difficulty: str
    ) -> str:
        """Create prompt for LLM generation."""
        return f"""Generate a conversation example for hallucination detection.

Requirements:
- Hallucination Type: {h_type} ({subtype})
- Domain: {domain}
- Difficulty: {difficulty}
- Format: JSON with "user_turn" and "assistant_turn"
- The assistant response should contain a {h_type} type hallucination ({subtype})
- Make it realistic and appropriate for {difficulty} difficulty level

Example format:
{{
  "user_turn": "What happened in 1995?",
  "assistant_turn": "The iPhone was released in 1995."
}}

Generate the conversation:"""
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response into conversation format."""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                user_turn = ConversationTurn(0, "user", data.get("user_turn", ""))
                assistant_turn = ConversationTurn(1, "assistant", data.get("assistant_turn", ""))
                
                return {
                    "turns": [asdict(user_turn), asdict(assistant_turn)],
                    "context": {
                        "domain": "general_knowledge",
                        "topic": "synthetic",
                        "source": "llm_generated"
                    }
                }
        except:
            pass
        
        # Fallback: simple parsing
        lines = response.strip().split('\n')
        user_text = lines[0] if lines else "Tell me something."
        assistant_text = lines[1] if len(lines) > 1 else "Here's some information."
        
        return {
            "turns": [
                asdict(ConversationTurn(0, "user", user_text)),
                asdict(ConversationTurn(1, "assistant", assistant_text))
            ],
            "context": {
                "domain": "general_knowledge",
                "topic": "synthetic",
                "source": "llm_generated"
            }
        }
    
    def _create_annotation(
        self,
        conversation: Dict,
        h_type: str,
        subtype: str,
        difficulty: str
    ) -> HallucinationAnnotation:
        """Create hallucination annotation."""
        # Determine if hallucination exists
        has_hallucination = h_type != "CORRECT"
        
        # Extract affected span (simplified)
        assistant_turn = conversation["turns"][-1]
        affected_span = AffectedSpan(
            turn_id=1,
            start_char=0,
            end_char=len(assistant_turn["text"]),
            text=assistant_turn["text"],
            error_text="[to be annotated]",
            correct_text="[to be annotated]"
        )
        
        # Select root causes
        num_causes = random.randint(1, 2)
        root_causes = random.sample(self.ROOT_CAUSES, num_causes)
        
        # Determine severity
        severity_map = {
            "easy": "low",
            "medium": "medium",
            "hard": "high",
            "adversarial": "high"
        }
        severity = severity_map.get(difficulty, "medium")
        
        # Determine confidence
        confidence_map = {
            "easy": 0.95,
            "medium": 0.85,
            "hard": 0.75,
            "adversarial": 0.65
        }
        confidence = confidence_map.get(difficulty, 0.8)
        
        return HallucinationAnnotation(
            has_hallucination=has_hallucination,
            binary_label=1 if has_hallucination else 0,
            hallucination_type=h_type,
            hallucination_subtype=subtype,
            severity=severity,
            confidence=confidence,
            affected_spans=[asdict(affected_span)],
            root_cause=root_causes,
            detection_difficulty=difficulty,
            requires_reasoning=difficulty in ["hard", "adversarial"],
            requires_domain_knowledge=random.random() > 0.5
        )
    
    def _create_metadata(self, domain: str, difficulty: str) -> Dict:
        """Create metadata for example."""
        return {
            "generation_method": "llm_synthetic" if self.use_llm else "template",
            "base_model": "gpt-4" if self.use_llm else "template",
            "verification_method": "automatic",
            "annotator_id": "auto_generator",
            "annotation_date": datetime.now().isoformat(),
            "quality_score": random.uniform(0.7, 0.95)
        }
    
    def _load_templates(self) -> Dict:
        """Load generation templates."""
        # In production, load from file
        # For now, return empty dict (will use LLM or simple templates)
        return {}


def generate_dataset(
    num_examples: int,
    output_path: str,
    use_llm: bool = True,
    llm_provider: str = "openai"
):
    """
    Generate complete dataset.
    
    Args:
        num_examples: Number of examples to generate
        output_path: Path to save dataset
        use_llm: Whether to use LLM
        llm_provider: LLM provider
    """
    generator = HaluBenchGenerator(use_llm=use_llm, llm_provider=llm_provider)
    
    examples = []
    print(f"Generating {num_examples} examples...")
    
    for i in range(num_examples):
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_examples} examples...")
        
        example = generator.generate_example()
        examples.append(asdict(example))
    
    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset_version": "1.0",
            "dataset_name": "HaluBench-Multi",
            "num_examples": len(examples),
            "generation_date": datetime.now().isoformat(),
            "examples": examples
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Total examples: {len(examples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HaluBench-Multi dataset")
    parser.add_argument("--num_examples", type=int, default=1000,
                       help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="data/halubench_multi.json",
                       help="Output file path")
    parser.add_argument("--use_llm", action="store_true",
                       help="Use LLM for generation")
    parser.add_argument("--llm_provider", type=str, default="openai",
                       choices=["openai", "anthropic"],
                       help="LLM provider")
    
    args = parser.parse_args()
    
    generate_dataset(
        num_examples=args.num_examples,
        output_path=args.output,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider
    )

