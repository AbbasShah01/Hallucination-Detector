"""
Agentic Verification Module for Hybrid Hallucination Detection
Uses a small LLM (local or API-based) to cross-check LLM responses
and provide verification scores for integration with hybrid fusion.
"""

import os
import json
import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import time

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class VerificationResult:
    """Result of agentic verification."""
    response: str
    verification_score: float  # 0-1 score (higher = more likely correct)
    reasoning: Optional[str] = None  # LLM's reasoning for the score
    method: str = "unknown"  # Verification method used


class LocalLLMVerifier:
    """
    Uses a local small LLM for verification.
    Supports models from HuggingFace transformers library.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """
        Initialize local LLM verifier.
        
        Args:
            model_name: HuggingFace model name for verification
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers not available. Install with: pip install transformers"
            )
        
        self.model_name = model_name
        print(f"Loading local LLM: {model_name}...")
        
        try:
            # Use text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_length=512,
                device=-1  # CPU, change to 0 for GPU if available
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("✓ Local LLM loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load {model_name}: {e}")
            print("Falling back to simple heuristic-based verification")
            self.pipeline = None
            self.tokenizer = None
    
    def verify(self, response: str, context: Optional[str] = None) -> VerificationResult:
        """
        Verify a response using local LLM.
        
        Args:
            response: Response to verify
            context: Optional context/prompt that generated the response
        
        Returns:
            VerificationResult with score and reasoning
        """
        if self.pipeline is None:
            # Fallback to heuristic
            return self._heuristic_verification(response)
        
        # Create verification prompt
        prompt = self._create_verification_prompt(response, context)
        
        try:
            # Generate verification response
            result = self.pipeline(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True
            )
            
            generated_text = result[0]['generated_text']
            reasoning = generated_text[len(prompt):].strip()
            
            # Extract score from reasoning
            score = self._extract_score_from_text(reasoning)
            
            return VerificationResult(
                response=response,
                verification_score=score,
                reasoning=reasoning,
                method=f"local_llm_{self.model_name}"
            )
        except Exception as e:
            print(f"Error in LLM verification: {e}")
            return self._heuristic_verification(response)
    
    def _create_verification_prompt(self, response: str, context: Optional[str] = None) -> str:
        """Create prompt for LLM verification."""
        prompt = """You are a fact-checker. Evaluate if the following response is factually correct or contains hallucinations.

Response to verify: "{response}"

Provide a score from 0.0 to 1.0 where:
- 1.0 = Completely factually correct
- 0.0 = Contains significant hallucinations or false information
- 0.5 = Uncertain or partially correct

Format your response as: "Score: X.XX. Reasoning: [your reasoning]"

Verification:""".format(response=response[:500])  # Limit length
        
        return prompt
    
    def _extract_score_from_text(self, text: str) -> float:
        """Extract numerical score from LLM response."""
        # Look for "Score: X.XX" pattern
        score_pattern = r'Score:\s*([0-9]*\.?[0-9]+)'
        match = re.search(score_pattern, text, re.IGNORECASE)
        
        if match:
            score = float(match.group(1))
            # Normalize to 0-1 range
            return max(0.0, min(1.0, score))
        
        # Look for percentage
        percent_pattern = r'([0-9]+)%'
        match = re.search(percent_pattern, text)
        if match:
            return float(match.group(1)) / 100.0
        
        # Look for words indicating correctness
        text_lower = text.lower()
        if any(word in text_lower for word in ['correct', 'accurate', 'true', 'valid']):
            if any(word in text_lower for word in ['very', 'highly', 'completely']):
                return 0.9
            return 0.7
        elif any(word in text_lower for word in ['incorrect', 'false', 'hallucination', 'wrong']):
            if any(word in text_lower for word in ['completely', 'entirely', 'totally']):
                return 0.1
            return 0.3
        else:
            return 0.5  # Neutral/uncertain
    
    def _heuristic_verification(self, response: str) -> VerificationResult:
        """Fallback heuristic-based verification."""
        # Simple heuristics
        suspicious_phrases = [
            'according to', 'some say', 'it is believed', 'rumor has it',
            'supposedly', 'allegedly', 'reportedly'
        ]
        
        confidence_phrases = [
            'fact', 'proven', 'verified', 'confirmed', 'established',
            'research shows', 'studies indicate'
        ]
        
        score = 0.5  # Start neutral
        
        response_lower = response.lower()
        
        # Adjust based on phrases
        if any(phrase in response_lower for phrase in suspicious_phrases):
            score -= 0.2
        if any(phrase in response_lower for phrase in confidence_phrases):
            score += 0.2
        
        # Normalize
        score = max(0.0, min(1.0, score))
        
        return VerificationResult(
            response=response,
            verification_score=score,
            reasoning="Heuristic-based verification (LLM unavailable)",
            method="heuristic"
        )


class APIVerifier:
    """
    Uses API-based LLM for verification (OpenAI, Anthropic, etc.).
    """
    
    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize API verifier.
        
        Args:
            provider: "openai" or "anthropic"
            model: Model name to use
            api_key: API key (or set environment variable)
        """
        self.provider = provider.lower()
        self.model = model
        
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not available. Install with: pip install openai")
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            openai.api_key = self.api_key
            print("✓ OpenAI API verifier initialized")
        
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic not available. Install with: pip install anthropic")
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print("✓ Anthropic API verifier initialized")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'")
    
    def verify(self, response: str, context: Optional[str] = None) -> VerificationResult:
        """
        Verify a response using API-based LLM.
        
        Args:
            response: Response to verify
            context: Optional context/prompt
        
        Returns:
            VerificationResult with score and reasoning
        """
        prompt = self._create_verification_prompt(response, context)
        
        try:
            if self.provider == "openai":
                return self._verify_openai(prompt, response)
            elif self.provider == "anthropic":
                return self._verify_anthropic(prompt, response)
        except Exception as e:
            print(f"Error in API verification: {e}")
            # Return neutral score on error
            return VerificationResult(
                response=response,
                verification_score=0.5,
                reasoning=f"API error: {str(e)}",
                method=f"api_{self.provider}_error"
            )
    
    def _verify_openai(self, prompt: str, response: str) -> VerificationResult:
        """Verify using OpenAI API."""
        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a fact-checker. Provide scores from 0.0 to 1.0."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            reasoning = completion.choices[0].message.content
            score = self._extract_score_from_text(reasoning)
            
            return VerificationResult(
                response=response,
                verification_score=score,
                reasoning=reasoning,
                method=f"openai_{self.model}"
            )
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def _verify_anthropic(self, prompt: str, response: str) -> VerificationResult:
        """Verify using Anthropic API."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            reasoning = message.content[0].text
            score = self._extract_score_from_text(reasoning)
            
            return VerificationResult(
                response=response,
                verification_score=score,
                reasoning=reasoning,
                method=f"anthropic_{self.model}"
            )
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")
    
    def _create_verification_prompt(self, response: str, context: Optional[str] = None) -> str:
        """Create verification prompt."""
        prompt = f"""Evaluate if the following response is factually correct or contains hallucinations.

Response to verify: "{response[:500]}"

Provide a score from 0.0 to 1.0 where:
- 1.0 = Completely factually correct
- 0.0 = Contains significant hallucinations or false information
- 0.5 = Uncertain or partially correct

Format your response as: "Score: X.XX. Reasoning: [your reasoning]"

Verification:"""
        return prompt
    
    def _extract_score_from_text(self, text: str) -> float:
        """Extract numerical score from API response."""
        # Look for "Score: X.XX" pattern
        score_pattern = r'Score:\s*([0-9]*\.?[0-9]+)'
        match = re.search(score_pattern, text, re.IGNORECASE)
        
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        
        # Look for percentage
        percent_pattern = r'([0-9]+)%'
        match = re.search(percent_pattern, text)
        if match:
            return float(match.group(1)) / 100.0
        
        # Default to neutral
        return 0.5


class AgenticVerifier:
    """
    Main agentic verification class that can use different backends.
    """
    
    def __init__(
        self,
        method: str = "local",
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize agentic verifier.
        
        Args:
            method: "local" or "api"
            model_name: Model name (for local) or API model name
            provider: "openai" or "anthropic" (for API method)
            api_key: API key (optional, can use environment variable)
        """
        self.method = method.lower()
        
        if self.method == "local":
            model = model_name or "microsoft/DialoGPT-small"
            self.verifier = LocalLLMVerifier(model_name=model)
        elif self.method == "api":
            if not provider:
                raise ValueError("provider required for API method")
            self.verifier = APIVerifier(
                provider=provider,
                model=model_name or ("gpt-3.5-turbo" if provider == "openai" else "claude-3-haiku-20240307"),
                api_key=api_key
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'local' or 'api'")
    
    def verify(self, response: str, context: Optional[str] = None) -> VerificationResult:
        """
        Verify a response and return score.
        
        Args:
            response: Response to verify
            context: Optional context/prompt
        
        Returns:
            VerificationResult with 0-1 score
        """
        return self.verifier.verify(response, context)
    
    def verify_batch(self, responses: List[str]) -> List[VerificationResult]:
        """Verify multiple responses."""
        results = []
        for response in responses:
            result = self.verify(response)
            results.append(result)
            time.sleep(0.1)  # Rate limiting
        return results


def integrate_with_hybrid_fusion(
    transformer_prob: float,
    factual_score: float,
    agentic_score: float,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2
) -> float:
    """
    Integrate agentic verification with hybrid fusion.
    
    Three-way fusion:
    - transformer_prob: Transformer model's hallucination probability
    - factual_score: Entity verification score (0-1, higher = more correct)
    - agentic_score: Agentic LLM verification score (0-1, higher = more correct)
    
    Weights:
    - alpha: Weight for transformer (default 0.5)
    - beta: Weight for factual/entity verification (default 0.3)
    - gamma: Weight for agentic verification (default 0.2)
    
    Args:
        transformer_prob: Transformer hallucination probability (0-1)
        factual_score: Entity verification score (0-1)
        agentic_score: Agentic verification score (0-1)
        alpha: Weight for transformer
        beta: Weight for factual
        gamma: Weight for agentic
    
    Returns:
        Final fused hallucination probability (0-1)
    """
    # Normalize weights
    total_weight = alpha + beta + gamma
    alpha = alpha / total_weight
    beta = beta / total_weight
    gamma = gamma / total_weight
    
    # Convert scores to hallucination probabilities
    factual_hallucination = 1.0 - factual_score
    agentic_hallucination = 1.0 - agentic_score
    
    # Three-way weighted fusion
    fusion_prob = (
        alpha * transformer_prob +
        beta * factual_hallucination +
        gamma * agentic_hallucination
    )
    
    return max(0.0, min(1.0, fusion_prob))


# Example usage and demonstrations
if __name__ == "__main__":
    print("=" * 70)
    print("Agentic Verification Module - Demonstration")
    print("=" * 70)
    
    # Initialize verifier (using local method for demo)
    print("\nInitializing agentic verifier...")
    try:
        verifier = AgenticVerifier(method="local")
    except Exception as e:
        print(f"Error initializing verifier: {e}")
        print("Note: For API-based verification, set API keys in environment variables")
        verifier = None
    
    # Sample responses to verify
    sample_responses = [
        {
            "text": "Barack Obama was the 44th President of the United States, serving from 2009 to 2017.",
            "transformer_prob": 0.15,  # Model thinks: low hallucination
            "factual_score": 0.95,     # Entity verification: high correctness
            "expected": "correct"
        },
        {
            "text": "Dr. Quantum invented the time machine in 2025 at the Institute of Impossible Science.",
            "transformer_prob": 0.85,  # Model thinks: high hallucination
            "factual_score": 0.20,     # Entity verification: low correctness
            "expected": "hallucination"
        },
        {
            "text": "The moon is made of cheese according to NASA scientists who published this in 2024.",
            "transformer_prob": 0.70,  # Model thinks: medium-high hallucination
            "factual_score": 0.30,     # Entity verification: low correctness
            "expected": "hallucination"
        },
        {
            "text": "Albert Einstein developed the theory of relativity, which revolutionized physics.",
            "transformer_prob": 0.25,   # Model thinks: low hallucination
            "factual_score": 0.90,     # Entity verification: high correctness
            "expected": "correct"
        },
        {
            "text": "Water boils at 100 degrees Celsius at sea level under standard atmospheric pressure.",
            "transformer_prob": 0.10,   # Model thinks: very low hallucination
            "factual_score": 0.98,      # Entity verification: very high correctness
            "expected": "correct"
        }
    ]
    
    if verifier:
        print("\n" + "=" * 70)
        print("Agentic Verification Results")
        print("=" * 70)
        
        results_with_agentic = []
        
        for i, sample in enumerate(sample_responses, 1):
            print(f"\n--- Response {i} ---")
            print(f"Text: {sample['text']}")
            
            # Get agentic verification
            agentic_result = verifier.verify(sample['text'])
            agentic_score = agentic_result.verification_score
            
            print(f"\nTransformer prob: {sample['transformer_prob']:.3f}")
            print(f"Factual score:    {sample['factual_score']:.3f}")
            print(f"Agentic score:    {agentic_score:.3f}")
            if agentic_result.reasoning:
                print(f"Agentic reasoning: {agentic_result.reasoning[:100]}...")
            
            # Compare: without agentic vs with agentic
            # Without agentic (original hybrid fusion)
            from hybrid_fusion import hybrid_predict as hybrid_predict_2way
            result_2way = hybrid_predict_2way(
                transformer_prob=sample['transformer_prob'],
                factual_score=sample['factual_score'],
                alpha=0.7,
                threshold=0.5
            )
            
            # With agentic (three-way fusion)
            fusion_prob_3way = integrate_with_hybrid_fusion(
                transformer_prob=sample['transformer_prob'],
                factual_score=sample['factual_score'],
                agentic_score=agentic_score,
                alpha=0.5,
                beta=0.3,
                gamma=0.2
            )
            is_hallucination_3way = fusion_prob_3way >= 0.5
            
            print(f"\nWithout agentic:")
            print(f"  Fusion prob: {result_2way.fusion_prob:.3f}")
            print(f"  Classification: {'HALLUCINATION' if result_2way.is_hallucination else 'CORRECT'}")
            
            print(f"\nWith agentic:")
            print(f"  Fusion prob: {fusion_prob_3way:.3f}")
            print(f"  Classification: {'HALLUCINATION' if is_hallucination_3way else 'CORRECT'}")
            
            # Check if agentic improved the prediction
            correct_2way = (result_2way.is_hallucination and sample['expected'] == 'hallucination') or \
                          (not result_2way.is_hallucination and sample['expected'] == 'correct')
            correct_3way = (is_hallucination_3way and sample['expected'] == 'hallucination') or \
                          (not is_hallucination_3way and sample['expected'] == 'correct')
            
            if correct_3way and not correct_2way:
                print(f"  ✓ Agentic verification IMPROVED prediction!")
            elif not correct_3way and correct_2way:
                print(f"  ✗ Agentic verification worsened prediction")
            else:
                print(f"  → Agentic verification maintained prediction")
            
            results_with_agentic.append({
                'response': sample['text'],
                'without_agentic': result_2way.fusion_prob,
                'with_agentic': fusion_prob_3way,
                'improved': correct_3way and not correct_2way
            })
        
        # Summary
        print("\n" + "=" * 70)
        print("Summary: Agentic Verification Impact")
        print("=" * 70)
        improvements = sum(1 for r in results_with_agentic if r['improved'])
        print(f"\nCases where agentic verification improved predictions: {improvements}/{len(results_with_agentic)}")
        
    else:
        print("\nSkipping agentic verification (verifier not available)")
        print("To use agentic verification:")
        print("  1. For local: Install transformers and download a model")
        print("  2. For API: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
    
    print("\n" + "=" * 70)
    print("Demonstration completed!")
    print("=" * 70)

