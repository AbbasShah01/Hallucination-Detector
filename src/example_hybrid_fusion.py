"""
Example: Using Hybrid Fusion with Entity Verification
Demonstrates the complete pipeline from entity verification to hybrid fusion.
"""

from hybrid_fusion import hybrid_predict, batch_predict, FusionResult
from entity_verification import EntityVerifier

# Note: This is a simplified example. In production, you would:
# 1. Load your trained transformer model
# 2. Get predictions from the model
# 3. Get factual scores from entity verification
# 4. Combine them using hybrid fusion


def simulate_transformer_prediction(text: str) -> float:
    """
    Simulate transformer model prediction.
    In production, this would call your actual trained model.
    
    Args:
        text: Input text
    
    Returns:
        Hallucination probability (0-1)
    """
    # This is a placeholder - replace with actual model inference
    # Example: model.predict(text) -> returns hallucination probability
    
    # Simple heuristic for demonstration
    suspicious_keywords = ["invented", "discovered", "according to", "fictional"]
    if any(keyword in text.lower() for keyword in suspicious_keywords):
        return 0.75  # Higher hallucination probability
    else:
        return 0.25  # Lower hallucination probability


def complete_hybrid_pipeline(
    response: str,
    verifier: EntityVerifier,
    alpha: float = 0.7,
    threshold: float = 0.5
) -> FusionResult:
    """
    Complete hybrid pipeline: transformer + entity verification + fusion.
    
    Args:
        response: LLM response to evaluate
        verifier: EntityVerifier instance
        alpha: Weight parameter for fusion
        threshold: Classification threshold
    
    Returns:
        FusionResult with final prediction
    """
    # Step 1: Get transformer model prediction
    transformer_prob = simulate_transformer_prediction(response)
    
    # Step 2: Get factual score from entity verification
    verification_result = verifier.verify_response(response, min_confidence=0.5)
    factual_score = verification_result.correctness_score
    
    # Step 3: Combine using hybrid fusion
    fusion_result = hybrid_predict(
        transformer_prob=transformer_prob,
        factual_score=factual_score,
        alpha=alpha,
        threshold=threshold
    )
    
    return fusion_result


def main():
    """Main example function."""
    print("=" * 70)
    print("Complete Hybrid Fusion Pipeline Example")
    print("=" * 70)
    
    # Initialize entity verifier (without Wikipedia for faster demo)
    print("\nInitializing entity verifier...")
    try:
        verifier = EntityVerifier(
            extractor_method="spacy",
            use_wikipedia=False  # Set to True for real verification
        )
        print("✓ Verifier initialized")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Note: This example requires spaCy. Install with: pip install spacy")
        print("  Then download model: python -m spacy download en_core_web_sm")
        return
    
    # Sample responses
    responses = [
        "Barack Obama was the 44th President of the United States.",
        "Dr. Quantum invented the time machine in 2025.",
        "Albert Einstein developed the theory of relativity at Princeton.",
        "The moon is made of cheese according to NASA scientists.",
        "Microsoft is a technology company founded in 1975."
    ]
    
    print(f"\nEvaluating {len(responses)} responses...")
    print("=" * 70)
    
    # Process each response
    for i, response in enumerate(responses, 1):
        print(f"\n--- Response {i} ---")
        print(f"Text: {response}")
        
        # Run complete pipeline
        result = complete_hybrid_pipeline(
            response=response,
            verifier=verifier,
            alpha=0.7,
            threshold=0.5
        )
        
        # Display results
        print(f"\nTransformer prediction: {result.transformer_prob:.3f}")
        print(f"Entity verification:     {result.factual_score:.3f}")
        print(f"Fused probability:      {result.fusion_prob:.3f}")
        print(f"Classification:         {'HALLUCINATION' if result.is_hallucination else 'CORRECT'}")
        print(f"Confidence:              {result.confidence:.3f}")
    
    # Batch processing example
    print("\n" + "=" * 70)
    print("Batch Processing Example")
    print("=" * 70)
    
    # Simulate batch predictions
    transformer_probs = [simulate_transformer_prediction(r) for r in responses]
    factual_scores = [
        verifier.verify_response(r, min_confidence=0.5).correctness_score
        for r in responses
    ]
    
    batch_results = batch_predict(
        transformer_probs=transformer_probs,
        factual_scores=factual_scores,
        alpha=0.7,
        threshold=0.5
    )
    
    print(f"\nBatch results for {len(batch_results)} responses:")
    print("-" * 70)
    for i, (response, result) in enumerate(zip(responses, batch_results), 1):
        status = "HALLUCINATION" if result.is_hallucination else "CORRECT"
        print(f"{i}. {status:12s} | Fusion: {result.fusion_prob:.3f} | "
              f"Trans: {result.transformer_prob:.3f} | Factual: {result.factual_score:.3f}")
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

