"""
Example usage of the entity verification module.
Demonstrates how to use the module for hybrid hallucination detection.
"""

from entity_verification import EntityVerifier, calculate_hybrid_score


def main():
    """Example usage of entity verification."""
    print("=" * 60)
    print("Entity Verification Example")
    print("=" * 60)
    
    # Initialize the verifier
    # Note: First time, you may need to download spaCy model:
    # python -m spacy download en_core_web_sm
    print("\nInitializing entity verifier...")
    try:
        verifier = EntityVerifier(
            extractor_method="spacy",  # or "transformers"
            use_wikipedia=True,
            rate_limit_delay=0.2  # Be respectful with API rate limiting
        )
        print("✓ Verifier initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing verifier: {e}")
        print("  Make sure spaCy is installed: pip install spacy")
        print("  Then download model: python -m spacy download en_core_web_sm")
        return
    
    # Example responses to verify
    responses = [
        # Response 1: Likely correct (verifiable entities)
        "Barack Obama was the 44th President of the United States. He was born in Hawaii in 1961.",
        
        # Response 2: Likely incorrect (fictional entities)
        "Dr. Quantum invented the time machine in 2025 at the Institute of Impossible Science.",
        
        # Response 3: Mixed (some verifiable, some not)
        "Albert Einstein developed the theory of relativity at Princeton University. He collaborated with Dr. Imaginary Scientist.",
        
        # Response 4: No specific entities
        "This is a general statement about concepts and ideas without specific named entities."
    ]
    
    print("\n" + "=" * 60)
    print("Verifying Responses")
    print("=" * 60)
    
    results = []
    for i, response in enumerate(responses, 1):
        print(f"\n--- Response {i} ---")
        print(f"Text: {response[:80]}...")
        
        # Verify the response
        result = verifier.verify_response(response, min_confidence=0.5)
        results.append(result)
        
        # Display results
        print(f"\nEntities found: {result.total_count}")
        if result.entities:
            print("Entity details:")
            for entity in result.entities[:5]:  # Show first 5
                status = "✓" if entity.verified else "✗"
                conf = f" (conf: {entity.confidence:.2f})" if entity.confidence else ""
                print(f"  {status} {entity.text} [{entity.label}]{conf}")
        
        print(f"\nVerification Summary:")
        print(f"  Verified entities: {result.verified_count}/{result.total_count}")
        print(f"  Correctness score: {result.correctness_score:.3f}")
        print(f"  Interpretation: {'HIGH' if result.correctness_score > 0.7 else 'MEDIUM' if result.correctness_score > 0.4 else 'LOW'} confidence")
    
    # Example: Hybrid scoring with model predictions
    print("\n" + "=" * 60)
    print("Hybrid Scoring Example")
    print("=" * 60)
    
    # Simulate model predictions (in real usage, these come from your trained model)
    model_predictions = [0.2, 0.8, 0.5, 0.3]  # Hallucination probabilities
    
    print("\nCombining model predictions with entity verification:")
    for i, (result, model_pred) in enumerate(zip(results, model_predictions), 1):
        # Model prediction: 0 = correct, 1 = hallucination
        # Convert to correctness score: 1 - model_pred
        model_correctness = 1.0 - model_pred
        
        # Calculate hybrid score
        hybrid_hallucination_prob = calculate_hybrid_score(
            model_score=model_pred,  # Model's hallucination probability
            entity_score=result.correctness_score,  # Entity verification score
            weight_model=0.7,  # 70% weight on model
            weight_entity=0.3   # 30% weight on entity verification
        )
        
        print(f"\nResponse {i}:")
        print(f"  Model prediction (hallucination prob): {model_pred:.3f}")
        print(f"  Entity verification score: {result.correctness_score:.3f}")
        print(f"  Hybrid hallucination probability: {hybrid_hallucination_prob:.3f}")
        print(f"  Final decision: {'HALLUCINATION' if hybrid_hallucination_prob > 0.5 else 'CORRECT'}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

