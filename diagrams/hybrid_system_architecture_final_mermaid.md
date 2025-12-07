# Hybrid Hallucination Detection System for LLMs

## Clean Left-to-Right Architecture Diagram

```mermaid
graph LR
    Input[User Prompt +<br/>LLM Generated Response] --> Preprocess[Text Preprocessing<br/>Cleaning, Tokenization]
    
    Preprocess --> Transformer[Transformer-based<br/>Classifier<br/>DistilBERT]
    
    Transformer --> Entity[Entity Verification<br/>NER + Wikipedia]
    
    Transformer -.->|Optional| Agentic[Agentic Verification<br/>LLM Cross-Check<br/>Optional]
    
    Entity --> Uncertainty[Uncertainty-Driven<br/>Scoring]
    
    Agentic -.->|Optional| Uncertainty
    
    Uncertainty --> Fusion[Hybrid Fusion Engine]
    
    Fusion --> Output[Final Output<br/>Hallucination Probability<br/>+ Decision Label]
    
    style Input fill:#808080,stroke:#000,stroke-width:2px
    style Preprocess fill:#e0e0e0,stroke:#000,stroke-width:1px
    style Transformer fill:#e0e0e0,stroke:#000,stroke-width:1px
    style Entity fill:#e0e0e0,stroke:#000,stroke-width:1px
    style Agentic fill:#e0e0e0,stroke:#666,stroke-width:1px,stroke-dasharray: 5 5
    style Uncertainty fill:#e0e0e0,stroke:#000,stroke-width:1px
    style Fusion fill:#e0e0e0,stroke:#000,stroke-width:1px
    style Output fill:#808080,stroke:#000,stroke-width:2px
```

## Component Flow (Left to Right)

1. **Input** (Rectangle, darker background)
   - User Prompt + LLM Generated Response

2. **Preprocessing** (Rounded rectangle)
   - Text Preprocessing (Cleaning, Tokenization)

3. **Transformer-based Classifier** (Primary block, larger rectangle)
   - Transformer-based Classifier (DistilBERT)
   - **Splits into two branches:**

4. **Entity Verification** (Top branch, solid line)
   - Entity Verification (NER + Wikipedia)

5. **Agentic Verification** (Bottom branch, optional, dashed)
   - Agentic Verification (LLM Cross-Check – Optional)

6. **Uncertainty-Driven Scoring** (Convergence point)
   - Uncertainty-Driven Scoring
   - Both branches converge here

7. **Hybrid Fusion Engine** (Hexagon)
   - Hybrid Fusion Engine

8. **Final Output** (Rectangle, darker background)
   - Final Output
   - Hallucination Probability
   - + Decision Label

## Styling Specifications

- **Layout**: Left-to-right flow
- **Colors**: Grayscale/neutral only
- **Font**: Sans-serif, medium size
- **Main Flow**: Solid arrows
- **Optional Path**: Dashed arrows and borders
- **Style**: Professional, research-paper quality
- **Alignment**: Balanced and evenly distributed

