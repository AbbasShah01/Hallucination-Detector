# LaTeX Research Paper

This directory contains a publication-ready LaTeX research paper describing the Hybrid Hallucination Detection System.

## Files

- **main.tex** - Main LaTeX document (NeurIPS/ACL format)
- **references.bib** - Bibliography with citations
- **neurips_2023.sty** - NeurIPS style file (placeholder)
- **acl.sty** - ACL style file (placeholder)

## Compilation

### Using pdflatex

```bash
cd papers
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using Overleaf

1. Upload all files to Overleaf
2. Set main document to `main.tex`
3. Compile

## Paper Structure

1. **Abstract** - Summary with key results
2. **Introduction** - Background, problem statement, contributions
3. **Related Work** - Review of existing approaches
4. **Methodology** - System architecture, uncertainty-driven scoring, fusion
5. **Experiments** - Datasets, baselines, metrics
6. **Results** - Performance metrics, ablation studies, per-type analysis
7. **Discussion** - Key findings, error analysis
8. **Limitations** - Current constraints
9. **Future Work** - Research directions
10. **Conclusion** - Summary and impact

## Key Features

- ✅ NeurIPS/ACL format
- ✅ TikZ diagrams for architecture
- ✅ Mathematical formulations
- ✅ Tables with experimental results
- ✅ Figures with uncertainty analysis
- ✅ Complete bibliography
- ✅ Appendix with implementation details

## Note

The style files (`neurips_2023.sty`, `acl.sty`) are placeholders. For actual submission:
- Download official style files from conference websites
- Replace placeholders with official styles
- Adjust formatting as needed for specific venue

## Citation Format

If you use this work, please cite:

```bibtex
@article{shah2024hybrid,
  title={Uncertainty-Driven Hybrid Multi-Component Hallucination Detection for Large Language Models},
  author={Shah, Abbas},
  journal={arXiv preprint},
  year={2024}
}
```

