#!/usr/bin/env python3
"""
Paper Analysis with Claude API

This script uses Claude to analyze papers found by PubMed/arXiv searches
and determine their relevance and contribution to textbook chapters.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

import anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperAnalyzer:
    """Analyze papers using Claude API."""
    
    def __init__(self, api_key: str):
        """
        Initialize paper analyzer.
        
        Args:
            api_key: Anthropic API key
        """
        self.client = anthropic.Anthropic(api_key=api_key.strip() if api_key else api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    
    def analyze_paper_relevance(
        self,
        paper: Dict[str, Any],
        chapter_id: str,
        chapter_description: str
    ) -> Dict[str, Any]:
        """
        Analyze if a paper is relevant to a chapter and extract key insights.
        
        Args:
            paper: Paper metadata dictionary
            chapter_id: Chapter identifier
            chapter_description: Description of chapter content
            
        Returns:
            Analysis results dictionary
        """
        prompt = f"""You are an expert in healthcare AI and machine learning, specializing in methods for underserved populations and health equity.

Analyze the following paper for relevance to a textbook chapter:

**Paper Information:**
Title: {paper['title']}
Authors: {', '.join(paper.get('authors', []))}
Journal: {paper['journal']}
Year: {paper['year']}
PMID: {paper.get('pmid', 'N/A')}
DOI: {paper.get('doi', 'N/A')}

Abstract:
{paper.get('abstract', 'No abstract available')}

**Chapter Context:**
Chapter ID: {chapter_id}
Chapter Description: {chapter_description}

**Analysis Task:**
Evaluate this paper's relevance and provide a structured analysis. Return your response as a JSON object with the following fields:

1. "relevance_score" (0-10): How relevant is this paper to the chapter? (0=not relevant, 10=highly relevant)
2. "is_highly_cited" (boolean): Does this appear to be a landmark/highly influential paper based on the authors, journal, and content?
3. "is_sota" (boolean): Does this paper present state-of-the-art methods or significant advances?
4. "has_equity_focus" (boolean): Does the paper explicitly address health disparities, bias, fairness, or underserved populations?
5. "key_contributions" (list of strings): What are the 2-3 most important contributions relevant to this chapter?
6. "methods_or_models" (list of strings): What specific methods, models, or techniques are introduced?
7. "citation_text" (string): If relevant enough (score ≥ 7), provide properly formatted JMLR-style citation text
8. "integration_suggestion" (string): If relevant enough (score ≥ 7), suggest how this should be integrated into the chapter (e.g., "Add to section on X, discussing Y")
9. "code_or_data" (string): Any mention of publicly available code, datasets, or implementations
10. "rationale" (string): Brief explanation of the relevance score

Respond ONLY with valid JSON, no other text."""

        try:
            create_kwargs = {
                "model": self.model,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.model.startswith("claude-sonnet-5-5"):
                create_kwargs["thinking"] = {"type": "between_tools"}
                create_kwargs["output_config"] = {"effort": "low"}
            message = self.client.messages.create(**create_kwargs)
            
            # Extract JSON from response
            response_text = "".join(
                b.text for b in message.content if getattr(b, "type", "") == "text"
            ).strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            analysis = json.loads(response_text.strip())
            
            # Add metadata
            analysis['paper_id'] = paper.get('pmid') or paper.get('arxiv_id')
            analysis['paper_title'] = paper['title']
            analysis['paper_doi'] = paper.get('doi')
            analysis['paper_journal'] = paper['journal']
            analysis['analyzed_date'] = time.strftime('%Y-%m-%d')
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.error(f"Response text: {response_text}")
            return {
                'relevance_score': 0,
                'rationale': f'JSON parsing error: {str(e)}',
                'error': True
            }
        except Exception as e:
            logger.error(f"Error analyzing paper: {e}")
            return {
                'relevance_score': 0,
                'rationale': f'Analysis error: {str(e)}',
                'error': True
            }
    
    def analyze_chapter_papers(
        self,
        chapter_id: str,
        papers: List[Dict[str, Any]],
        chapter_descriptions: Dict[str, str],
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Analyze all papers for a chapter.
        
        Args:
            chapter_id: Chapter identifier
            papers: List of paper metadata
            chapter_descriptions: Dictionary of chapter descriptions
            output_dir: Output directory for results
            
        Returns:
            List of analysis results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        chapter_desc = chapter_descriptions.get(
            chapter_id,
            "Healthcare AI chapter focusing on equity and underserved populations"
        )
        
        logger.info(f"Analyzing {len(papers)} papers for {chapter_id}")
        
        analyses = []
        for i, paper in enumerate(papers, 1):
            logger.info(f"  [{i}/{len(papers)}] Analyzing: {paper['title'][:80]}...")
            
            analysis = self.analyze_paper_relevance(paper, chapter_id, chapter_desc)
            analyses.append(analysis)
            
            # Rate limiting
            time.sleep(1.5)
        
        # Filter to only relevant papers (score >= 7)
        relevant_analyses = [a for a in analyses if a.get('relevance_score', 0) >= 7]
        
        # Sort by relevance score
        relevant_analyses.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Save results
        output_file = output_dir / f"{chapter_id}_analysis.json"
        with open(output_file, 'w') as f:
            json.dump({
                'chapter_id': chapter_id,
                'total_papers_analyzed': len(papers),
                'relevant_papers': len(relevant_analyses),
                'analyses': relevant_analyses
            }, f, indent=2)
        
        logger.info(f"  Found {len(relevant_analyses)} relevant papers (score >= 7)")
        logger.info(f"  Saved analysis to {output_file}")
        
        return relevant_analyses


# Chapter descriptions for context
CHAPTER_DESCRIPTIONS = {
    "chapter_01_clinical_informatics": "Foundations of clinical informatics with focus on how traditional healthcare AI fails underserved populations. Covers EHR systems, data standards (HL7 FHIR, SNOMED CT), representation bias, measurement bias, and working with biased clinical data.",
    
    "chapter_02_mathematical_foundations": "Mathematical foundations (linear algebra, probability, optimization) grounded in health equity applications. Covers matrix operations for social determinants, Bayesian reasoning with biased priors, and sensitivity analyses for assumption violations.",
    
    "chapter_03_healthcare_data_engineering": "Healthcare data engineering with equity focus. Covers data quality as social determinant, systematic missingness patterns, incomplete documentation in under-resourced settings, and integration of social determinants data.",
    
    "chapter_04_machine_learning_fundamentals": "Supervised learning, model selection, and hyperparameter tuning with healthcare-specific considerations. Covers clinical feature engineering, imbalanced outcomes, fairness-aware model development, and regularization for fairness.",
    
    "chapter_05_deep_learning_healthcare": "Neural network architectures for clinical prediction, medical imaging, and time-series analysis. Covers CNNs, RNNs, transformers, fairness-aware training, and uncertainty quantification in neural networks.",
    
    "chapter_06_clinical_nlp": "Natural language processing for clinical text with bias detection and mitigation. Covers clinical NER, relation extraction, multilingual text processing, and large language models with comprehensive safety evaluation.",
    
    "chapter_07_medical_imaging": "Medical image analysis with attention to generalization across diverse populations and care settings. Covers segmentation, detection, classification, self-supervised learning, and robustness to equipment variations.",
    
    "chapter_08_causal_inference": "Causal inference from observational healthcare data. Covers potential outcomes framework, propensity scores, instrumental variables, regression discontinuity, difference-in-differences, and synthetic controls for policy evaluation.",
    
    "chapter_09_advanced_nlp": "State-of-the-art NLP for complex clinical tasks. Covers transformer fine-tuning, clinical question answering, abstractive summarization, knowledge graph construction, and retrieval-augmented generation.",
    
    "chapter_10_survival_analysis": "Survival analysis and time-to-event modeling with competing risks. Covers Cox proportional hazards, parametric survival models, competing risks analysis, time-varying covariates, and random survival forests.",
    
    "chapter_11_reinforcement_learning": "Reinforcement learning for clinical decision support. Covers Markov decision processes, Q-learning, policy gradient methods, off-policy evaluation, safe RL for healthcare, and fairness in sequential decisions.",
    
    "chapter_12_federated_learning": "Federated learning for privacy-preserving healthcare AI. Covers federated averaging, communication-efficient FL, differential privacy, secure aggregation, and fairness across federated sites.",
    
    "chapter_13_fairness_metrics": "Formal definitions of algorithmic fairness and their limitations. Covers demographic parity, equalized odds, calibration, fairness impossibility results, and choosing appropriate fairness metrics for clinical contexts.",
    
    "chapter_14_interpretability": "Model interpretability and explainability for clinical use. Covers LIME, SHAP, attention visualization, concept activation vectors, counterfactual explanations, and trustworthy interpretability.",
    
    "chapter_15_validation": "Validation strategies for clinical AI that surface fairness issues before deployment. Covers validation design, sample size for fairness metrics, external validation across sites, prospective evaluation, and monitoring frameworks.",
    
    "chapter_16_uncertainty_quantification": "Uncertainty quantification and calibration across patient populations. Covers calibration assessment, conformal prediction, Bayesian uncertainty, ensemble methods, and out-of-distribution detection.",
    
    "chapter_26_llms_healthcare": "Large language models in healthcare with clinical applications, fine-tuning strategies, bias mitigation in foundation models, multilingual adaptation, and comprehensive safety evaluation before clinical use.",
    
    "chapter_27_multimodal_learning": "Multimodal learning combining imaging, text, time series, and structured EHR data. Covers vision-language models, multimodal transformers, attention mechanisms, missing modality handling, and fairness in multimodal systems.",
    
    "chapter_28_continual_learning": "Continual learning and model updating as clinical practices evolve. Covers catastrophic forgetting mitigation, elastic weight consolidation, online learning, drift adaptation, and fairness-preserving updates.",
}


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze papers with Claude API"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (or set ANTHROPIC_API_KEY environment variable)"
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("data/papers"),
        help="Directory containing paper JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis"),
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--chapters",
        nargs='+',
        help="Specific chapter IDs to process (default: all)"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        logger.error("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or use --api-key")
        return 1
    
    logger.info("Starting paper analysis with Claude")
    logger.info(f"Papers directory: {args.papers_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize analyzer
    analyzer = PaperAnalyzer(api_key=args.api_key)
    
    # Find all chapter paper files
    paper_files = list(args.papers_dir.glob("chapter_*_papers.json"))
    
    if not paper_files:
        logger.error(f"No paper files found in {args.papers_dir}")
        return 1
    
    logger.info(f"Found {len(paper_files)} chapter paper files")
    
    # Process each chapter
    all_results = {}
    for paper_file in paper_files:
        chapter_id = paper_file.stem.replace('_papers', '')
        
        # Skip if not in specified chapters
        if args.chapters and chapter_id not in args.chapters:
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {chapter_id}")
        logger.info(f"{'='*60}")
        
        # Load papers
        with open(paper_file, 'r') as f:
            papers = json.load(f)
        
        if not papers:
            logger.info(f"No papers to analyze for {chapter_id}")
            continue
        
        # Analyze papers
        analyses = analyzer.analyze_chapter_papers(
            chapter_id=chapter_id,
            papers=papers,
            chapter_descriptions=CHAPTER_DESCRIPTIONS,
            output_dir=args.output_dir
        )
        
        all_results[chapter_id] = {
            'total_analyzed': len(papers),
            'relevant_count': len(analyses),
            'analyses': analyses
        }
    
    # Save combined results
    combined_file = args.output_dir / "all_analyses.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")
    
    for chapter_id, results in all_results.items():
        logger.info(f"{chapter_id}:")
        logger.info(f"  - Analyzed: {results['total_analyzed']} papers")
        logger.info(f"  - Relevant: {results['relevant_count']} papers")
        if results['relevant_count'] > 0:
            high_scores = sum(1 for a in results['analyses'] if a.get('relevance_score', 0) >= 9)
            logger.info(f"  - High relevance (≥9): {high_scores} papers")
    
    logger.info(f"\nSaved combined results to {combined_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
