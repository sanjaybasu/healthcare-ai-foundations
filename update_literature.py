#!/usr/bin/env python3
"""
Automated literature search and chapter update system.

This script:
1. Searches PubMed, arXiv, and other sources for recent publications
2. Uses Claude to assess relevance to each chapter
3. Generates update suggestions for chapters
4. Creates properly formatted citations in JMLR style
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time

import requests
from Bio import Entrez
import anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Chapter mapping with search terms
CHAPTER_MAPPINGS = {
    "chapter_01_clinical_informatics.md": {
        "title": "Clinical Informatics Foundations",
        "keywords": [
            "electronic health records", "clinical informatics", "health equity",
            "algorithmic bias healthcare", "FHIR", "clinical decision support",
            "healthcare data bias", "pulse oximetry bias"
        ]
    },
    "chapter_02_mathematical_foundations.md": {
        "title": "Mathematical Foundations",
        "keywords": [
            "fairness metrics machine learning", "algorithmic fairness",
            "Bayesian inference healthcare", "optimization algorithms",
            "linear algebra medical applications"
        ]
    },
    "chapter_03_healthcare_data_engineering.md": {
        "title": "Healthcare Data Engineering",
        "keywords": [
            "missing data healthcare", "data quality EHR",
            "healthcare data pipelines", "FHIR processing",
            "clinical data standardization", "health data interoperability"
        ]
    },
    "chapter_04_machine_learning_fundamentals.md": {
        "title": "Machine Learning Fundamentals",
        "keywords": [
            "fairness machine learning", "logistic regression clinical",
            "random forests healthcare", "gradient boosting medical",
            "algorithmic fairness evaluation"
        ]
    },
    "chapter_05_deep_learning_healthcare.md": {
        "title": "Deep Learning for Healthcare",
        "keywords": [
            "neural networks clinical", "transformers healthcare",
            "LSTM medical time series", "deep learning fairness",
            "attention mechanisms clinical data"
        ]
    },
    "chapter_06_clinical_nlp.md": {
        "title": "Clinical NLP",
        "keywords": [
            "clinical natural language processing", "medical NER",
            "bias clinical notes", "clinical text mining",
            "medical language models", "BioBERT", "ClinicalBERT"
        ]
    },
    "chapter_07_medical_imaging.md": {
        "title": "Medical Imaging AI",
        "keywords": [
            "medical image analysis", "radiology AI fairness",
            "pathology deep learning", "medical imaging bias",
            "chest X-ray AI", "skin lesion classification fairness"
        ]
    },
    "chapter_08_clinical_time_series.md": {
        "title": "Clinical Time Series",
        "keywords": [
            "clinical time series analysis", "ICU prediction models",
            "physiological signal processing", "irregular time series clinical",
            "missing data time series healthcare"
        ]
    },
    "chapter_09_advanced_clinical_nlp.md": {
        "title": "Advanced Clinical NLP",
        "keywords": [
            "medical knowledge graphs", "clinical question answering",
            "clinical information retrieval", "medical entity linking",
            "clinical reasoning AI"
        ]
    },
    "chapter_10_survival_analysis.md": {
        "title": "Survival Analysis",
        "keywords": [
            "survival analysis machine learning", "competing risks",
            "Cox proportional hazards", "random survival forests",
            "fairness survival models"
        ]
    },
    "chapter_11_causal_inference.md": {
        "title": "Causal Inference",
        "keywords": [
            "causal inference healthcare", "causal machine learning",
            "instrumental variables health", "difference-in-differences",
            "causal fairness", "counterfactual fairness"
        ]
    },
    "chapter_12_federated_learning_privacy.md": {
        "title": "Federated Learning and Privacy",
        "keywords": [
            "federated learning healthcare", "differential privacy medical",
            "privacy-preserving machine learning", "multi-site learning",
            "secure computation healthcare"
        ]
    },
    "chapter_13_transfer_learning.md": {
        "title": "Transfer Learning",
        "keywords": [
            "transfer learning medical", "domain adaptation healthcare",
            "few-shot learning clinical", "meta-learning medicine",
            "cross-site generalization"
        ]
    },
    "chapter_14_interpretability_explainability.md": {
        "title": "Interpretability and Explainability",
        "keywords": [
            "interpretable machine learning healthcare", "SHAP clinical",
            "explainable AI medical", "attention visualization",
            "counterfactual explanations clinical"
        ]
    },
    "chapter_15_validation_strategies.md": {
        "title": "Validation Strategies",
        "keywords": [
            "clinical validation machine learning", "external validation",
            "temporal validation healthcare", "fairness evaluation methods",
            "performance metrics clinical AI"
        ]
    },
    "chapter_16_uncertainty_calibration.md": {
        "title": "Uncertainty Quantification",
        "keywords": [
            "uncertainty quantification clinical", "calibration machine learning",
            "conformal prediction healthcare", "Bayesian neural networks medical",
            "uncertainty fairness"
        ]
    },
    "chapter_17_regulatory_considerations.md": {
        "title": "Regulatory Considerations",
        "keywords": [
            "FDA software medical device", "AI regulation healthcare",
            "clinical AI approval", "predetermined change control",
            "algorithmic fairness regulation"
        ]
    },
    "chapter_18_implementation_science.md": {
        "title": "Implementation Science",
        "keywords": [
            "clinical AI implementation", "stakeholder engagement AI",
            "workflow integration machine learning", "clinician training AI",
            "implementation equity"
        ]
    },
    "chapter_19_human_ai_collaboration.md": {
        "title": "Human-AI Collaboration",
        "keywords": [
            "human-AI collaboration healthcare", "automation bias medical",
            "clinical decision support design", "appropriate reliance AI",
            "complementary team performance"
        ]
    },
    "chapter_20_monitoring_maintenance.md": {
        "title": "Post-Deployment Monitoring",
        "keywords": [
            "model monitoring healthcare", "distribution shift detection",
            "fairness monitoring deployed", "performance degradation",
            "concept drift clinical"
        ]
    },
    "chapter_21_health_equity_metrics.md": {
        "title": "Health Equity Metrics",
        "keywords": [
            "health equity metrics", "fairness metrics comprehensive",
            "intersectional fairness", "algorithmic fairness healthcare",
            "disparity measurement AI"
        ]
    },
    "chapter_22_clinical_decision_support.md": {
        "title": "Clinical Decision Support",
        "keywords": [
            "clinical decision support systems", "diagnostic AI",
            "treatment recommendation algorithms", "alert fatigue",
            "decision support equity"
        ]
    },
    "chapter_23_precision_medicine_genomics.md": {
        "title": "Precision Medicine and Genomics",
        "keywords": [
            "pharmacogenomics AI", "polygenic risk scores",
            "multi-omic integration", "genomic prediction fairness",
            "precision medicine equity"
        ]
    },
    "chapter_24_population_health_screening.md": {
        "title": "Population Health Management",
        "keywords": [
            "population health AI", "risk stratification equity",
            "screening strategies fairness", "care management algorithms",
            "outbreak detection"
        ]
    },
    "chapter_25_sdoh_integration.md": {
        "title": "Social Determinants of Health",
        "keywords": [
            "social determinants health AI", "neighborhood effects health",
            "environmental health AI", "community data integration",
            "structural determinants health"
        ]
    },
    "chapter_26_llms_in_healthcare.md": {
        "title": "Large Language Models",
        "keywords": [
            "large language models healthcare", "clinical LLM",
            "GPT medical applications", "LLM bias healthcare",
            "foundation models clinical", "prompt engineering medical"
        ]
    },
    "chapter_27_multimodal_learning.md": {
        "title": "Multi-Modal Learning",
        "keywords": [
            "multimodal learning healthcare", "fusion models clinical",
            "vision-language models medical", "missing modality imputation",
            "multimodal fairness"
        ]
    },
    "chapter_28_continual_learning.md": {
        "title": "Continual Learning",
        "keywords": [
            "continual learning healthcare", "catastrophic forgetting medical",
            "lifelong learning clinical", "model updating fairness",
            "distribution shift adaptation"
        ]
    },
    "chapter_29_global_health_ai.md": {
        "title": "Global Health AI",
        "keywords": [
            "global health AI", "resource-limited settings",
            "offline AI healthcare", "low-resource medical AI",
            "task-shifting algorithms", "mobile health AI"
        ]
    },
    "chapter_30_research_frontiers_equity.md": {
        "title": "Research Frontiers in Equity",
        "keywords": [
            "algorithmic reparations", "environmental justice AI",
            "intersectional machine learning", "structural racism algorithms",
            "equity-centered AI research"
        ]
    }
}


# Major journals and conferences to monitor
SOURCES = {
    "journals": [
        "Nature", "Science", "Cell", "Lancet",
        "New England Journal of Medicine", "JAMA",
        "Nature Medicine", "Nature Machine Intelligence",
        "JMIR", "npj Digital Medicine"
    ],
    "conferences": [
        "NeurIPS", "ICML", "ICLR", "AAAI",
        "ACM CHIL", "ML4H", "AMIA"
    ],
    "arxiv_categories": [
        "cs.LG",  # Machine Learning
        "cs.AI",  # Artificial Intelligence
        "stat.ML",  # Machine Learning (Statistics)
        "cs.CY",  # Computers and Society
        "q-bio"  # Quantitative Biology
    ]
}


class LiteratureSearcher:
    """Search multiple sources for relevant recent publications."""
    
    def __init__(
        self,
        pubmed_api_key: str,
        days_back: int = 7
    ):
        """
        Initialize literature searcher.
        
        Args:
            pubmed_api_key: NCBI API key for PubMed
            days_back: How many days back to search
        """
        self.pubmed_api_key = pubmed_api_key
        self.days_back = days_back
        Entrez.email = "automated@github-actions.com"
        Entrez.api_key = pubmed_api_key
        
    def search_pubmed(
        self,
        keywords: List[str],
        max_results: int = 20
    ) -> List[Dict]:
        """
        Search PubMed for recent publications.
        
        Args:
            keywords: List of search terms
            max_results: Maximum number of results to return
            
        Returns:
            List of publication dictionaries
        """
        logger.info(f"Searching PubMed for: {', '.join(keywords[:3])}...")
        
        # Build search query
        query_parts = [f'"{kw}"' for kw in keywords]
        query = " OR ".join(query_parts)
        
        # Add date filter
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        date_filter = f" AND {start_date.strftime('%Y/%m/%d')}[PDAT]:{end_date.strftime('%Y/%m/%d')}[PDAT]"
        full_query = query + date_filter
        
        try:
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=full_query,
                retmax=max_results,
                sort="relevance"
            )
            results = Entrez.read(handle)
            handle.close()
            
            pmids = results["IdList"]
            if not pmids:
                logger.info("No results found")
                return []
            
            logger.info(f"Found {len(pmids)} articles")
            
            # Fetch article details
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                rettype="medline",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
            
            # Parse articles
            articles = []
            for record in records['PubmedArticle']:
                try:
                    article = self._parse_pubmed_record(record)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing record: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def _parse_pubmed_record(self, record: Dict) -> Optional[Dict]:
        """Parse a PubMed record into structured format."""
        try:
            article = record['MedlineCitation']['Article']
            pmid = record['MedlineCitation']['PMID']
            
            # Get title
            title = article.get('ArticleTitle', '')
            
            # Get authors
            authors = []
            if 'AuthorList' in article:
                for author in article['AuthorList'][:3]:  # First 3 authors
                    if 'LastName' in author and 'Initials' in author:
                        authors.append(f"{author['LastName']} {author['Initials']}")
            
            # Get journal
            journal = article['Journal']['Title'] if 'Journal' in article else ''
            
            # Get publication date
            pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', '')
            
            # Get abstract
            abstract = ''
            if 'Abstract' in article:
                abstract_text = article['Abstract'].get('AbstractText', [])
                if isinstance(abstract_text, list):
                    abstract = ' '.join([str(text) for text in abstract_text])
                else:
                    abstract = str(abstract_text)
            
            # Get DOI
            doi = None
            if 'ELocationID' in article:
                for eloc in article['ELocationID']:
                    if eloc.attributes.get('EIdType') == 'doi':
                        doi = str(eloc)
                        break
            
            return {
                'pmid': str(pmid),
                'title': title,
                'authors': authors,
                'journal': journal,
                'year': year,
                'abstract': abstract,
                'doi': doi,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                'source': 'PubMed'
            }
            
        except Exception as e:
            logger.warning(f"Error parsing record: {e}")
            return None
    
    def search_arxiv(
        self,
        keywords: List[str],
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search arXiv for recent pre-prints.
        
        Args:
            keywords: List of search terms
            max_results: Maximum number of results
            
        Returns:
            List of pre-print dictionaries
        """
        logger.info(f"Searching arXiv for: {', '.join(keywords[:3])}...")
        
        # Build query
        query_parts = [f'all:"{kw}"' for kw in keywords[:5]]  # Limit keywords for arXiv
        query = " OR ".join(query_parts)
        
        # Add category filter for health/ML papers
        query += " AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML OR cat:q-bio)"
        
        try:
            import urllib.parse
            import xml.etree.ElementTree as ET
            
            base_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"arXiv API error: {response.status_code}")
                return []
            
            # Parse XML response
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            articles = []
            for entry in root.findall('atom:entry', namespace):
                try:
                    article = self._parse_arxiv_entry(entry, namespace)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing arXiv entry: {e}")
                    continue
            
            logger.info(f"Found {len(articles)} arXiv papers")
            return articles
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def _parse_arxiv_entry(self, entry, namespace: Dict) -> Optional[Dict]:
        """Parse an arXiv entry into structured format."""
        try:
            # Get ID and extract arXiv number
            id_elem = entry.find('atom:id', namespace)
            arxiv_id = id_elem.text.split('/')[-1] if id_elem is not None else None
            
            # Get title
            title_elem = entry.find('atom:title', namespace)
            title = title_elem.text.strip() if title_elem is not None else ''
            
            # Get authors
            authors = []
            for author in entry.findall('atom:author', namespace)[:3]:
                name_elem = author.find('atom:name', namespace)
                if name_elem is not None:
                    # Convert "FirstName LastName" to "LastName F"
                    name_parts = name_elem.text.strip().split()
                    if len(name_parts) >= 2:
                        formatted = f"{name_parts[-1]} {name_parts[0][0]}"
                        authors.append(formatted)
            
            # Get abstract
            summary_elem = entry.find('atom:summary', namespace)
            abstract = summary_elem.text.strip() if summary_elem is not None else ''
            
            # Get published date
            published_elem = entry.find('atom:published', namespace)
            year = published_elem.text[:4] if published_elem is not None else ''
            
            return {
                'arxiv_id': arxiv_id,
                'title': title,
                'authors': authors,
                'journal': 'arXiv',
                'year': year,
                'abstract': abstract,
                'url': f"https://arxiv.org/abs/{arxiv_id}",
                'source': 'arXiv'
            }
            
        except Exception as e:
            logger.warning(f"Error parsing arXiv entry: {e}")
            return None


class ChapterUpdater:
    """Use Claude to assess relevance and update chapters."""
    
    def __init__(self, anthropic_api_key: str):
        """
        Initialize chapter updater.
        
        Args:
            anthropic_api_key: Anthropic API key
        """
        self.client = anthropic.Anthropic(
            api_key=anthropic_api_key.strip() if anthropic_api_key else anthropic_api_key
        )
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        
    def assess_relevance(
        self,
        article: Dict,
        chapter_title: str,
        chapter_keywords: List[str]
    ) -> Tuple[bool, float, str]:
        """
        Use Claude to assess if article is relevant to chapter.
        
        Args:
            article: Article dictionary
            chapter_title: Chapter title
            chapter_keywords: Chapter keywords
            
        Returns:
            Tuple of (is_relevant, confidence_score, reasoning)
        """
        prompt = f"""You are an expert in healthcare AI research. Assess whether this paper is highly relevant to a textbook chapter.

Chapter: {chapter_title}
Chapter Keywords: {', '.join(chapter_keywords)}

Paper Title: {article['title']}
Authors: {', '.join(article['authors'])}
Journal/Source: {article['journal']} ({article['year']})
Abstract: {article['abstract'][:1000]}

Is this paper highly relevant to this chapter? Consider:
1. Does it present novel methods or significant findings?
2. Is it from a top-tier venue (Nature, Science, NeurIPS, ICML, NEJM, JAMA)?
3. Does it address fairness, equity, or underserved populations?
4. Is it technically rigorous and implementation-focused?

Respond with a JSON object:
{{
    "is_relevant": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "key_contributions": ["list", "of", "contributions"],
    "suggested_section": "where in chapter this belongs"
}}
"""
        
        try:
            create_kwargs = {
                "model": self.model,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.model.startswith("claude-sonnet-5-5"):
                create_kwargs["thinking"] = {"type": "between_tools"}
                create_kwargs["output_config"] = {"effort": "low"}
            response = self.client.messages.create(**create_kwargs)
            
            # Parse response
            text = "".join(
                b.text for b in response.content if getattr(b, "type", "") == "text"
            )
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return (
                    result['is_relevant'],
                    result['confidence'],
                    result['reasoning']
                )
            else:
                logger.warning("Could not parse Claude response")
                return False, 0.0, "Parse error"
                
        except Exception as e:
            logger.error(f"Error assessing relevance: {e}")
            return False, 0.0, str(e)
    
    def generate_citation(self, article: Dict) -> str:
        """
        Generate JMLR-style citation for article.
        
        Args:
            article: Article dictionary
            
        Returns:
            Formatted citation string
        """
        try:
            authors_str = ", ".join(article['authors'])
            if len(article['authors']) > 3:
                authors_str = ", ".join(article['authors'][:3]) + ", et al."
            
            # Format based on source
            if article['source'] == 'PubMed':
                citation = f"{authors_str} ({article['year']}). "
                citation += f"{article['title']} "
                citation += f"*{article['journal']}*"
                if article.get('doi'):
                    citation += f". https://doi.org/{article['doi']}"
                else:
                    citation += f". {article['url']}"
            else:  # arXiv
                citation = f"{authors_str} ({article['year']}). "
                citation += f"{article['title']} "
                citation += f"*arXiv preprint {article['arxiv_id']}*. "
                citation += article['url']
            
            return citation
            
        except Exception as e:
            logger.error(f"Error generating citation: {e}")
            return ""
    
    def update_chapter(
        self,
        chapter_path: Path,
        relevant_articles: List[Tuple[Dict, float, str]]
    ) -> bool:
        """
        Update chapter with relevant articles.
        
        Args:
            chapter_path: Path to chapter file
            relevant_articles: List of (article, confidence, reasoning) tuples
            
        Returns:
            True if chapter was updated
        """
        if not relevant_articles:
            logger.info(f"No updates for {chapter_path.name}")
            return False
        
        logger.info(f"Updating {chapter_path.name} with {len(relevant_articles)} articles")
        
        try:
            # Read current chapter
            with open(chapter_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Prepare update section
            update_section = "\n\n## Recent Developments\n\n"
            update_section += f"*Last updated: {datetime.now().strftime('%B %d, %Y')}*\n\n"
            
            for article, confidence, reasoning in relevant_articles:
                update_section += f"### {article['title']}\n\n"
                update_section += f"**Authors:** {', '.join(article['authors'])}\n\n"
                update_section += f"**Source:** {article['journal']} ({article['year']})\n\n"
                update_section += f"**Relevance:** {reasoning}\n\n"
                update_section += f"**Citation:** {self.generate_citation(article)}\n\n"
                update_section += f"[Access Paper]({article['url']})\n\n"
                update_section += "---\n\n"
            
            # Check if chapter already has Recent Developments section
            if "## Recent Developments" in content:
                # Replace existing section
                import re
                pattern = r"## Recent Developments.*?(?=\n## |\Z)"
                content = re.sub(pattern, update_section.strip(), content, flags=re.DOTALL)
            else:
                # Add before bibliography if it exists
                if "## Bibliography" in content:
                    content = content.replace("## Bibliography", update_section + "## Bibliography")
                else:
                    # Add at end
                    content += "\n" + update_section
            
            # Write updated chapter
            with open(chapter_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Successfully updated {chapter_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating chapter {chapter_path.name}: {e}")
            return False


def main():
    """Main execution function."""
    logger.info("Starting literature update process")
    
    # Get API keys
    pubmed_key = os.environ.get('PUBMED_API_KEY')
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not pubmed_key or not anthropic_key:
        logger.error("Missing required API keys")
        sys.exit(1)
    
    # Initialize searcher and updater
    searcher = LiteratureSearcher(pubmed_key, days_back=7)
    updater = ChapterUpdater(anthropic_key)
    
    # Process each chapter
    chapters_updated = 0
    chapters_dir = Path(".")
    
    for chapter_file, chapter_info in CHAPTER_MAPPINGS.items():
        logger.info(f"\nProcessing: {chapter_info['title']}")
        
        chapter_path = chapters_dir / chapter_file
        if not chapter_path.exists():
            logger.warning(f"Chapter file not found: {chapter_file}")
            continue
        
        # Search PubMed
        pubmed_articles = searcher.search_pubmed(
            chapter_info['keywords'],
            max_results=15
        )
        
        # Search arXiv
        arxiv_articles = searcher.search_arxiv(
            chapter_info['keywords'],
            max_results=10
        )
        
        all_articles = pubmed_articles + arxiv_articles
        logger.info(f"Found {len(all_articles)} total articles")
        
        # Assess relevance
        relevant_articles = []
        for article in all_articles:
            is_relevant, confidence, reasoning = updater.assess_relevance(
                article,
                chapter_info['title'],
                chapter_info['keywords']
            )
            
            # Only include highly relevant articles
            if is_relevant and confidence >= 0.7:
                relevant_articles.append((article, confidence, reasoning))
                logger.info(f"✓ Relevant: {article['title'][:60]}... ({confidence:.2f})")
            
            # Rate limiting
            time.sleep(1)
        
        # Update chapter
        if relevant_articles:
            if updater.update_chapter(chapter_path, relevant_articles):
                chapters_updated += 1
        
        # Rate limiting between chapters
        time.sleep(2)
    
    logger.info(f"\nUpdate complete: {chapters_updated} chapters updated")
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'chapters_updated': chapters_updated,
        'total_chapters': len(CHAPTER_MAPPINGS)
    }
    
    with open('update_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
