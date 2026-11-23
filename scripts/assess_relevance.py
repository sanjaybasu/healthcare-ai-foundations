#!/usr/bin/env python3
"""
Ultra-Selective Relevance Assessment for Healthcare AI Equity Textbook

This version focuses ONLY on groundbreaking, highly-cited papers that represent
major advances in the field. Uses aggressive filtering to minimize API calls
while ensuring only the most impactful research is included.

Author: Sanjay Basu, MD PhD
License: MIT
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import time
import random
from datetime import datetime, timedelta

from anthropic import Anthropic, RateLimitError, APIError, APIConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RelevanceAssessment:
    """Represents a relevance assessment for a paper-chapter pair."""
    paper_id: str
    chapter_file: str
    relevance_score: float  # 0.0 to 1.0
    confidence: str  # 'high', 'medium', 'low'
    reasoning: str
    should_include: bool
    suggested_section: Optional[str] = None
    key_contributions: Optional[List[str]] = None
    impact_score: Optional[float] = None  # New: Combined impact metric
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class GroundbreakingPaperFilter:
    """
    Multi-stage filter to identify only groundbreaking papers before API assessment.
    """
    
    # Top-tier journals for healthcare AI
    TIER1_JOURNALS = {
        'nature', 'science', 'cell', 'nature medicine', 'nature biotechnology',
        'nature biomedical engineering', 'the new england journal of medicine',
        'nejm', 'jama', 'the lancet', 'cell systems', 'nature methods'
    }
    
    # High-impact healthcare AI journals
    TIER2_JOURNALS = {
        'science translational medicine', 'lancet digital health',
        'npj digital medicine', 'nature communications', 'pnas',
        'jama internal medicine', 'bmj', 'annals of internal medicine',
        'plos medicine', 'nature human behaviour', 'cell reports medicine'
    }
    
    # Keywords indicating major methodological advances
    BREAKTHROUGH_KEYWORDS = {
        'foundation model', 'large language model', 'multimodal', 'transformer',
        'state-of-the-art', 'sota', 'benchmark', 'outperforms', 'breakthrough',
        'novel', 'first-in-human', 'clinical trial', 'prospective', 'randomized',
        'fda approval', 'real-world', 'deployment', 'implementation',
        'validation', 'external validation', 'generalization', 'fairness',
        'bias mitigation', 'health equity', 'disparities', 'underserved',
        'explainability', 'interpretability', 'causal', 'counterfactual'
    }
    
    # Keywords indicating theoretical/review papers (lower priority)
    REVIEW_KEYWORDS = {
        'review', 'survey', 'perspective', 'commentary', 'opinion',
        'letter', 'correspondence', 'editorial'
    }
    
    def __init__(
        self,
        min_citation_threshold: int = 10,
        recency_months: int = 24,
        require_tier1_or_citations: bool = True
    ):
        """
        Initialize the groundbreaking paper filter.
        
        Args:
            min_citation_threshold: Minimum citations for older papers
            recency_months: Papers within this period get relaxed citation requirements
            require_tier1_or_citations: Require either tier 1 journal OR high citations
        """
        self.min_citation_threshold = min_citation_threshold
        self.recency_months = recency_months
        self.require_tier1_or_citations = require_tier1_or_citations
    
    def calculate_impact_score(self, paper: Dict) -> float:
        """
        Calculate a composite impact score for a paper.
        
        Factors:
        - Journal tier (0-40 points)
        - Citation count (0-30 points)
        - Recency bonus (0-10 points)
        - Breakthrough keywords (0-10 points)
        - Equity focus (0-10 points)
        
        Returns:
            Impact score from 0-100
        """
        score = 0.0
        
        # Journal tier (40 points max)
        journal = (paper.get('journal') or '').lower()  # ✅ FIXED
        if any(tier1 in journal for tier1 in self.TIER1_JOURNALS):
            score += 40.0
        elif any(tier2 in journal for tier2 in self.TIER2_JOURNALS):
            score += 25.0
        elif journal:
            score += 10.0
        
        # Citation count (30 points max)
        citations = paper.get('citations')  # ✅ FIXED
        if citations is not None:
            if citations >= 100:
                score += 40
            elif citations >= 50:
                score += 30
            elif citations >= 20:
                score += 20
            elif citations >= 10:
                score += 10
            elif citations >= 5:
                score += 5
        else:
            score += 5
        
        # Recency bonus (10 points max)
        try:
            pub_date = datetime.fromisoformat(paper.get('publication_date', '2000-01-01'))
            months_old = (datetime.now() - pub_date).days / 30.0
            if months_old <= 6:
                score += 10.0
            elif months_old <= 12:
                score += 7.0
            elif months_old <= self.recency_months:
                score += 4.0
        except:
            pass
        
        # Breakthrough keywords (10 points max)
        title = (paper.get('title') or '').lower()      # ✅ FIXED
        abstract = (paper.get('abstract') or '').lower()  # ✅ FIXED
        text = f"{title} {abstract}"
        
        breakthrough_count = sum(1 for kw in self.BREAKTHROUGH_KEYWORDS if kw in text)
        score += min(breakthrough_count * 2, 10.0)
        
        # Equity focus bonus (10 points max)
        equity_keywords = ['equity', 'fairness', 'bias', 'disparity', 'underserved', 
                          'vulnerable', 'marginalized', 'racial', 'socioeconomic']
        equity_count = sum(1 for kw in equity_keywords if kw in text)
        score += min(equity_count * 2.5, 10.0)
        
        # Penalty for review papers (-20 points)
        if any(kw in title for kw in self.REVIEW_KEYWORDS):
            score -= 20.0
        
        return max(0.0, min(100.0, score))
        
    def is_groundbreaking(self, paper: Dict, impact_threshold: float = 50.0) -> Tuple[bool, float, str]:
        """
        Determine if a paper is groundbreaking enough to assess.
        
        Args:
            paper: Paper metadata
            impact_threshold: Minimum impact score (0-100)
            
        Returns:
            (is_groundbreaking, impact_score, reason)
        """
        impact_score = self.calculate_impact_score(paper)
        
        # Check if meets threshold
        if impact_score < impact_threshold:
            return False, impact_score, f"Impact score {impact_score:.1f} below threshold {impact_threshold}"
        
        # Additional filters
        journal = (paper.get('journal') or '').lower()
        citations = paper.get('citations') or 0  
        title = (paper.get('title') or '').lower()
        
        # Must not be a review paper (unless very highly cited)
        is_review = any(kw in title for kw in self.REVIEW_KEYWORDS)
        if is_review and citations < 50:
            return False, impact_score, "Review paper with insufficient citations"
        
        # If not tier 1 journal, must have significant citations (unless very recent)
        is_tier1 = any(t1 in journal for t1 in self.TIER1_JOURNALS)
        try:
            pub_date = datetime.fromisoformat(paper.get('publication_date', '2000-01-01'))
            is_recent = (datetime.now() - pub_date).days <= 180  # 6 months
        except:
            is_recent = False
        
        if not is_tier1 and not is_recent and citations < self.min_citation_threshold:
            return False, impact_score, f"Not tier 1 journal and citations ({citations}) below threshold"
        
        reason = []
        if is_tier1:
            reason.append("Tier 1 journal")
        if citations >= 20:
            reason.append(f"{citations} citations")
        if is_recent:
            reason.append("Recent publication")
        
        return True, impact_score, " + ".join(reason)
    
    def filter_papers_for_chapter(
        self,
        papers: List[Dict],
        chapter: Dict,
        impact_threshold: float = 50.0,
        max_papers: int = 15
    ) -> List[Tuple[Dict, float]]:
        """
        Filter papers for a chapter, keeping only the most groundbreaking.
        
        Args:
            papers: List of all papers
            chapter: Chapter metadata
            impact_threshold: Minimum impact score
            max_papers: Maximum papers to return
            
        Returns:
            List of (paper, impact_score) tuples, sorted by impact
        """
        # First pass: Calculate impact scores and check if groundbreaking
        groundbreaking_papers = []
        
        for paper in papers:
            is_gb, impact_score, reason = self.is_groundbreaking(paper, impact_threshold)
            if is_gb:
                groundbreaking_papers.append((paper, impact_score, reason))
        
        # Sort by impact score
        groundbreaking_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Second pass: Keyword relevance to chapter
        chapter_keywords = self._extract_chapter_keywords(chapter)
        scored_papers = []
        
        for paper, impact_score, reason in groundbreaking_papers:
            relevance = self._calculate_keyword_relevance(paper, chapter_keywords)
            # Combined score: 70% impact, 30% keyword relevance
            combined_score = 0.7 * impact_score + 0.3 * relevance
            scored_papers.append((paper, combined_score))
        
        # Sort by combined score and take top N
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        return scored_papers[:max_papers]
    
    def _extract_chapter_keywords(self, chapter: Dict) -> Set[str]:
        """Extract important keywords from chapter."""
        text = f"{chapter.get('title', '')} {' '.join(chapter.get('objectives', []))}"
        text = text.lower()
        
        # Extract meaningful words (3+ characters, not common words)
        stopwords = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'are', 'was', 'were'}
        words = set()
        for word in text.split():
            word = word.strip('.,;:!?()[]{}')
            if len(word) >= 3 and word not in stopwords:
                words.add(word)
        
        return words
    
    def _calculate_keyword_relevance(self, paper: Dict, chapter_keywords: Set[str]) -> float:
        """Calculate keyword-based relevance score (0-100)."""
        paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        # Count keyword matches
        matches = sum(1 for kw in chapter_keywords if kw in paper_text)
        
        # Normalize to 0-100
        if not chapter_keywords:
            return 50.0
        
        relevance = (matches / len(chapter_keywords)) * 100.0
        return min(100.0, relevance)


class UltraSelectiveRelevanceAssessor:
    """Assess only the most groundbreaking papers using aggressive filtering."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_delay: float = 2.0,
        max_retries: int = 5,
        impact_threshold: float = 55.0,
        min_citations: int = 10
    ):
        """
        Initialize the ultra-selective assessor.
        
        Args:
            api_key: Anthropic API key
            base_delay: Base delay between API calls
            max_retries: Maximum retry attempts
            impact_threshold: Minimum impact score (0-100) for consideration
            min_citations: Minimum citations for non-tier1 journals
        """
        self.api_key = (api_key or os.getenv('ANTHROPIC_API_KEY') or '').strip()
        if not self.api_key:
            raise ValueError("Anthropic API key required")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.base_delay = base_delay
        self.max_retries = max_retries
        
        # Initialize groundbreaking filter
        self.filter = GroundbreakingPaperFilter(
            min_citation_threshold=min_citations,
            recency_months=24,
            require_tier1_or_citations=True
        )
        self.impact_threshold = impact_threshold
        
        # Track statistics
        self.api_calls = 0
        self.api_errors = 0
        self.rate_limit_hits = 0
        self.papers_filtered_out = 0
        self.papers_assessed = 0
    
    def _load_chapter_metadata(self, chapter_path: Path) -> Dict:
        """Extract metadata from a chapter markdown file."""
        try:
            with open(chapter_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title
            title = ""
            for line in content.split('\n'):
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            
            # Extract learning objectives
            objectives = []
            in_objectives = False
            for line in content.split('\n'):
                if 'learning objectives' in line.lower():
                    in_objectives = True
                    continue
                if in_objectives:
                    if line.startswith('- ') or line.startswith('* '):
                        objectives.append(line[2:].strip())
                    elif line.startswith('#'):
                        break
            
            # Get first 1000 characters as context
            context = content[:1000]
            
            return {
                'title': title,
                'filename': chapter_path.name,
                'objectives': objectives,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Error loading chapter {chapter_path}: {e}")
            return {
                'title': chapter_path.stem,
                'filename': chapter_path.name,
                'objectives': [],
                'context': ''
            }
    
    def _build_assessment_prompt(self, paper: Dict, chapter: Dict, impact_score: float) -> str:
        """Build ultra-selective assessment prompt emphasizing groundbreaking nature."""
        prompt = f"""You are an expert academic editor for a prestigious healthcare AI textbook focused on equity. Your task is to assess whether a GROUNDBREAKING research paper merits inclusion in a specific chapter.

CRITICAL: This textbook includes ONLY the most impactful, seminal papers that represent major advances in the field. Be HIGHLY selective. Only recommend papers that:
1. Introduce novel methods/algorithms with demonstrated superiority
2. Report major clinical findings from rigorous studies
3. Present groundbreaking applications with real-world impact
4. Advance health equity in meaningful, measurable ways
5. Set new benchmarks or establish new research directions

CHAPTER INFORMATION:
Title: {chapter['title']}
File: {chapter['filename']}

Learning Objectives:
{chr(10).join(f"- {obj}" for obj in chapter['objectives']) if chapter['objectives'] else "Not specified"}

PAPER INFORMATION:
Title: {paper['title']}
Authors: {', '.join(paper['authors'][:5])}{'...' if len(paper['authors']) > 5 else ''}
Journal: {paper.get('journal', 'Unknown')}
Citations: {paper.get('citations', 'Unknown')}
Date: {paper['publication_date']}
Source: {paper['source']}
Impact Score: {impact_score:.1f}/100 (pre-filtered as high-impact)

Abstract:
{paper['abstract'][:1000]}{'...' if len(paper.get('abstract', '')) > 1000 else ''}

ULTRA-SELECTIVE ASSESSMENT CRITERIA:
1. **Novelty & Innovation**: Does this introduce genuinely NEW methods/insights? (Not incremental)
2. **Rigor**: Is methodology gold-standard? (RCTs, large cohorts, multi-site validation)
3. **Impact**: Does this change practice or open new research directions?
4. **Equity Leadership**: Does this advance health equity in substantive ways?
5. **Pedagogical Value**: Is this THE paper to teach this concept?

SCORING GUIDANCE (Be STRICT):
- 0.9-1.0: Seminal paper, absolutely must include (e.g., paper that introduced transformers for clinical NLP)
- 0.75-0.89: Major advance, strongly recommend (e.g., first prospective validation of AI in clinical workflow)
- 0.6-0.74: Significant contribution, consider including (high-quality but not paradigm-shifting)
- 0.0-0.59: Does not meet bar for inclusion (default to rejection unless exceptional)

RESPONSE FORMAT (JSON only):
{{
  "relevance_score": <float 0.0-1.0>,
  "confidence": "<high|medium|low>",
  "should_include": <true|false>,
  "reasoning": "<2-3 sentences explaining why this is/isn't groundbreaking>",
  "suggested_section": "<section name or null>",
  "key_contributions": [<2-4 specific major contributions>]
}}

Remember: Default to REJECTION unless this is truly a groundbreaking paper that advances the field."""

        return prompt
    
    def _exponential_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0, 0.3 * delay)
        return min(delay + jitter, 60.0)
    
    def assess_paper_chapter_relevance(
        self,
        paper: Dict,
        chapter: Dict,
        impact_score: float
    ) -> Optional[RelevanceAssessment]:
        """Assess a pre-filtered groundbreaking paper."""
        prompt = self._build_assessment_prompt(paper, chapter, impact_score)
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = self._exponential_backoff_delay(attempt - 1)
                    logger.info(f"Retry {attempt}/{self.max_retries}, waiting {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    time.sleep(self.base_delay)
                
                self.api_calls += 1
                create_kwargs = {
                    "model": self.model,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if self.model.startswith("claude-sonnet-5-5"):
                    create_kwargs["thinking"] = {"type": "between_tools"}
                    create_kwargs["output_config"] = {"effort": "low"}
                response = self.client.messages.create(**create_kwargs)
                
                response_text = "".join(
                    b.text for b in response.content if getattr(b, "type", "") == "text"
                ).strip()
                
                # Remove markdown code blocks
                if response_text.startswith('```'):
                    response_text = response_text.split('```')[1]
                    if response_text.startswith('json'):
                        response_text = response_text[4:]
                    response_text = response_text.strip()
                
                assessment_data = json.loads(response_text)
                
                return RelevanceAssessment(
                    paper_id=paper['source_id'],
                    chapter_file=chapter['filename'],
                    relevance_score=assessment_data['relevance_score'],
                    confidence=assessment_data['confidence'],
                    reasoning=assessment_data['reasoning'],
                    should_include=assessment_data['should_include'],
                    suggested_section=assessment_data.get('suggested_section'),
                    key_contributions=assessment_data.get('key_contributions'),
                    impact_score=impact_score
                )
                
            except (RateLimitError, APIConnectionError, APIError) as e:
                if isinstance(e, RateLimitError):
                    self.rate_limit_hits += 1
                self.api_errors += 1
                logger.warning(
                    f"API error (attempt {attempt + 1}/{self.max_retries}): "
                    f"{type(e).__name__}: {e} (cause={e.__cause__!r})"
                )
                if attempt == self.max_retries - 1:
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                if attempt == self.max_retries - 1:
                    return None
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt == self.max_retries - 1:
                    return None
        
        return None
    
    def assess_all_papers(
        self,
        papers: List[Dict],
        chapters_dir: Path,
        threshold: float = 0.75,  # Higher default threshold
        max_papers_per_chapter: int = 15  # Fewer papers
    ) -> Dict[str, List[RelevanceAssessment]]:
        """
        Assess papers with ultra-selective filtering.
        
        Args:
            papers: List of papers
            chapters_dir: Directory with chapters
            threshold: Minimum relevance score (default: 0.75, very selective)
            max_papers_per_chapter: Max papers to assess per chapter (default: 15)
        """
        if not papers:
            logger.warning("No papers to assess")
            return {}
        
        # Load chapters
        chapter_files = sorted(chapters_dir.glob("*.md"))
        chapters = [self._load_chapter_metadata(f) for f in chapter_files]
        
        logger.info(f"Found {len(chapters)} chapters")
        
        if not chapters:
            logger.error(f"No chapter files found in {chapters_dir}")
            return {}
        
        chapter_assessments: Dict[str, List[RelevanceAssessment]] = {
            ch['filename']: [] for ch in chapters
        }
        
        # Process each chapter
        for idx, chapter in enumerate(chapters, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Chapter {idx}/{len(chapters)}: {chapter['title']}")
            logger.info(f"{'='*80}")
            
            # Ultra-selective filtering
            logger.info(f"Filtering {len(papers)} papers for groundbreaking contributions...")
            filtered_papers = self.filter.filter_papers_for_chapter(
                papers,
                chapter,
                impact_threshold=self.impact_threshold,
                max_papers=max_papers_per_chapter
            )
            
            filtered_count = len(papers) - len(filtered_papers)
            self.papers_filtered_out += filtered_count
            
            logger.info(
                f"✂️ Filtered out {filtered_count} papers "
                f"({filtered_count/len(papers)*100:.1f}%)"
            )
            logger.info(
                f"📋 Assessing top {len(filtered_papers)} groundbreaking papers"
            )
            
            # Assess each filtered paper
            assessed = 0
            relevant = 0
            
            for paper_idx, (paper, impact_score) in enumerate(filtered_papers, 1):
                try:
                    journal_info = f" [{paper.get('journal', 'N/A')}]" if paper.get('journal') else ""
                    citation_info = f" ({paper.get('citations')} citations)" if paper.get('citations') else ""
                    logger.info(
                        f"  [{paper_idx}/{len(filtered_papers)}] "
                        f"Impact:{impact_score:.1f} "
                        f"{paper['title'][:50]}...{journal_info}{citation_info}"
                    )
                    
                    assessment = self.assess_paper_chapter_relevance(paper, chapter, impact_score)
                    assessed += 1
                    self.papers_assessed += 1
                    
                    if assessment and assessment.relevance_score >= threshold:
                        chapter_assessments[chapter['filename']].append(assessment)
                        relevant += 1
                        logger.info(
                            f"    ✓ ACCEPTED! Score: {assessment.relevance_score:.2f} "
                            f"(Impact: {impact_score:.1f})"
                        )
                        logger.info(f"       Reason: {assessment.reasoning[:100]}...")
                    elif assessment:
                        logger.info(
                            f"    ✗ Rejected (score: {assessment.relevance_score:.2f})"
                        )
                    else:
                        logger.warning(f"    ⚠ Assessment failed")
                        
                except KeyboardInterrupt:
                    logger.warning("\nInterrupted by user")
                    self._print_statistics()
                    raise
                except Exception as e:
                    logger.error(f"  Error: {e}")
                    continue
            
            logger.info(f"\n📊 Chapter summary: {relevant}/{assessed} papers meet threshold")
            
            if idx % 5 == 0:
                self._print_statistics()
        
        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info("ULTRA-SELECTIVE ASSESSMENT COMPLETE")
        logger.info(f"{'='*80}")
        
        total_relevant = sum(len(assessments) for assessments in chapter_assessments.values())
        logger.info(f"Total groundbreaking papers included: {total_relevant}")
        logger.info(f"Papers filtered out before assessment: {self.papers_filtered_out}")
        logger.info(f"Filtering efficiency: {self.papers_filtered_out/(self.papers_filtered_out + self.papers_assessed)*100:.1f}%")
        
        for chapter_file, assessments in chapter_assessments.items():
            if assessments:
                # Sort by relevance score
                assessments.sort(key=lambda a: a.relevance_score, reverse=True)
                avg_score = sum(a.relevance_score for a in assessments) / len(assessments)
                logger.info(
                    f"  {chapter_file}: {len(assessments)} papers "
                    f"(avg score: {avg_score:.2f})"
                )
        
        self._print_statistics()
        
        return chapter_assessments
    
    def _print_statistics(self):
        """Print comprehensive statistics."""
        logger.info(f"\n📈 Statistics:")
        logger.info(f"  Papers filtered out: {self.papers_filtered_out}")
        logger.info(f"  Papers assessed via API: {self.papers_assessed}")
        logger.info(f"  API calls: {self.api_calls}")
        logger.info(f"  API errors: {self.api_errors}")
        logger.info(f"  Rate limit hits: {self.rate_limit_hits}")
        if self.api_calls > 0:
            success_rate = ((self.api_calls - self.api_errors) / self.api_calls) * 100
            logger.info(f"  Success rate: {success_rate:.1f}%")
        if self.papers_assessed + self.papers_filtered_out > 0:
            total = self.papers_assessed + self.papers_filtered_out
            api_reduction = (self.papers_filtered_out / total) * 100
            logger.info(f"  API call reduction: {api_reduction:.1f}%")
    
    def save_assessments(
        self,
        assessments: Dict[str, List[RelevanceAssessment]],
        papers: List[Dict],
        output_file: Path
    ):
        """Save assessments with impact scores."""
        paper_lookup = {p['source_id']: p for p in papers}
        
        output_data = {
            'assessment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'assessment_type': 'ultra_selective_groundbreaking',
            'total_papers_assessed': self.papers_assessed,
            'total_papers_filtered': self.papers_filtered_out,
            'impact_threshold': self.impact_threshold,
            'api_statistics': {
                'total_calls': self.api_calls,
                'errors': self.api_errors,
                'rate_limit_hits': self.rate_limit_hits
            },
            'chapters': {}
        }
        
        for chapter_file, chapter_assessments in assessments.items():
            if not chapter_assessments:
                continue
            
            output_data['chapters'][chapter_file] = {
                'relevant_papers_count': len(chapter_assessments),
                'average_relevance_score': sum(a.relevance_score for a in chapter_assessments) / len(chapter_assessments),
                'average_impact_score': sum(a.impact_score or 0 for a in chapter_assessments) / len(chapter_assessments),
                'papers': []
            }
            
            for assessment in chapter_assessments:
                paper = paper_lookup.get(assessment.paper_id)
                if paper:
                    paper_with_assessment = {
                        'paper': paper,
                        'assessment': assessment.to_dict()
                    }
                    output_data['chapters'][chapter_file]['papers'].append(paper_with_assessment)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"\n💾 Saved assessments to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ultra-selective assessment for groundbreaking papers only"
    )
    parser.add_argument('--input-dir', type=str, default='data/literature')
    parser.add_argument('--chapters-dir', type=str, default='_chapters')
    parser.add_argument('--output-file', type=str, default='data/groundbreaking_papers.json')
    parser.add_argument('--threshold', type=float, default=0.75,
                       help='Minimum relevance score (default: 0.75, very selective)')
    parser.add_argument('--impact-threshold', type=float, default=55.0,
                       help='Minimum impact score for pre-filtering (default: 55/100)')
    parser.add_argument('--min-citations', type=int, default=10,
                       help='Minimum citations for non-tier1 journals (default: 10)')
    parser.add_argument('--max-papers-per-chapter', type=int, default=15,
                       help='Maximum papers to assess per chapter (default: 15)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test with limited papers and chapters')
    parser.add_argument('--base-delay', type=float, default=2.0)
    parser.add_argument('--max-retries', type=int, default=5)
    
    args = parser.parse_args()
    
    # Validate directories
    chapters_dir = Path(args.chapters_dir)
    if not chapters_dir.exists():
        logger.error(f"Chapters directory not found: {chapters_dir}")
        return 1
    
    chapter_files = list(chapters_dir.glob("*.md"))
    if not chapter_files:
        logger.error(f"No markdown files found in {chapters_dir}")
        return 1
    
    logger.info(f"Found {len(chapter_files)} chapter files")
    
    # Load papers
    input_file = Path(args.input_dir) / 'papers.json'
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    with open(input_file, 'r') as f:
        data = json.load(f)
        papers = data.get('papers', [])
    
    if not papers:
        logger.error("No papers found")
        return 1
    
    logger.info(f"Loaded {len(papers)} papers")
    
    # Test mode
    if args.test_mode:
        logger.info("\n" + "="*80)
        logger.info("RUNNING IN TEST MODE")
        logger.info("="*80)
        papers = papers[:10]
        logger.info(f"Limited to first 10 papers")
    
    # Initialize assessor
    try:
        assessor = UltraSelectiveRelevanceAssessor(
            base_delay=args.base_delay,
            max_retries=args.max_retries,
            impact_threshold=args.impact_threshold,
            min_citations=args.min_citations
        )
    except ValueError as e:
        logger.error(f"Failed to initialize: {e}")
        return 1
    
    # Assess
    try:
        if args.test_mode:
            import tempfile
            import shutil
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_chapters_dir = Path(tmpdir) / "chapters"
                tmp_chapters_dir.mkdir()
                for chapter_file in sorted(chapters_dir.glob("*.md"))[:2]:
                    shutil.copy(chapter_file, tmp_chapters_dir / chapter_file.name)
                
                assessments = assessor.assess_all_papers(
                    papers=papers,
                    chapters_dir=tmp_chapters_dir,
                    threshold=args.threshold,
                    max_papers_per_chapter=args.max_papers_per_chapter
                )
        else:
            assessments = assessor.assess_all_papers(
                papers=papers,
                chapters_dir=chapters_dir,
                threshold=args.threshold,
                max_papers_per_chapter=args.max_papers_per_chapter
            )
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during assessment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save results
    try:
        output_file = Path(args.output_file)
        assessor.save_assessments(assessments, papers, output_file)
    except Exception as e:
        logger.error(f"Error saving: {e}")
        return 1
    
    logger.info("\n" + "="*80)
    logger.info("✓ ULTRA-SELECTIVE ASSESSMENT COMPLETE")
    logger.info("="*80)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
