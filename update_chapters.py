#!/usr/bin/env python3
"""
Chapter Update Script

This script uses Claude to integrate new research findings into textbook chapters
while maintaining academic rigor, proper citations, and production-ready code examples.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import time

import anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChapterUpdater:
    """Update chapters with new research findings."""
    
    def __init__(self, api_key: str):
        """
        Initialize chapter updater.
        
        Args:
            api_key: Anthropic API key
        """
        self.client = anthropic.Anthropic(api_key=api_key.strip() if api_key else api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    
    def generate_update_suggestions(
        self,
        chapter_id: str,
        chapter_content: str,
        relevant_papers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate suggestions for integrating new papers into a chapter.
        
        Args:
            chapter_id: Chapter identifier
            chapter_content: Current chapter content
            relevant_papers: List of relevant paper analyses
            
        Returns:
            Update suggestions dictionary
        """
        # Limit chapter content for context window
        chapter_preview = chapter_content[:15000] if len(chapter_content) > 15000 else chapter_content
        
        # Format papers for prompt
        papers_text = "\n\n".join([
            f"**Paper {i+1}:**\n"
            f"Title: {p['paper_title']}\n"
            f"Relevance Score: {p['relevance_score']}/10\n"
            f"Key Contributions: {', '.join(p.get('key_contributions', []))}\n"
            f"Methods/Models: {', '.join(p.get('methods_or_models', []))}\n"
            f"Has Equity Focus: {p.get('has_equity_focus', False)}\n"
            f"Is SOTA: {p.get('is_sota', False)}\n"
            f"Citation: {p.get('citation_text', 'N/A')}\n"
            f"Integration Suggestion: {p.get('integration_suggestion', 'N/A')}"
            for i, p in enumerate(relevant_papers[:10])  # Limit to top 10
        ])
        
        prompt = f"""You are an expert in healthcare AI, specializing in developing equitable AI systems for underserved populations.

**TASK:** Review the following chapter and new research papers, then provide structured suggestions for how to integrate the most important new findings.

**CHAPTER ID:** {chapter_id}

**CURRENT CHAPTER CONTENT (preview):**
```markdown
{chapter_preview}
```

**NEW RESEARCH PAPERS:**
{papers_text}

**INSTRUCTIONS:**
Analyze these papers and provide integration suggestions. Return a JSON object with:

1. "priority_updates" (list): Top 3-5 papers that MUST be integrated (highest relevance and impact)
   Each item should have:
   - "paper_title": string
   - "rationale": why this is critical to add
   - "suggested_section": where in chapter to add
   - "integration_approach": "new_section" | "expand_existing" | "update_code_example" | "add_citation"
   
2. "suggested_additions" (list): Specific text additions/modifications
   Each item should have:
   - "section": section name or line number range
   - "addition_type": "new_paragraph" | "updated_code" | "new_citation" | "new_subsection"
   - "content": the actual text to add (for citations) or description of what to add
   - "paper_ids": list of paper titles this relates to

3. "bibliography_additions" (list): New citations in JMLR format
   - "citation_key": string (e.g., "author2024title")
   - "citation_text": full JMLR formatted citation
   - "paper_title": for reference

4. "code_updates_needed" (boolean): Are there new methods/models that need code examples?

5. "major_developments" (list of strings): Any paradigm shifts or major advances in the field

6. "equity_implications" (string): How do these papers advance or challenge equity considerations in this area?

Prioritize:
- Papers with explicit health equity focus
- State-of-the-art methods that improve upon existing approaches in the chapter
- Highly cited papers from top venues
- Papers with production implications

Respond ONLY with valid JSON."""

        try:
            create_kwargs = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self.model.startswith("claude-sonnet-5-5"):
                create_kwargs["thinking"] = {"type": "between_tools"}
                create_kwargs["output_config"] = {"effort": "low"}
            message = self.client.messages.create(**create_kwargs)
            
            response_text = "".join(
                b.text for b in message.content if getattr(b, "type", "") == "text"
            ).strip()
            
            # Clean up response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            suggestions = json.loads(response_text.strip())
            suggestions['chapter_id'] = chapter_id
            suggestions['generated_date'] = time.strftime('%Y-%m-%d')
            
            return suggestions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response: {e}")
            logger.error(f"Response: {response_text[:500]}")
            return {'error': f'JSON parsing error: {str(e)}'}
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return {'error': str(e)}
    
    def apply_updates_to_chapter(
        self,
        chapter_path: Path,
        suggestions: Dict[str, Any],
        dry_run: bool = True
    ) -> bool:
        """
        Apply update suggestions to a chapter file.
        
        Args:
            chapter_path: Path to chapter markdown file
            suggestions: Update suggestions from generate_update_suggestions
            dry_run: If True, don't actually modify files
            
        Returns:
            True if successful
        """
        if 'error' in suggestions:
            logger.error(f"Cannot apply updates due to error: {suggestions['error']}")
            return False
        
        if not chapter_path.exists():
            logger.warning(f"Chapter file not found: {chapter_path}")
            return False
        
        # Read current content
        with open(chapter_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Generate update summary
        priority_count = len(suggestions.get('priority_updates', []))
        citation_count = len(suggestions.get('bibliography_additions', []))
        
        summary = f"""
# Automated Update Summary for {chapter_path.name}
Generated: {suggestions.get('generated_date')}

## Priority Updates: {priority_count}
{chr(10).join(f"- {p['paper_title']}: {p['rationale']}" for p in suggestions.get('priority_updates', []))}

## New Citations to Add: {citation_count}

## Major Developments:
{chr(10).join(f"- {dev}" for dev in suggestions.get('major_developments', []))}

## Equity Implications:
{suggestions.get('equity_implications', 'None noted')}

## Bibliography Additions:
{chr(10).join(f"- {c['citation_key']}: {c['paper_title']}" for c in suggestions.get('bibliography_additions', []))}

---
*This update was automatically generated. Manual review recommended.*
"""
        
        if dry_run:
            logger.info(f"DRY RUN - Would update {chapter_path.name}")
            logger.info(summary)
            
            # Save suggestions to file
            suggestions_file = chapter_path.parent / f"{chapter_path.stem}_update_suggestions.json"
            with open(suggestions_file, 'w') as f:
                json.dump(suggestions, f, indent=2)
            logger.info(f"Saved suggestions to {suggestions_file}")
            
            # Save summary
            summary_file = chapter_path.parent / f"{chapter_path.stem}_update_summary.md"
            with open(summary_file, 'w') as f:
                f.write(summary)
            logger.info(f"Saved summary to {summary_file}")
            
        else:
            # Actually apply updates (would need more sophisticated logic)
            logger.warning("Actual file modification not implemented - manual review required")
            logger.warning("Review the _update_suggestions.json and _update_summary.md files")
        
        return True
    
    def update_all_chapters(
        self,
        analyses_dir: Path,
        chapters_dir: Path,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Process all chapter updates.
        
        Args:
            analyses_dir: Directory containing paper analyses
            chapters_dir: Directory containing chapter markdown files
            dry_run: If True, don't actually modify files
            
        Returns:
            Dictionary of update results
        """
        # Find all analysis files
        analysis_files = list(analyses_dir.glob("chapter_*_analysis.json"))
        
        if not analysis_files:
            logger.error(f"No analysis files found in {analyses_dir}")
            return {}
        
        logger.info(f"Found {len(analysis_files)} chapter analyses")
        
        results = {}
        
        for analysis_file in analysis_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {analysis_file.name}")
            logger.info(f"{'='*60}")
            
            # Load analysis
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            chapter_id = analysis_data['chapter_id']
            relevant_papers = analysis_data.get('analyses', [])
            
            if not relevant_papers:
                logger.info(f"No relevant papers for {chapter_id}")
                continue
            
            logger.info(f"Found {len(relevant_papers)} relevant papers")
            
            # Find chapter file
            chapter_file = chapters_dir / f"{chapter_id}.md"
            if not chapter_file.exists():
                logger.warning(f"Chapter file not found: {chapter_file}")
                continue
            
            # Read chapter content
            with open(chapter_file, 'r', encoding='utf-8') as f:
                chapter_content = f.read()
            
            logger.info(f"Generating update suggestions...")
            suggestions = self.generate_update_suggestions(
                chapter_id=chapter_id,
                chapter_content=chapter_content,
                relevant_papers=relevant_papers
            )
            
            # Apply updates
            success = self.apply_updates_to_chapter(
                chapter_path=chapter_file,
                suggestions=suggestions,
                dry_run=dry_run
            )
            
            results[chapter_id] = {
                'success': success,
                'papers_analyzed': len(relevant_papers),
                'priority_updates': len(suggestions.get('priority_updates', [])),
                'new_citations': len(suggestions.get('bibliography_additions', []))
            }
            
            # Rate limiting
            time.sleep(2)
        
        return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Update chapters with new research findings"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key"
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/analysis"),
        help="Directory containing paper analyses"
    )
    parser.add_argument(
        "--chapters-dir",
        type=Path,
        default=Path("chapters"),
        help="Directory containing chapter markdown files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Don't actually modify files (default: True)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (overrides --dry-run)"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        logger.error("Anthropic API key required")
        return 1
    
    dry_run = not args.apply
    
    logger.info("Starting chapter updates")
    logger.info(f"Analysis directory: {args.analysis_dir}")
    logger.info(f"Chapters directory: {args.chapters_dir}")
    logger.info(f"Dry run: {dry_run}")
    
    # Initialize updater
    updater = ChapterUpdater(api_key=args.api_key)
    
    # Update chapters
    results = updater.update_all_chapters(
        analyses_dir=args.analysis_dir,
        chapters_dir=args.chapters_dir,
        dry_run=dry_run
    )
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("UPDATE COMPLETE")
    logger.info(f"{'='*60}")
    
    for chapter_id, result in results.items():
        logger.info(f"\n{chapter_id}:")
        logger.info(f"  Success: {result['success']}")
        logger.info(f"  Papers analyzed: {result['papers_analyzed']}")
        logger.info(f"  Priority updates: {result['priority_updates']}")
        logger.info(f"  New citations: {result['new_citations']}")
    
    total_chapters = len(results)
    total_priority = sum(r['priority_updates'] for r in results.values())
    total_citations = sum(r['new_citations'] for r in results.values())
    
    logger.info(f"\nTotals:")
    logger.info(f"  Chapters processed: {total_chapters}")
    logger.info(f"  Priority updates: {total_priority}")
    logger.info(f"  New citations: {total_citations}")
    
    if dry_run:
        logger.info("\n*** DRY RUN MODE ***")
        logger.info("No files were modified.")
        logger.info("Review _update_suggestions.json and _update_summary.md files.")
        logger.info("Run with --apply to actually modify chapters.")
    
    return 0


if __name__ == "__main__":
    exit(main())
