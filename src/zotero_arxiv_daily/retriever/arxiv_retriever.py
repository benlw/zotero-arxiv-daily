from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_text_from_arxiv_html
from tempfile import TemporaryDirectory
import feedparser
from urllib.request import urlretrieve
from tqdm import tqdm
import os
from loguru import logger
import re
@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        # Keep retriever usable in direct unit tests / standalone calls.
        # Executor may auto-infer categories, but retriever can also fallback safely.
        categories = self.config.source.arxiv.get("category", None)
        if not categories:
            required = list(self.config.source.arxiv.get("required_categories", []) or [])
            fallback = required or ["math.OC", "eess.SY", "cs.RO", "cs.LG", "cs.MA", "math.DS", "math.DG"]
            self.config.source.arxiv.category = fallback
            logger.info(f"ArxivRetriever fallback categories: {fallback}")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        categories = self.config.source.arxiv.category
        client = arxiv.Client(num_retries=10,delay_seconds=10)
        query = '+'.join(categories)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.get("arxiv_announce_type","new") == 'new']
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        for i in range(0,len(all_paper_ids),20):
            search = arxiv.Search(id_list=all_paper_ids[i:i+20])
            batch = list(client.results(search))
            bar.update(len(batch))
            raw_papers.extend(batch)
        bar.close()

        return raw_papers

    def _extract_code_url(self, raw_paper:ArxivResult) -> str | None:
        candidates = []
        summary = getattr(raw_paper, "summary", "") or ""
        comment = getattr(raw_paper, "comment", "") or ""
        candidates.extend(re.findall(r"https?://[^\s\]\)\}",]+", summary + "\n" + comment))

        for link in getattr(raw_paper, "links", []) or []:
            href = getattr(link, "href", None)
            if href:
                candidates.append(href)

        for u in candidates:
            ul = u.lower()
            if "github.com/" in ul or "gitlab.com/" in ul or "huggingface.co/" in ul or "codeocean.com/" in ul:
                return u.rstrip(').,]')
        return None

    def convert_to_paper(self, raw_paper:ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url

        # Optional full-text extraction.
        # Default OFF to keep CI stable and fast.
        extract_full_text = bool(self.config.executor.get("extract_full_text", False))
        full_text_source = str(self.config.executor.get("full_text_source", "html")).lower()
        full_text = None
        if extract_full_text:
            try:
                if full_text_source == "html":
                    html_url = raw_paper.entry_id.replace("/abs/", "/html/")
                    full_text = extract_text_from_arxiv_html(html_url)
                else:
                    with TemporaryDirectory() as temp_dir:
                        path = os.path.join(temp_dir, "paper.pdf")
                        urlretrieve(pdf_url, path)
                        full_text = extract_markdown_from_pdf(path)
            except Exception as e:
                logger.warning(f"Failed to extract full text ({full_text_source}) of {title}: {e}")
                full_text = None

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
            code_url=self._extract_code_url(raw_paper),
        )