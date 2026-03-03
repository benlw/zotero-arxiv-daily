from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm
from collections import Counter
import re

class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if not self.config.zotero.include_path:
            return corpus
        new_corpus = []
        logger.info(f"Selecting zotero papers matching include_path: {self.config.zotero.include_path}")
        for c in corpus:
            match_results = [glob_match(p, self.config.zotero.include_path) for p in c.paths]
            if any(match_results):
                new_corpus.append(c)
        samples = random.sample(new_corpus, min(5, len(new_corpus)))
        samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
        logger.info(f"Selected {len(new_corpus)} zotero papers:\n{samples}\n...")
        return new_corpus

    def infer_arxiv_categories(self, corpus:list[CorpusPaper]) -> list[str]:
        keyword_to_categories = {
            # geometric mechanics / control
            "geometric": ["math.OC", "eess.SY", "cs.RO"],
            "lagrangian": ["math.OC", "eess.SY"],
            "hamilton": ["math.OC", "math.DS"],
            "integrator": ["math.NA", "math.OC"],
            "stabilization": ["math.OC", "eess.SY"],
            "nonlinear control": ["eess.SY", "math.OC"],
            # robotics / multibody / soft
            "robot": ["cs.RO"],
            "robotics": ["cs.RO"],
            "soft robot": ["cs.RO"],
            "multibody": ["cs.RO", "eess.SY"],
            "flexible": ["cs.RO", "eess.SY"],
            "swarm": ["cs.MA", "cs.RO", "cs.LG"],
            "multi-agent": ["cs.MA", "cs.LG"],
            # learning
            "physics-informed": ["cs.LG"],
            "reinforcement learning": ["cs.LG", "cs.MA"],
            "learning": ["cs.LG"],
        }

        hint = str(self.config.source.arxiv.get("interest_profile", "") or "")
        text_chunks = [hint.lower()]
        text_chunks.extend([(c.title + " " + c.abstract).lower() for c in corpus])
        full_text = "\n".join(text_chunks)

        scores = Counter()
        for kw, cats in keyword_to_categories.items():
            hit = full_text.count(kw)
            if hit <= 0:
                continue
            for cat in cats:
                scores[cat] += hit

        # Fallback for general CS discovery.
        if not scores:
            return ["cs.AI", "cs.LG", "cs.RO"]

        ranked = [cat for cat, _ in scores.most_common()]
        # Keep top 4 categories for manageable query size.
        return ranked[:4]

    def maybe_autofill_arxiv_categories(self, corpus:list[CorpusPaper]):
        has_arxiv = "arxiv" in self.config.executor.source
        if not has_arxiv:
            return

        required = list(self.config.source.arxiv.get("required_categories", []) or [])
        current = self.config.source.arxiv.get("category", None)
        auto_enabled = bool(self.config.executor.get("auto_category_from_zotero", True))

        # Case 1: user already provided categories; merge required ones.
        if current:
            merged = list(dict.fromkeys(list(current) + required))
            self.config.source.arxiv.category = merged
            logger.info(f"Using configured arXiv categories (with required merged): {merged}")
            return

        # Case 2: auto disabled and no categories.
        if not auto_enabled:
            return

        inferred = self.infer_arxiv_categories(corpus)
        merged = list(dict.fromkeys(inferred + required))
        self.config.source.arxiv.category = merged
        logger.info(f"Auto-inferred arXiv categories from Zotero/profile: {merged}")

    
    def _canonical_key(self, p):
        url = (p.url or "").lower().strip()
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
        if m:
            return f"arxiv:{m.group(1)}"
        title = re.sub(r"\s+", " ", (p.title or "").lower()).strip()
        return f"title:{title}" if title else url

    def deduplicate_papers(self, papers:list):
        if not papers:
            return papers
        kept = {}
        for p in papers:
            key = self._canonical_key(p)
            if key not in kept:
                kept[key] = p
                continue
            old = kept[key]
            old_len = len((old.full_text or old.abstract or ""))
            new_len = len((p.full_text or p.abstract or ""))
            if new_len > old_len:
                kept[key] = p
        removed = len(papers) - len(kept)
        if removed > 0:
            logger.info(f"Deduplicated papers: removed {removed} duplicates")
        return list(kept.values())

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return

        self.maybe_autofill_arxiv_categories(corpus)
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        all_papers = self.deduplicate_papers(all_papers)
        logger.info(f"Total {len(all_papers)} papers after deduplication")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        category_info = None
        if "arxiv" in self.config.executor.source:
            category_info = list(self.config.source.arxiv.get("category", []) or [])
        email_content = render_email(
            reranked_papers,
            arxiv_categories=category_info,
            interest_profile=self.config.source.arxiv.get("interest_profile", None) if "arxiv" in self.config.executor.source else None,
            top_k_highlights=int(self.config.executor.get("highlight_top_k", 3)),
        )

        # In debug mode, skip SMTP side effects to keep CI/test runs stable.
        if bool(self.config.executor.get("debug", False)):
            logger.info("Debug mode enabled: skip sending email.")
            logger.debug(f"Email preview (first 1000 chars): {email_content[:1000]}")
            return

        send_email(self.config, email_content)
        logger.info("Email sent successfully")