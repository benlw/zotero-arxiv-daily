from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    code_url: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        lang_lower = str(lang).lower()
        zh_mode = any(k in lang_lower for k in ["chinese", "中文", "zh"])
        deep_mode = bool(self.full_text)

        if zh_mode and deep_mode:
            prompt = (
                "请基于给定论文信息，输出面向“有数学/工程背景但未接触该具体领域”的研究者的学术中文导读。"
                "请严格按以下结构输出，并控制总长度在 600~800 中文字：\n"
                "TL;DR：4~6句，说明论文在做什么、主要思路与结果概貌。\n"
                "Q1（核心科学问题与难点）：6~10句，必须给出结论性陈述，不要写反问句。\n"
                "Q2（理论基础与推进）：6~10句，必须给出结论性陈述，不要写反问句。\n"
                "要求：术语准确、逻辑紧凑、避免空话，不要编造未给出的实验细节与数字。\n"
                "术语优先映射（尽量使用以下中文）：\n"
                "Mechanical systems→力学系统；Mechanical connection→力学联络；Fully actuated/Underactuated→全驱动/欠驱动；\n"
                "Holonomic/Nonholonomic→完整/非完整；Lagrange--d'Alembert Principle→Lagrange--d'Alembert原理；\n"
                "Hamilton's principle→Hamilton原理；Hamilton--Pontryagin Principle→Hamilton--Pontryagin原理；Hamel's formalism→Hamel形式；\n"
                "Euler--Lagrange Equations→Euler--Lagrange方程；Constraint distribution→约束分布；Virtual displacement/variation→虚位移/变分；\n"
                "Stabilization→镇定；Control law→控制律；Lyapunov criterion→Lyapunov判据；Moving frame→活动标架；\n"
                "Distributed moving frame→分布式活动标架；Frame operators→标架算子；Structure-preserving→保结构；\n"
                "Variational integrator→变分积分子；Diffeomorphism→微分同胚；Lagrangian density→Lagrange 密度；\n"
                "Artificial viscosity→人工粘性；Nonmaterial velocity→非物质速度。\n\n"
            )
        elif zh_mode:
            prompt = (
                "请基于论文标题与摘要，输出中文 TL;DR。"
                "要求：4~6句，总长度 220~380 中文字，术语准确、逻辑紧凑、避免空话。"
                "仅输出 TL;DR 内容，不要输出 Q1/Q2 标题或分节。\n\n"
            )
        else:
            prompt = (
                f"Given the following paper information, generate a concise summary in {lang}.\n"
                "If full text is available, include: TL;DR, Q1, Q2.\n"
                "If only abstract is available, output TL;DR only.\n\n"
            )

        if self.title:
            prompt += f"Title:\n {self.title}\n\n"
        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"
        elif self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"
        else:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a rigorous scientific writing assistant. Keep outputs precise and faithful to evidence. Use {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr
    
    def _is_tldr_valid(self, tldr:str) -> bool:
        text = (tldr or "").strip()
        if not text:
            return False
        deep_mode = bool(self.full_text)
        if deep_mode:
            has_sections = ("Q1" in text) and ("Q2" in text)
            # reject question-only placeholders
            bad_q1 = bool(re.search(r"Q1[^\n]{0,80}[?？]", text))
            bad_q2 = bool(re.search(r"Q2[^\n]{0,80}[?？]", text))
            return has_sections and (not bad_q1) and (not bad_q2)
        # abstract-only: should be concise and without Q1/Q2 sections
        if "Q1" in text or "Q2" in text:
            return False
        return 180 <= len(text) <= 500

    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            max_attempts = 2
            tldr = None
            for i in range(max_attempts):
                tldr = self._generate_tldr_with_llm(openai_client,llm_params)
                if self._is_tldr_valid(tldr):
                    break
                logger.warning(f"TLDR quality check failed ({i+1}/{max_attempts}) for {self.url}, retrying...")
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]