from .protocol import Paper
import math
import re


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def _nl2br(text:str|None) -> str:
    if not text:
        return ""
    return text.replace("\n", "<br>")


def _tokenize_keywords(text:str|None) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)]


def _relevance_reason(paper:Paper, interest_profile:str|None) -> str:
    if not interest_profile:
        return "与您的近期研究兴趣存在方法或应用层面的交叉，具备跟进价值。"
    interest = set(_tokenize_keywords(interest_profile))
    paper_text = " ".join([paper.title or "", paper.abstract or ""]).lower()
    hits = [k for k in interest if k in paper_text]
    if hits:
        shown = ", ".join(sorted(hits)[:4])
        return f"关键词与您的兴趣画像重合（{shown}），因此该文与现有研究路径相关。"
    return "该文虽非同主题直击，但在方法论或问题设定上对您的方向有借鉴意义。"


def get_block_html(title:str, authors:str, rate:str, tldr:str, pdf_url:str, affiliations:str=None, relevance_reason:str|None=None, code_url:str|None=None):
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
            <br>
            <strong>Why relevant:</strong> {relevance_reason}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0; line-height: 1.6;">
            <strong>TLDR:</strong><br>{tldr}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px; margin-right: 8px;">PDF</a>
            {code_link}
        </td>
    </tr>
</table>
"""
    code_link = ""
    if code_url:
        code_link = f'<a href="{code_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #5bc0de; padding: 8px 16px; border-radius: 4px;">Code</a>'

    return block_template.format(
        title=title,
        authors=authors,
        rate=rate,
        tldr=_nl2br(tldr),
        pdf_url=pdf_url,
        affiliations=affiliations,
        relevance_reason=relevance_reason or "",
        code_link=code_link,
    )

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(
    papers:list[Paper],
    arxiv_categories:list[str]|None=None,
    interest_profile:str|None=None,
    top_k_highlights:int=3,
) -> str:
    parts = []
    if arxiv_categories:
        parts.append(
            '<div style="font-family: Arial, sans-serif; font-size: 14px; color: #444; margin: 8px 0 16px 0;">'
            '<strong>arXiv Categories Used:</strong> ' + ', '.join(arxiv_categories) +
            '</div>'
        )

    if len(papers) == 0 :
        parts.append(get_empty_html())
        return framework.replace('__CONTENT__', ''.join(parts))
    
    def _paper_card(p:Paper) -> str:
        #rate = get_stars(p.score)
        rate = round(p.score, 1) if p.score is not None else 'Unknown'
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        relevance_reason = _relevance_reason(p, interest_profile)
        return get_block_html(p.title, authors, rate, p.tldr, p.pdf_url, affiliations, relevance_reason, p.code_url)

    highlights = papers[:max(0, top_k_highlights)]
    others = papers[max(0, top_k_highlights):]

    if highlights:
        parts.append('<h3 style="font-family: Arial, sans-serif; color:#222;">今日重点论文</h3>')
        parts.extend([_paper_card(p) for p in highlights])

    if others:
        parts.append(
            f'<details style="font-family: Arial, sans-serif;"><summary style="cursor:pointer; font-weight:bold;">其余论文（{len(others)} 篇，点击展开）</summary><br>'
            + '</br><br>'.join([_paper_card(p) for p in others])
            + '</details>'
        )

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
