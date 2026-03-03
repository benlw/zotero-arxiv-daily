from .protocol import Paper
import math


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <meta charset="UTF-8">
</head>
<body style="margin:0;padding:0;background:#f5f7fb;">
  <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="background:#f5f7fb;padding:12px 0;">
    <tr>
      <td align="center">
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="680" style="width:680px;max-width:680px;background:#ffffff;border:1px solid #e8edf3;border-radius:10px;">
          <tr>
            <td style="padding:16px 14px 6px 14px;font-family:Arial,Helvetica,sans-serif;color:#1f2937;font-size:20px;font-weight:700;">arXiv Daily Digest</td>
          </tr>
          <tr>
            <td style="padding:0 14px 14px 14px;font-family:Arial,Helvetica,sans-serif;color:#4b5563;font-size:13px;line-height:1.6;">__CONTENT__</td>
          </tr>
          <tr>
            <td style="padding:10px 14px 16px 14px;font-family:Arial,Helvetica,sans-serif;color:#9aa3af;font-size:12px;border-top:1px solid #eef2f7;">To unsubscribe, remove your email in your Github Action setting.</td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
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


def get_block_html(title:str, authors:str, rate:str, tldr:str, pdf_url:str, affiliations:str=None, code_url:str|None=None):
    block_template = """
    <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family:Arial,Helvetica,sans-serif;border:1px solid #e6ebf2;border-radius:8px;background:#fbfcfe;margin:0 0 12px 0;">
      <tr>
        <td style="padding:12px 12px 6px 12px;font-size:18px;line-height:1.35;font-weight:700;color:#1f2937;">{title}</td>
      </tr>
      <tr>
        <td style="padding:0 12px 6px 12px;font-size:13px;line-height:1.65;color:#4b5563;">{authors}<br><i>{affiliations}</i></td>
      </tr>
      <tr>
        <td style="padding:0 12px 8px 12px;font-size:13px;color:#374151;"><strong>Relevance:</strong> {rate}</td>
      </tr>
      <tr>
        <td style="padding:0 12px 10px 12px;font-size:14px;line-height:1.75;color:#111827;"><strong>TL;DR</strong><br>{tldr}</td>
      </tr>
      <tr>
        <td style="padding:0 12px 12px 12px;">
          <a href="{pdf_url}" style="display:inline-block;text-decoration:none;font-size:13px;font-weight:700;color:#fff;background:#d9534f;padding:7px 12px;border-radius:4px;margin-right:8px;">PDF</a>
          {code_link}
        </td>
      </tr>
    </table>
"""
    code_link = ""
    if code_url:
        code_link = f'<a href="{code_url}" style="display:inline-block;text-decoration:none;font-size:13px;font-weight:700;color:#fff;background:#3b82f6;padding:7px 12px;border-radius:4px;">Code</a>'

    return block_template.format(
        title=title,
        authors=authors,
        rate=rate,
        tldr=_nl2br(tldr),
        pdf_url=pdf_url,
        affiliations=affiliations,
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
    top_k_highlights:int=3,
) -> str:
    parts = []
    parts.append(
        '<div style="font-size:13px;line-height:1.65;color:#4b5563;margin:4px 0 12px 0;">'
        f'<strong>总论文数：</strong>{len(papers)} &nbsp;|&nbsp; '
        f'<strong>重点：</strong>{min(len(papers), max(0, top_k_highlights))}'
        '</div>'
    )
    if arxiv_categories:
        parts.append(
            '<div style="font-size:13px;line-height:1.65;color:#4b5563;margin:0 0 12px 0;">'
            '<strong>arXiv Categories:</strong> ' + ', '.join(arxiv_categories) +
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
        return get_block_html(p.title, authors, rate, p.tldr, p.pdf_url, affiliations, p.code_url)

    highlights = papers[:max(0, top_k_highlights)]
    others = papers[max(0, top_k_highlights):]

    if highlights:
        parts.append('<h3 style="font-family:Arial,Helvetica,sans-serif;color:#111827;font-size:16px;margin:8px 0;">今日重点论文</h3>')
        parts.extend([_paper_card(p) for p in highlights])

    if others:
        parts.append(f'<h3 style="font-family:Arial,Helvetica,sans-serif;color:#111827;font-size:16px;margin:8px 0;">其余论文（{len(others)} 篇）</h3>')
        parts.extend([_paper_card(p) for p in others])

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
